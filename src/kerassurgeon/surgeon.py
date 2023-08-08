import logging
from collections.abc import Sequence
from functools import partial
from typing import cast

import keras.layers as L
import numpy as np
import tensorflow as tf

from kerassurgeon import utils
from kerassurgeon._utils import node as node_utils
from kerassurgeon._utils.layer import make_new_layer as _make_new_layer
from kerassurgeon._utils.tensor_dict import TensorDict
from kerassurgeon.operable_layer import OperableLayerMixin
from kerassurgeon.types import Inputs, Masks, ModificationFunction, Node
from kerassurgeon.utils import find_nodes_in_model, validate_node_indices

# Set up logging
logger = logging.getLogger("kerassurgeon.surgeon")
logger.addHandler(logging.NullHandler())


# pylint: disable=R0912, R0914, R0915
def _apply_delete_mask_to_layer(
    layer: L.Layer,
    inbound_masks: Masks,
    input_shape: tuple[int, ...],
    inputs: Inputs,
    output_shape: tuple[int, ...],
) -> tuple[L.Layer, Masks]:
    """
    Given some upstream changes that yield some input masks,
    apply these to the given layer,
    which might cause changes to the layer (e.g. if input channels are missing)
    or changes to the masks (e.g. because the layer "absorbs" those changes).
    """
    index: list[int] | list[slice] | list[int | slice] | slice
    outbound_mask: np.ndarray | None
    data_format = getattr(layer, 'data_format', 'channels_last')
    channel_axis = -1 if data_format == "channels_last" else 0
    make_new_layer = partial(_make_new_layer, inputs=inputs)
    if isinstance(layer, L.Dense):
        index = [slice(None, 1, None) for _ in inbound_masks.shape[:-1]] + [slice(None)]
        channel_indices = np.where(~inbound_masks[tuple(index)])[-1]

        weights = layer.get_weights()
        channel_axis = -2
        weights[0] = np.delete(weights[0], channel_indices, axis=channel_axis)
        new_layer = make_new_layer(layer, weights=weights)
        outbound_mask = None

    elif isinstance(layer, L.Flatten):
        outbound_mask = np.reshape(inbound_masks, [-1])
        new_layer = layer

    elif isinstance(layer, (L.Conv1D, L.Conv2D, L.Conv3D)):
        if data_format == 'channels_first':
            inbound_masks = np.swapaxes(inbound_masks, 0, -1)
        index = [slice(None, 1, None) for _ in inbound_masks.shape[:-1]] + [slice(None)]
        channel_indices = np.where(~inbound_masks[tuple(index)])[-1]
        weights = layer.get_weights()
        channel_axis = -2
        weights[0] = np.delete(weights[0], channel_indices, axis=channel_axis)

        # Instantiate new layer with new_weights
        new_layer = make_new_layer(layer, weights=weights)
        outbound_mask = None

    elif isinstance(layer, L.Conv2DTranspose):
        if data_format == 'channels_first':
            inbound_masks = np.swapaxes(inbound_masks, 0, -1)
        # Conv layer: trim down inbound_masks to filter shape
        k_size = layer.kernel_size
        index = [slice(None, 1, None) for _ in k_size]
        inbound_masks = inbound_masks[tuple(index + [slice(None)])]
        weights = layer.get_weights()
        # Delete unused weights to obtain new_weights
        # Each deleted channel was connected to all of the channels
        # in layer; therefore, the mask must be repeated for each
        # channel.
        # `delete_mask`'s size: size(weights[0])
        delete_mask = np.tile(
            inbound_masks[..., np.newaxis], list(k_size) + [1, weights[0].shape[-2]]
        ).transpose(0, 1, 3, 2)
        new_shape = list(weights[0].shape)
        new_shape[-1] = -1  # Input size channels
        weights[0] = np.reshape(weights[0][delete_mask], new_shape)
        # Instantiate new layer with new_weights
        new_layer = make_new_layer(layer, weights=weights)
        outbound_mask = None

    elif isinstance(layer, (L.SeparableConv1D, L.SeparableConv2D)):
        if layer.depth_multiplier > 1:
            raise ValueError(
                "Depthwise Convolutions with depth_multiplier > 1 currently not supported"
            )
        if data_format == 'channels_first':
            inbound_masks = np.swapaxes(inbound_masks, 0, -1)

        # channels are last
        index = [slice(None, 1, None) for _ in inbound_masks.shape[:-1]] + [slice(None)]
        channel_indices = np.where(~inbound_masks[tuple(index)])[-1]
        weights = layer.get_weights()
        depthwise_kernel, pointwise_kernel = weights[0], weights[1]
        channel_axis = -2
        weights[0] = np.delete(depthwise_kernel, channel_indices, axis=channel_axis)
        weights[1] = np.delete(pointwise_kernel, channel_indices, axis=channel_axis)

        # Instantiate new layer with new_weights
        new_layer = make_new_layer(layer, weights=weights)
        outbound_mask = None

    elif isinstance(
        layer,
        (
            L.Cropping1D,
            L.Cropping2D,
            L.Cropping3D,
            L.MaxPooling1D,
            L.MaxPooling2D,
            L.MaxPooling3D,
            L.AveragePooling1D,
            L.AveragePooling2D,
            L.AveragePooling3D,
        ),
    ):
        if output_shape is None:
            outbound_mask = None
            new_layer = layer
        else:
            index = [slice(None, x, None) for x in layer.output_shape[1:]]
            index = cast(list[slice], index)
            index[channel_axis] = slice(None)
            outbound_mask = inbound_masks[tuple(index)]
            new_layer = layer

    elif isinstance(
        layer,
        (
            L.UpSampling1D,
            L.UpSampling2D,
            L.UpSampling3D,
            L.ZeroPadding1D,
            L.ZeroPadding2D,
            L.ZeroPadding3D,
        ),
    ):

        # Get slice of mask with all singleton dimensions except
        # channels dimension
        index = [slice(1)] * (len(input_shape) - 1)
        index = cast(list[slice], index)
        tile_shape = list(output_shape[1:])
        index[channel_axis] = slice(None)
        tile_shape[channel_axis] = 1
        channels_vector = inbound_masks[tuple(index)]
        # Tile this slice to create the outbound mask
        outbound_mask = np.tile(channels_vector, tile_shape)
        new_layer = layer

    elif isinstance(
        layer,
        (
            L.GlobalMaxPooling1D,
            L.GlobalMaxPooling2D,
            L.GlobalAveragePooling1D,
            L.GlobalAveragePooling2D,
        ),
    ):
        # Get slice of mask with all singleton dimensions except
        # channels dimension
        index = [0] * (len(input_shape) - 1)
        assert isinstance(index, list)
        index = cast(list[int | slice], index)

        index[channel_axis] = slice(None)
        channels_vector = inbound_masks[tuple(index)]
        # Tile this slice to create the outbound mask
        outbound_mask = channels_vector
        new_layer = layer

    elif isinstance(
        layer,
        (
            L.Dropout,
            L.Activation,
            L.SpatialDropout1D,
            L.SpatialDropout2D,
            L.SpatialDropout3D,
            L.ActivityRegularization,
            L.Masking,
            L.LeakyReLU,
            L.ELU,
            L.ThresholdedReLU,
            L.GaussianNoise,
            L.GaussianDropout,
            L.AlphaDropout,
            L.ReLU,
        ),
    ):
        # Pass-through layers
        outbound_mask = inbound_masks
        new_layer = layer

    elif isinstance(layer, L.Reshape):
        outbound_mask = np.reshape(inbound_masks, layer.target_shape)
        new_layer = layer

    elif isinstance(layer, L.Permute):
        outbound_mask = np.transpose(inbound_masks, [x - 1 for x in layer.dims])
        new_layer = layer

    elif isinstance(layer, L.RepeatVector):
        outbound_mask = np.repeat(np.expand_dims(inbound_masks, 0), layer.n, axis=0)
        new_layer = layer

    elif isinstance(layer, L.Embedding):
        # Embedding will always be the first layer so it doesn't need
        # to consider the inbound_delete_mask
        if inbound_masks is not None:
            raise ValueError(
                'Channels cannot be deleted before Embedding '
                'layers because they change the number of '
                'channels.'
            )
        outbound_mask = None
        new_layer = layer

    elif isinstance(layer, (L.Add, L.Multiply, L.Average, L.Maximum)):
        # The inputs must be the same size
        if not utils.all_equal(inbound_masks):
            raise ValueError(
                f'{layer.__class__.__name__} layers must have the same size inputs. All '
                'inbound nodes must have the same channels deleted'
            )
        outbound_mask = inbound_masks[1]
        new_layer = layer

    elif isinstance(layer, L.Concatenate):
        axis = layer.axis
        if layer.axis < 0:
            axis = axis % len(layer.input_shape[0])
        # Below: axis=axis-1 because the mask excludes the batch dimension
        outbound_mask = np.concatenate(inbound_masks, axis=axis - 1)
        new_layer = layer

    elif isinstance(layer, (L.SimpleRNN, L.GRU, L.LSTM)):
        weights = layer.get_weights()
        weights[0] = weights[0][np.where(inbound_masks[0, :])[0], :]
        new_layer = make_new_layer(layer, weights=weights)
        outbound_mask = None

    elif isinstance(layer, L.BatchNormalization):
        outbound_mask = inbound_masks
        # Get slice of mask with all singleton dimensions except
        # channels dimension
        index = [0] * (len(input_shape))
        assert len(layer.axis) == 1
        first_axis: int = layer.axis[0]
        index[first_axis] = slice(None)  # type: ignore
        index = index[1:]
        # TODO: Maybe use channel indices everywhere instead of masks?
        channel_indices = np.where(~inbound_masks[tuple(index)])[0]
        weights = [np.delete(w, channel_indices, axis=-1) for w in layer.get_weights()]
        new_layer = make_new_layer(layer, weights=weights)

    elif isinstance(layer, L.MultiHeadAttention):
        weights = layer.get_weights()
        query_kernel, query_mask = weights[0], inbound_masks[0]
        key_kernel, key_mask = weights[2], inbound_masks[1]
        value_kernel, value_mask = weights[4], inbound_masks[2]
        weights[0] = query_kernel[np.where(query_mask[0])[0], :]
        weights[2] = key_kernel[np.where(key_mask[0])[0], :]
        weights[4] = value_kernel[np.where(value_mask[0])[0], :]

        config = layer.get_config()
        config['output_shape'] = layer.output_shape[2:]  # retain original output shape
        n_new_query_channels = len(np.where(query_mask[0])[0])
        new_query_shape = tf.TensorShape(
            config['query_shape'].as_list()[:-1] + [n_new_query_channels]
        )
        config['query_shape'] = new_query_shape
        n_new_key_channels = len(np.where(key_mask[0])[0])
        new_key_shape = tf.TensorShape(config['key_shape'].as_list()[:-1] + [n_new_key_channels])
        config['key_shape'] = new_key_shape
        n_new_value_channels = len(np.where(value_mask[0])[0])
        new_value_shape = tf.TensorShape(
            config['value_shape'].as_list()[:-1] + [n_new_value_channels]
        )
        config['value_shape'] = new_value_shape

        new_layer = make_new_layer(layer, config=config, weights=weights)
        outbound_mask = None

    elif isinstance(layer, (L.DepthwiseConv1D, L.DepthwiseConv2D)):
        if layer.depth_multiplier > 1:
            raise ValueError(
                "Depthwise Convolutions with depth_multiplier > 1 currently not supported"
            )
        index = [slice(None, x, None) for x in layer.output_shape[1:]]
        index = cast(list[slice], index)
        index[channel_axis] = slice(None)
        outbound_mask = inbound_masks[tuple(index)]

        if data_format == 'channels_first':
            inbound_masks = np.swapaxes(inbound_masks, 0, -1)

        channel_indices = np.where(~inbound_masks[tuple(index)][-1])[-1]
        weights = layer.get_weights()
        channel_axis = -2
        weights[0] = np.delete(weights[0], channel_indices, axis=channel_axis)  # depthwise kernel
        if len(weights) == 2:
            weights[1] = np.delete(weights[1], channel_indices, axis=-1)  # bias

        # Instantiate new layer with new_weights
        new_layer = make_new_layer(layer, weights=weights)
    elif isinstance(layer, L.Identity):
        new_layer, outbound_mask = layer, inbound_masks

    elif isinstance(layer, OperableLayerMixin):
        new_layer, outbound_mask = layer.apply_delete_mask(
            inbound_masks, input_shape, inputs, output_shape
        )

    else:
        raise ValueError(f'"{layer.__class__.__name__}" layers are currently unsupported.')
    return new_layer, outbound_mask


class Surgeon:
    """Performs network surgery on a model.

    Surgeons can perform multiple network surgeries (jobs) at once.
    This is much faster than performing them sequentially.
    See `add_jobs` for a list of valid jobs and their required keyword arguments.

    Examples:
        Delete some channels from layer_1 and layer_2:
            surgeon = Surgeon(model)
            surgeon.add_job('delete_channels', layer_1, channels_1)
            surgeon.add_job('delete_channels', layer_2, channels_2)
            new_model = surgeon.operate()

    Arguments:
        model: The model to be modified
        copy: If True, the model will be copied before and after any operations
              This keeps the layers in the original model and the new model separate.
    """

    # pylint: disable=R0902
    def __init__(self, model: tf.keras.Model, copy: bool = False):
        if copy:
            self.model = utils.clean_copy(model)
        else:
            self.model = model
        self.nodes: list = []
        self._copy = copy
        self._finished_nodes: dict = {}
        self._replace_tensors = TensorDict()
        self._new_layers_map: dict[tf.keras.layers.Layer, tf.keras.layers.Layer] = {}
        self._replace_layers_map: dict[
            tf.keras.layers.Layer, tuple[tf.keras.layers.Layer, Masks]
        ] = {}
        self._mod_func_map: dict[Node, ModificationFunction] = {}
        self._kwargs_map: dict[Node, dict] = {}
        self.valid_jobs = ('delete_layer', 'insert_layer', 'replace_layer', 'delete_channels')

    def add_job(
        self,
        job: str,
        layer: tf.keras.layers.Layer,
        *,
        channels: list[int] | None = None,
        new_layer: tf.keras.layers.Layer | None = None,
        node_indices: list[int] | None = None,
    ) -> None:
        """Adds a job for the Surgeon to perform on the model.

        Job options are:
        'delete_layer': delete `layer` from the model
                        required keyword arguments: None
        'insert_layer': insert `new_layer` before `layer`
                        required keyword arguments: `new_layer`
        'replace_layer': replace `layer` with `new_layer`
                         required keyword arguments: `new_layer`
        'delete_channels': delete `channels` from `layer`
                           required keyword arguments: `channels`

        Jobs can be added in any order. They will be performed in order of
        decreasing network depth.
        A maximum of one job can be performed per node.

        Args:
            job: job identifier. One of `Surgeon.valid_jobs`.
            layer: A layer from `model` to be modified.
            channels: A list of channels used for the job.
                                 Used in `delete_channels`.
            new_layer: A new layer used for the job. Used in
                              `insert_layer` and `replace_layer`.
            node_indices: (optional) A list of node indices used to
                                    selectively apply the job to a subset of
                                    the layer's nodes. Nodes are selected with:
                                    node[i] = layer.inbound_nodes[node_indices[i]]
        """
        # If the model has been copied, identify `layer` in the copied model.
        if self._copy:
            layer = self.model.get_layer(layer.name)
        # Check that layer is in the model
        if layer not in self.model.layers:
            raise ValueError('layer is not a valid Layer in model.')

        node_indices = validate_node_indices(layer, self.model, node_indices)

        # Select the modification function and any keyword arguments.
        kwargs = {}
        mod_func: ModificationFunction
        match job:
            case 'delete_channels':
                # If not all inbound_nodes are selected, the new layer is renamed
                # to avoid duplicate layer names.
                if set(node_indices) != set(find_nodes_in_model(self.model, layer)):
                    kwargs['layer_name'] = layer.name + '_' + job
                kwargs['channels'] = channels
                mod_func = self._delete_channels
            case 'delete_layer':
                mod_func = self._delete_layer
            case 'insert_layer':
                kwargs['new_layer'] = new_layer
                mod_func = self._insert_layer
            case 'replace_layer':
                kwargs['new_layer'] = new_layer
                mod_func = self._replace_layer
            case _:
                raise ValueError(
                    job + ' is not a recognised job. Valid jobs ' 'are:\n-',
                    '\n- '.join(self.valid_jobs),
                )

        # Get nodes to be operated on for this job
        job_nodes = []
        for node_index in node_indices:
            job_nodes.append(layer.inbound_nodes[node_index])
        # Check that the nodes do not already have jobs assigned to them.
        if set(job_nodes).intersection(self.nodes):
            raise ValueError('Cannot apply several jobs to the same node.')

        # Add the modification function and keyword arguments to the
        # self._mod_func_map and _kwargs_map dictionaries for later retrieval.
        for node in job_nodes:
            self._mod_func_map[node] = mod_func
            self._kwargs_map[node] = kwargs
        self.nodes.extend(job_nodes)

    def operate(self) -> tf.keras.Model:
        """Perform all jobs assigned to the surgeon."""
        # Operate on each node in self.nodes by order of decreasing depth.
        sorted_nodes = sorted(
            self.nodes, reverse=True, key=lambda x: utils.get_node_depth(self.model, x)
        )
        for node in sorted_nodes:
            # Rebuild submodel up to this node
            sub_output_nodes = node_utils.parent_nodes(node)
            outputs, output_masks = self._rebuild_graph(self.model.inputs, sub_output_nodes)

            # Perform surgery at this node
            kwargs = self._kwargs_map[node]
            self._mod_func_map[node](node, outputs, output_masks, **kwargs)

        # Finish rebuilding model
        output_nodes = []
        for output in self.model.outputs:
            # pylint: disable=protected-access
            layer, node_index, _ = output._keras_history
            output_nodes.append(layer.inbound_nodes[node_index])
        new_outputs, _ = self._rebuild_graph(self.model.inputs, output_nodes)
        new_model = tf.keras.Model(self.model.inputs, new_outputs)

        if self._copy:
            return utils.clean_copy(new_model)
        return new_model

    def _rebuild_graph(
        self, graph_inputs, output_nodes, graph_input_masks=None
    ) -> tuple[Inputs, Masks]:
        """Rebuild the graph from graph_inputs to output_nodes.

        This does not return a model object, it re-creates the connections
        between layers and returns the output tensors and masks of the submodel
        This is a building block for the higher level surgery methods.
        See `Surgeon.operate` for details of how this method is used.

        Arguments:
            graph_inputs: List of the submodel's input tensor(s).
            output_nodes(list[Node]): List of the submodel's output node(s)
            graph_input_masks: Boolean mask for each submodel input.

        Returns:
            (tuple) containing :
                List of the output tensors of the rebuilt submodel
                List of the output masks of the rebuilt submodel
            tuple[submodel output tensors, output masks]

        """
        if not graph_input_masks:
            graph_input_masks = [None] * len(graph_inputs)

        def _rebuild_rec(node):
            """Rebuild the graph up to `node` recursively.

            Args:
                node(Node): Node to rebuild up to.
            Returns:
                (tuple) containing :
                The output tensor of the rebuilt `node`
                The output mask of the rebuilt `node`

            """
            # TODO: What happens if nodes have multiple output tensors?
            # Does that ever happen?
            layer = node.outbound_layer
            logger.debug(f"getting inputs for: {layer.name}")
            node_output = utils.single_element(node.output_tensors)
            # First check for conditions to bottom out the recursion
            # Check for replaced tensors before any other checks:
            # these are created by the surgery methods.
            if node_output in self._replace_tensors.keys():
                logger.debug(f"bottomed out at replaced output: {node_output}")
                output, output_mask = self._replace_tensors[node_output]
                return output, output_mask
            # Next check if the current node has already been rebuilt.
            if node in self._finished_nodes:
                # pylint: disable=consider-using-f-string
                logger.debug(
                    "reached finished node: from {} to {}".format(
                        node.inbound_layers.name
                        if isinstance(node.inbound_layers, tf.keras.layers.Layer)
                        else [_layer.name for _layer in node.inbound_layers],
                        node.outbound_layer.name,
                    )
                )
                return self._finished_nodes[node]
            # Next check if one of the graph_inputs has been reached.
            mask_map = TensorDict()
            for graph_input, mask in zip(graph_inputs, graph_input_masks):
                mask_map[graph_input] = mask

            if node_output in mask_map:
                output_mask = mask_map[node_output]
                logger.debug('bottomed out at a model input')
                return node_output, output_mask

            # Otherwise recursively call this method on the inbound nodes.
            inbound_nodes = node_utils.parent_nodes(node)
            logger.debug(f'inbound_layers: {[node.outbound_layer.name for node in inbound_nodes]}')
            # Recursively rebuild the model up to `node`s inbound nodes to
            # obtain its inputs and input masks
            inputs, input_masks = zip(*[_rebuild_rec(n) for n in inbound_nodes])

            logger.debug(
                f'rebuilt model up to: {[node.outbound_layer.name for node in inbound_nodes]}'
            )
            if all(i is None for i in inputs):
                logger.debug(f'No inputs for {layer.name}')
                output = None
                try:
                    assert len(node.output_tensors) <= 1
                except TypeError:
                    # Cannot call length on tensors
                    pass

                output_mask = np.zeros(node.output_tensors.shape[1:], dtype=bool)
            elif any(i is None for i in inputs):
                logger.debug(f'At least one input is missing for {layer.name}')
                if node.outbound_layer.__class__.__name__ != 'Concatenate':
                    raise TypeError('Inputs can only be missing for concatenate layers.')
                # remove Nones from inputs list
                inputs = [i for i in inputs if i is not None]
                new_layer, output_mask = self._apply_delete_mask(node, input_masks, inputs)
                if len(inputs) == 1:
                    output = utils.single_element(list(inputs))
                else:
                    output = new_layer(utils.single_element(list(inputs)))
            else:
                new_layer, output_mask = self._apply_delete_mask(node, input_masks, inputs)
                try:
                    output = new_layer(utils.single_element(list(inputs)))
                except TypeError:
                    # layer expects multiple inputs
                    output = new_layer(*list(map(utils.single_element, inputs)))

            # Record that this node has been rebuilt
            self._finished_nodes[node] = (output, output_mask)
            logger.debug(f"layer complete: {layer.name}")
            return output, output_mask

        # Call the recursive _rebuild_rec method to rebuild the submodel up to
        # each output layer
        outputs, output_masks = zip(*[_rebuild_rec(n) for n in output_nodes])
        return utils.single_element(outputs), output_masks

    def _delete_layer(self, node: Node, inputs: Inputs, input_masks: Masks, **_) -> None:
        """Skip adding node.outbound_layer when building the graph."""
        # Skip the deleted layer by replacing its outputs with it inputs
        if isinstance(inputs, Sequence) and len(inputs) >= 2:
            raise ValueError('Cannot insert new layer at node with multiple inbound layers.')
        inputs = utils.single_element(inputs)
        input_masks = utils.single_element(input_masks)
        deleted_layer_output = utils.single_element(node.output_tensors)
        self._replace_tensors[deleted_layer_output] = (inputs, input_masks)

    def _insert_layer(
        self,
        node: Node,
        inputs: Inputs,
        input_masks: Masks,
        *,
        new_layer: tf.keras.layers.Layer | None = None,
        **_,
    ) -> None:
        """Insert new_layer into the graph before node.outbound_layer."""
        assert new_layer is not None, "new layer must be provided"

        # This will not work for nodes with multiple inbound layers
        if isinstance(inputs, Sequence) and len(inputs) >= 2:
            raise ValueError('Cannot insert new layer at node with multiple inbound layers.')
        # Call the new layer on the inbound layer's output
        new_output = new_layer(utils.single_element(inputs))
        # Replace the inbound layer's output with the new layer's output
        old_output = utils.get_one_tensor(node.input_tensors)
        input_masks = utils.single_element(input_masks)
        self._replace_tensors[old_output] = (new_output, input_masks)

    def _replace_layer(
        self,
        node: Node,
        inputs: Inputs,
        input_masks: Masks,
        *,
        new_layer: tf.keras.layers.Layer | None = None,
        **_,
    ) -> None:
        """Replace node.outbound_layer with new_layer. Add it to the graph."""
        assert new_layer is not None, 'new layer must be provided'

        # Call the new layer on the rebuild submodel's inputs
        new_output = new_layer(utils.single_element(inputs))

        # Replace the original layer's output with the new layer's output
        replaced_layer_output = utils.single_element(node.output_tensors)
        input_masks = utils.single_element(input_masks)
        self._replace_tensors[replaced_layer_output] = (new_output, input_masks)

    # pylint: disable=R0913
    def _delete_channels(
        self,
        node: Node,
        inputs: Inputs,
        input_masks: Masks,
        *,
        channels: list[int] | None = None,
        layer_name: str | None = None,
        **_,
    ) -> None:
        """Delete selected channels of node.outbound_layer. Add it to the graph."""
        assert channels is not None, 'channels must be provided'
        old_layer = node.outbound_layer
        old_layer_output = utils.single_element(node.output_tensors)
        # Create a mask to propagate the deleted channels to downstream layers
        new_delete_mask = self._make_delete_mask(old_layer, channels)

        if len(set(channels)) == getattr(old_layer, utils.get_channels_attr(old_layer)):
            self._replace_tensors[old_layer_output] = (None, new_delete_mask)
            return

        # If this layer has already been operated on, use the cached copy of
        # the new layer. Otherwise, apply the inbound delete mask and
        # delete channels to obtain the new layer
        if old_layer in self._new_layers_map:
            new_layer = self._new_layers_map[old_layer]
        else:
            temp_layer, __ = self._apply_delete_mask(node, input_masks, inputs)
            # This call is needed to initialise input_shape and output_shape
            temp_layer(utils.single_element(inputs))
            new_layer = self._delete_channel_weights(temp_layer, channels, inputs)
            if layer_name is not None:
                new_layer.name = layer_name
            self._new_layers_map[old_layer] = new_layer
        new_output = new_layer(utils.single_element(inputs))
        # Replace the original layer's output with the modified layer's output
        self._replace_tensors[old_layer_output] = (new_output, new_delete_mask)

    def _apply_delete_mask(
        self, node: Node, inbound_masks: Masks, inputs: Inputs
    ) -> tuple[tf.keras.layers.Layer, Masks]:
        """Apply the inbound delete mask and return the outbound delete mask

        When specific channels in a layer or layer instance are deleted, the
        mask propagates information about which channels are affected to
        downstream layers.
        If the layer contains weights, those which were previously connected
        to the deleted channels are deleted and outbound masks are set to None
        since further downstream layers aren't affected.
        If the layer does not contain weights, its output mask is calculated to
        reflect any transformations performed by the layer to ensure that
        information about the deleted channels is propagated downstream.


        Arguments:
            node(Node): The node where the delete mask is applied.
            inbound_masks: Mask(s) from inbound node(s).

        Returns:
            new_layer: Pass through `layer` if it has no weights, otherwise a
                       new `Layer` object with weights corresponding to the
                       inbound mask deleted.
            outbound_mask: Mask corresponding to `new_layer`.
        """

        # if delete_mask is None or all values are True, it does not affect
        # this layer or any layers above/downstream from it
        layer = node.outbound_layer
        if all(mask is None for mask in inbound_masks):
            logger.debug(f'No need to apply delete mask to {layer.name}, as mask is empty')
            new_layer = layer
            return new_layer, None

        # If one or more of the masks are None, replace them with ones.
        if any(mask is None for mask in inbound_masks):
            inbound_masks = [
                np.ones(shape[1:], dtype=bool) if inbound_masks[i] is None else inbound_masks[i]
                for i, shape in enumerate(node.input_shapes)
            ]

        # If the layer is shared and has already been affected by this
        # operation, use the cached new layer.
        if len(layer.inbound_nodes) > 1 and layer in self._replace_layers_map:
            logger.debug(f'Using already replaced layer {layer.name}')
            return self._replace_layers_map[layer]

        output_shape = utils.single_element(node.output_shapes)
        input_shape = utils.single_element(node.input_shapes)
        inbound_masks = utils.single_element(inbound_masks)
        inbound_masks = cast(np.ndarray, inbound_masks)

        if isinstance(layer, L.InputLayer):
            raise RuntimeError('This should never get here!')

        if all(np.all(mask) for mask in inbound_masks):
            # all inbound masks are all true, which equivalents to having no mask at all
            return layer, None

        new_layer, outbound_mask = _apply_delete_mask_to_layer(
            layer, inbound_masks, input_shape, inputs, output_shape
        )

        if len(layer.inbound_nodes) > 1 and new_layer != layer:
            self._replace_layers_map[layer] = (new_layer, outbound_mask)

        return new_layer, outbound_mask

    def _delete_channel_weights(
        self, layer: tf.keras.layers.Layer, channel_indices: list[int], inputs: Inputs
    ) -> tf.keras.layers.Layer:
        """Delete channels from layer and remove the corresponding weights.

        Arguments:
            layer: A layer whose channels are to be deleted
            channel_indices: The indices of the channels to be deleted.

        Returns:
            A new layer with the channels and corresponding weights deleted.
        """
        layer_config = layer.get_config()
        channels_attr = utils.get_channels_attr(layer)
        channel_count = layer_config[channels_attr]
        # Check inputs
        if any(i + 1 > channel_count for i in channel_indices):
            raise ValueError(
                'Channels_index value(s) out of range. '
                f'This layer only has {channel_count} channels.'
            )
        logger.info(
            f'Deleting {len(channel_indices)}/{channel_count} channels from layer: {layer.name}'
        )
        # numpy.delete ignores negative indices in lists: wrap indices
        channel_indices = [i % channel_count for i in channel_indices]

        # Reduce layer channel count in config.
        layer_config[channels_attr] -= len(channel_indices)

        # Delete weights corresponding to deleted channels from config.
        # Except for recurrent layers, the weights' channels dimension is last.
        # Each recurrent layer type has a different internal weights layout.
        if isinstance(layer, L.Wrapper):
            wrapped_layer = self._delete_channel_weights(layer.layer, channel_indices, inputs)
            layer_config['layer']['config'] = wrapped_layer.get_config()
        if isinstance(layer, L.SimpleRNN):
            weights = [np.delete(w, channel_indices, axis=-1) for w in layer.get_weights()]
            weights[1] = np.delete(weights[1], channel_indices, axis=0)
        elif isinstance(layer, L.GRU):
            # Repeat the channel indices for all internal GRU weights.
            channel_indices_gru = [layer.units * m + i for m in range(3) for i in channel_indices]
            weights = [np.delete(w, channel_indices_gru, axis=-1) for w in layer.get_weights()]
            weights[1] = np.delete(weights[1], channel_indices, axis=0)
        elif isinstance(layer, L.LSTM):
            # Repeat the channel indices for all internal LSTM weights.
            channel_indices_lstm = [layer.units * m + i for m in range(4) for i in channel_indices]
            weights = [np.delete(w, channel_indices_lstm, axis=-1) for w in layer.get_weights()]
            weights[1] = np.delete(weights[1], channel_indices, axis=0)
        elif isinstance(layer, L.Conv2DTranspose):
            weights = layer.get_weights()
            weights[0] = np.delete(weights[0], channel_indices, axis=-2)
            if len(weights) == 2:
                weights[1] = np.delete(weights[1], channel_indices, axis=-1)
        else:
            weights = [np.delete(w, channel_indices, axis=-1) for w in layer.get_weights()]
        layer_config['weights'] = weights

        # Create new layer from the modified configuration and return it.
        new_layer = _make_new_layer(layer, inputs, config=layer_config, weights=weights)
        return new_layer

    def _make_delete_mask(
        self, layer: tf.keras.layers.Layer, channel_indices: list[int]
    ) -> np.ndarray:
        """Make the boolean delete mask for layer's output deleting channels.

        The mask is used to remove the weights of the downstream layers which
        were connected to channels which have been deleted in this layer.
        The mask is a boolean array with the same size as the layer output
        excluding the first (batch) dimension.
        All elements of the mask corresponding to the removed channels are set
        to False. Other elements are set to True.

        Arguments:
            layer: A layer
            channel_indices: The indices of the channels to be deleted.

        Returns:
            A Numpy array of booleans of the same size as the output of layer
            excluding the batch dimension.
        """
        data_format = getattr(layer, 'data_format', 'channels_last')
        new_delete_mask = np.ones(layer.output_shape[1:], dtype=bool)
        if data_format == 'channels_first':
            new_delete_mask[channel_indices, ...] = False
        elif data_format == 'channels_last':
            new_delete_mask[..., channel_indices] = False
        else:
            raise ValueError('Invalid data_format property value')
        return new_delete_mask
