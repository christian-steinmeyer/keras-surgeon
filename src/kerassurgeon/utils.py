"""Utilities used across other modules."""
from collections.abc import Collection, Iterable
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.activations import linear

from kerassurgeon._utils import node as node_utils


def clean_copy(model: tf.keras.Model) -> tf.keras.Model:
    """Returns a copy of the model without other model uses of its layers."""
    weights = model.get_weights()
    new_model = model.__class__.from_config(model.get_config())
    new_model.set_weights(weights)
    return new_model


def get_channels_attr(layer: tf.keras.layers.Layer) -> str:
    layer_config = layer.get_config()
    if 'units' in layer_config.keys():
        channels_attr = 'units'
    elif 'filters' in layer_config.keys():
        channels_attr = 'filters'
    else:
        raise ValueError('This layer has not got any channels.')
    return channels_attr


def get_node_depth(model: tf.keras.Model, node) -> int:
    """Get the depth of a node in a model.

    Arguments:
        model: Keras Model object
        node: Keras Node object

    Returns:
        The node depth as an integer. The model outputs are at depth 0.

    Raises:
        KeyError: if the node is not contained in the model.
    """
    # pylint: disable=protected-access
    for (depth, nodes_at_depth) in model._nodes_by_depth.items():
        if node in nodes_at_depth:
            return depth
    raise KeyError('The node is not contained in the model.')


def find_nodes_in_model(model: tf.keras.Model, layer: tf.keras.layers.Layer) -> list[int]:
    """Find the indices of layer's inbound nodes which are in model"""
    model_nodes = get_model_nodes(model)
    node_indices = []
    for i, node in enumerate(layer.inbound_nodes):
        if node in model_nodes:
            node_indices.append(i)
    return node_indices


def check_nodes_in_model(model: tf.keras.Model, nodes):
    """Check if nodes are in model"""
    model_nodes = get_model_nodes(model)
    nodes_in_model = [False] * len(nodes)
    for i, node in enumerate(nodes):
        if node in model_nodes:
            nodes_in_model[i] = True
    return nodes_in_model


def get_model_nodes(model: tf.keras.Model):
    """Return all nodes in the model"""
    # pylint: disable=protected-access
    return [node for v in model._nodes_by_depth.values() for node in v]


def get_shallower_nodes(node):
    possible_nodes = node.outbound_layer.outbound_nodes
    next_nodes = []
    for n in possible_nodes:
        if node in node_utils.parent_nodes(n):
            next_nodes.append(n)
    return next_nodes


def get_node_index(node) -> int:
    for i, n in enumerate(node.outbound_layer.inbound_nodes):
        if node == n:
            return i
    raise IndexError(f"{node.name} was not found in its outbound layer's inbound nodes.")


def find_activation_layer(
    layer: tf.keras.layers.Layer, node_index: int
) -> tuple[tf.keras.layers.Layer, int]:
    """

    Args:
        layer(Layer):
        node_index:
    """
    output_shape = layer.get_output_shape_at(node_index)
    maybe_layer = layer
    node = maybe_layer.inbound_nodes[node_index]
    # Loop will be broken by an error if an output layer is encountered
    while True:
        # If maybe_layer has a nonlinear activation function return it and its index
        activation = getattr(maybe_layer, 'activation', linear)
        if activation.__name__ != 'linear':
            if maybe_layer.get_output_shape_at(node_index) != output_shape:
                raise ValueError(
                    f'The activation layer ({maybe_layer.name}), does not have the same'
                    f' output shape as {layer.name}'
                )
            return maybe_layer, node_index

        # If not, move to the next layer in the datastream
        next_nodes = get_shallower_nodes(node)
        # test if node is a list of nodes with more than one item
        if len(next_nodes) > 1:
            raise ValueError(
                'The model must not branch between the chosen layer and the activation layer.'
            )
        node = next_nodes[0]
        node_index = get_node_index(node)
        maybe_layer = node.outbound_layer

        # Check if maybe_layer has weights, no activation layer has been found
        if maybe_layer.weights and (not maybe_layer.__class__.__name__.startswith('Global')):
            raise AttributeError(
                f'There is no nonlinear activation layer between {layer.name}'
                f' and {maybe_layer.name}'
            )


def sort_x_by_y(x: Iterable, y: Iterable) -> Iterable:
    """Sort the iterable x by the order of iterable y"""
    x = [x for (_, x) in sorted(zip(y, x))]
    return x


def _is_tensor(x: Any) -> bool:
    try:
        is_keras_tensor = tf.keras.backend.is_keras_tensor(x)
    except ValueError:
        is_keras_tensor = False
    if isinstance(x, tf.Tensor) or is_keras_tensor:
        return True
    return False


def single_element(x):
    """If x contains a single element, return it; otherwise return x"""
    if _is_tensor(x):
        return x

    if isinstance(x, (Collection, tuple)) and len(x) == 1:
        return x[0]
    return x


def get_one_tensor(x):
    if _is_tensor(x):
        return x

    if isinstance(x, (Collection, tuple)):
        assert (
            len(x) == 1
        ), "Ambiguous result: cannot get one item from collection with zero or more than two items"
        return x[0]
    return x


def all_equal(iterator: Iterable) -> bool:
    try:
        iterator = iter(iterator)
        first = next(iterator)
        return all(np.array_equal(first, rest) for rest in iterator)
    except StopIteration:
        return True


class MeanCalculator:
    def __init__(self, sum_axis: int):
        self.values: np.ndarray | None = None
        self.n: int = 0
        self.sum_axis = sum_axis

    def add(self, v: np.ndarray) -> None:
        if self.values is None:
            self.values = v.sum(axis=self.sum_axis)
        else:
            self.values += v.sum(axis=self.sum_axis)
        self.n += v.shape[self.sum_axis]

    def calculate(self) -> np.ndarray:
        assert self.values is not None, "No values have been added to the calculator."
        return self.values / self.n


def validate_node_indices(
    layer: tf.keras.layers.Layer, model: tf.keras.Model, node_indices
) -> list[int]:
    layer_node_indices = find_nodes_in_model(model, layer)
    # If no nodes are specified, all of the layer's inbound nodes which are
    # in model are selected.
    if node_indices is None:
        node_indices = layer_node_indices
    # Check for duplicate node indices
    elif len(node_indices) != len(set(node_indices)):
        raise ValueError('`node_indices` contains duplicate values.')
    # Check that all of the selected nodes are in the layer
    elif not set(node_indices).issubset(layer_node_indices):
        raise ValueError(
            'One or more nodes specified by `layer` and `node_indices` are not in `model`.'
        )
    return node_indices
