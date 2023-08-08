from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from kerassurgeon.types import Inputs, Masks


class OperableLayerMixin(ABC):
    # pylint: disable=R0903
    @abstractmethod
    def apply_delete_mask(
        self, inbound_masks: Masks, input_shape, inputs: Inputs
    ) -> tuple[tf.keras.layers.Layer, np.ndarray | None]:
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
            inbound_masks: Mask(s) from inbound node(s).
            input_shape: input shape of the original layer
            inputs: inputs to the original layer (can be used to set weights)

        Returns:
            new_layer: Pass through `layer` if it has no weights, otherwise a
                       new `Layer` object with weights corresponding to the
                       inbound mask deleted.
            outbound_mask: Mask corresponding to `new_layer`.
        """
        raise NotImplementedError
