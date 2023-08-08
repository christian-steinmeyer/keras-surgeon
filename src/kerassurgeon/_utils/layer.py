from collections.abc import Iterable
from typing import TypeVar

import numpy as np
import tensorflow as tf

from kerassurgeon import utils
from kerassurgeon.types import Inputs


def inbound_nodes(layer: tf.keras.layers.Layer):
    return layer.inbound_nodes


L = TypeVar("L", bound=tf.keras.layers.Layer)


def make_new_layer(
    layer: L,
    inputs: Inputs,
    config: dict | None = None,
    weights: list[np.ndarray] | None = None,
) -> L:
    if config is None:
        config = layer.get_config()
    if weights is None:
        weights = layer.get_weights()
    config["weights"] = weights
    new_layer: tf.keras.layers.Layer = type(layer).from_config(config)

    if isinstance(layer, (tf.keras.layers.MultiHeadAttention, tf.keras.layers.DepthwiseConv2D)):
        # multi head attention layer does not initialize weights correctly
        # see https://github.com/keras-team/keras/issues/18285
        new_input_shapes = tuple(tuple(_input.shape.as_list()) for _input in tuple(inputs))

        def sample_input(shape: Iterable[int]) -> np.ndarray:
            return np.ones(tuple(1 if dim is None else dim for dim in shape), dtype=float)

        new_layer.build(utils.single_element(new_input_shapes))
        new_layer(*tuple(map(sample_input, new_input_shapes)))
    else:
        new_layer(utils.single_element(inputs))

    new_layer.set_weights(weights)
    return new_layer
