from typing import TypeVar

import numpy as np
import tensorflow as tf


def inbound_nodes(layer: tf.keras.layers.Layer):
    return layer.inbound_nodes


L = TypeVar("L", bound=tf.keras.layers.Layer)


def make_new_layer(
    layer: L,
    config: dict | None = None,
    weights: np.ndarray | None = None,
) -> L:
    if config is None:
        config = layer.get_config()
    if weights is None:
        weights = layer.get_weights()
    config["weights"] = weights
    return type(layer).from_config(config)
