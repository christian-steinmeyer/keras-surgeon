import tensorflow as tf


def inbound_nodes(layer: tf.keras.layers.Layer):
    return layer.inbound_nodes
