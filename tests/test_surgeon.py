# pylint: disable=W0621,R0914,C0302

import json
import os

import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.backend as K
from numpy import random
from tensorflow.keras.models import Model

from kerassurgeon import operations, utils
from kerassurgeon.operable_layer import OperableLayerMixin
from kerassurgeon.surgeon import Surgeon
from kerassurgeon.types import Masks

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Clear the tensorflow session and reset the default graph after each test.
# This prevents the testing from slowing down as more and more models are added
# to the tensorflow session.
@pytest.fixture(autouse=True)
def clear_tf():
    yield
    K.clear_session()
    tf.compat.v1.reset_default_graph()


@pytest.fixture(params=['channels_first', 'channels_last'])
def data_format(request):
    return request.param


@pytest.fixture(params=[[0], [-1], [1, 2]], ids=str)
def channel_index(request):
    return request.param


@pytest.fixture
def model_2():
    """Basic Lenet-style model test fixture with minimal channels"""
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv2D(
            2, [3, 3], input_shape=[28, 28, 1], data_format='channels_last', activation='relu'
        )
    )
    model.add(tf.keras.layers.Conv2D(2, [3, 3], activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    return Model(model.inputs, model.outputs)


def test_rebuild_submodel(model_2):
    output_nodes = []
    for output in model_2.outputs:
        # pylint: disable=protected-access
        layer, node_index, _ = output._keras_history
        output_nodes.append(layer.inbound_nodes[node_index])
    surgeon = Surgeon(model_2)
    # pylint: disable=protected-access
    outputs, _ = surgeon._rebuild_graph(model_2.inputs, output_nodes)
    new_model = Model(model_2.inputs, outputs)
    assert compare_models(model_2, new_model)


def test_delete_channels_rec_1():
    inputs = tf.keras.layers.Input(shape=(784,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    operations.delete_channels(model, model.layers[2], [0])


def model_3(data_format):
    if data_format == 'channels_last':
        main_input = tf.keras.layers.Input(shape=[7, 7, 1])
    elif data_format == 'channels_first':
        main_input = tf.keras.layers.Input(shape=[1, 7, 7])
    else:
        raise ValueError(data_format + ' is not a valid "data_format" value.')
    x = tf.keras.layers.Conv2D(3, [3, 3], data_format=data_format)(main_input)
    x = tf.keras.layers.Conv2D(3, [3, 3], data_format=data_format)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(3)(x)
    main_output = tf.keras.layers.Dense(1)(x)

    model = Model(inputs=main_input, outputs=main_output)

    # Set all of the weights
    w1 = [
        np.asarray(
            [
                [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]],
                [[[10, 11, 12]], [[13, 14, 15]], [[16, 17, 18]]],
                [[[19, 20, 21]], [[22, 23, 24]], [[25, 26, 27]]],
            ],
            dtype='float32',
        ),
        np.asarray([100, 200, 300], dtype='float32'),
    ]
    model.layers[1].set_weights(w1)
    w2 = [
        np.reshape(np.arange(0, 3 * 3 * 3 * 3, dtype='float32'), [3, 3, 3, 3]),
        np.asarray([100, 200, 300], dtype='float32'),
    ]
    model.layers[2].set_weights(w2)

    w4 = [
        np.reshape(np.arange(0, 3 * 3 * 3 * 3, dtype='float32'), [3 * 3 * 3, 3]),
        np.asarray([100, 200, 300], dtype='float32'),
    ]
    model.layers[4].set_weights(w4)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def model_4(data_format):
    if data_format == 'channels_last':
        main_input = tf.keras.layers.Input(shape=[2, 2, 1])
    elif data_format == 'channels_first':
        main_input = tf.keras.layers.Input(shape=[1, 2, 2])
    else:
        raise ValueError(data_format + ' is not a valid "data_format" value.')
    x = tf.keras.layers.Conv2D(3, [3, 3], data_format=data_format, padding='same')(main_input)
    x = tf.keras.layers.Conv2D(3, [3, 3], data_format=data_format, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(3)(x)
    main_output = tf.keras.layers.Dense(1)(x)

    model = Model(inputs=main_input, outputs=main_output)

    # Set all of the weights
    w1 = [
        np.asarray(
            [
                [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]],
                [[[10, 11, 12]], [[13, 14, 15]], [[16, 17, 18]]],
                [[[19, 20, 21]], [[22, 23, 24]], [[25, 26, 27]]],
            ],
            dtype='float32',
        ),
        np.asarray([100, 200, 300], dtype='float32'),
    ]
    model.layers[1].set_weights(w1)
    w2 = [
        np.reshape(np.arange(0, 3 * 3 * 3 * 3, dtype='float32'), [3, 3, 3, 3]),
        np.asarray([100, 200, 300], dtype='float32'),
    ]
    model.layers[2].set_weights(w2)

    w4 = [
        np.reshape(np.arange(0, 2 * 2 * 3 * 3, dtype='float32'), [2 * 2 * 3, 3]),
        np.asarray([100, 200, 300], dtype='float32'),
    ]
    model.layers[4].set_weights(w4)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def test_delete_channels_conv2d_conv2d(channel_index, data_format):
    model = model_3(data_format)
    layer_index = 1
    new_model = operations.delete_channels(
        model, model.layers[layer_index], channel_index, copy=True
    )
    channel_count = model.layers[layer_index].filters
    channel_index = [i % channel_count for i in channel_index]
    w = model.layers[layer_index].get_weights()
    correct_w = [np.delete(w[0], channel_index, axis=-1), np.delete(w[1], channel_index, axis=0)]
    new_w = new_model.layers[layer_index].get_weights()
    assert weights_equal(correct_w, new_w)


def test_delete_channels_conv2d_conv2d_small_shape():
    model = model_4("channels_last")
    layer_index = 1
    channel_index = [0]
    new_model = operations.delete_channels(
        model, model.layers[layer_index], channel_index, copy=True
    )
    channel_count = model.layers[layer_index].filters
    channel_index = [i % channel_count for i in channel_index]
    w = model.layers[layer_index].get_weights()
    correct_w = [np.delete(w[0], channel_index, axis=-1), np.delete(w[1], channel_index, axis=0)]
    new_w = new_model.layers[layer_index].get_weights()
    assert weights_equal(correct_w, new_w)


def test_delete_channels_conv2d_conv2d_next_layer(channel_index, data_format):
    model = model_3(data_format)
    layer_index = 1
    next_layer_index = 2
    new_model = operations.delete_channels(model, model.layers[layer_index], channel_index)
    channel_count = model.layers[layer_index].filters
    channel_index = [i % channel_count for i in channel_index]
    w = model.layers[next_layer_index].get_weights()
    correct_w = [np.delete(w[0], channel_index, axis=-2), w[1]]
    new_w = new_model.layers[next_layer_index].get_weights()
    assert weights_equal(correct_w, new_w)


def test_delete_channels_flatten(channel_index, data_format):
    # Create model
    main_input = tf.keras.layers.Input(shape=list(random.randint(4, 10, size=3)))
    x = tf.keras.layers.Conv2D(3, [3, 3], data_format=data_format)(main_input)
    x = tf.keras.layers.Flatten()(x)
    main_output = tf.keras.layers.Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    layer_index = 1
    next_layer_index = 3
    layer = model.layers[layer_index]
    new_model = operations.delete_channels(model, layer, channel_index)
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    flat_sz = np.prod(layer.output_shape[1:])
    channel_count = getattr(layer, utils.get_channels_attr(layer))
    channel_index = [i % channel_count for i in channel_index]
    if data_format == 'channels_first':
        delete_indices = [
            x * flat_sz // channel_count + i
            for x in channel_index
            for i in range(
                0,
                flat_sz // channel_count,
            )
        ]
    elif data_format == 'channels_last':
        delete_indices = [x + i for i in range(0, flat_sz, channel_count) for x in channel_index]
    else:
        raise ValueError
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], delete_indices, axis=0)

    assert weights_equal(correct_w, new_w)


def test_delete_channels_dense1d(channel_index):
    layer = tf.keras.layers.Dense(11)
    layer_test_helper_flatten_1d(layer, channel_index, should_forward_delete_masks=False)


def test_delete_channels_maxpooling1d(channel_index):
    layer = tf.keras.layers.MaxPool1D(2)
    layer_test_helper_flatten_1d(layer, channel_index)


def test_delete_channels_cropping1d(channel_index):
    layer = tf.keras.layers.Cropping1D(3)
    layer_test_helper_flatten_1d(layer, channel_index)


def test_delete_channels_upsampling1d(channel_index):
    layer = tf.keras.layers.UpSampling1D(3)
    layer_test_helper_flatten_1d(layer, channel_index)


def test_delete_channels_zeropadding1d(channel_index):
    layer = tf.keras.layers.ZeroPadding1D(3)
    layer_test_helper_flatten_1d(layer, channel_index)


def test_delete_channels_averagepooling1d(channel_index):
    layer = tf.keras.layers.AveragePooling1D(3)
    layer_test_helper_flatten_1d(layer, channel_index)


def test_delete_channels_separableconv1d(channel_index):
    layer = tf.keras.layers.SeparableConv1D(11, 2)
    layer_test_helper_flatten_1d(layer, channel_index, should_forward_delete_masks=False)


def test_delete_channels_depthwiseconv1d(channel_index):
    layer = tf.keras.layers.DepthwiseConv1D(3, 2)
    layer_test_helper_flatten_1d(layer, channel_index)


def test_delete_channels_globalaveragepooling1d(channel_index):
    layer = tf.keras.layers.GlobalAveragePooling1D()
    layer_test_helper_1d_global(layer, channel_index)


def test_delete_channels_multiheadattention(channel_index):
    layer = tf.keras.layers.MultiHeadAttention(num_heads=5, key_dim=3, value_dim=4)
    layer_test_helper_flatten_1d(
        layer, channel_index, repeat_layer_input=True, should_forward_delete_masks=False
    )


def test_delete_channels_maxpooling2d(channel_index, data_format):
    layer = tf.keras.layers.MaxPool2D([2, 2], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_cropping2d(channel_index, data_format):
    layer = tf.keras.layers.Cropping2D([2, 3], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_upsampling2d(channel_index, data_format):
    layer = tf.keras.layers.UpSampling2D([2, 3], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_zeropadding2d(channel_index, data_format):
    layer = tf.keras.layers.ZeroPadding2D([2, 3], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_averagepooling2d(channel_index, data_format):
    layer = tf.keras.layers.AveragePooling2D([2, 3], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_globalaveragepooling2d(channel_index, data_format):
    layer = tf.keras.layers.GlobalAveragePooling2D(data_format=data_format)
    layer_test_helper_2d_global(layer, channel_index, data_format)


def test_delete_channels_separableconv2d(channel_index, data_format):
    layer = tf.keras.layers.SeparableConv2D(11, [2, 3], data_format=data_format)
    layer_test_helper_flatten_2d(
        layer, channel_index, data_format, should_forward_delete_masks=False
    )


def test_delete_channels_depthwiseconv2d(channel_index, data_format):
    layer = tf.keras.layers.DepthwiseConv2D([2, 3], data_format=data_format)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def test_delete_channels_maxpooling3d(channel_index, data_format):
    layer = tf.keras.layers.MaxPool3D([2, 3, 2], data_format=data_format)
    layer_test_helper_flatten_3d(layer, channel_index, data_format=data_format)


def test_delete_channels_cropping3d(channel_index, data_format):
    layer = tf.keras.layers.Cropping3D([2, 3, 2], data_format=data_format)
    layer_test_helper_flatten_3d(layer, channel_index, data_format=data_format)


def test_delete_channels_upsampling3d(channel_index, data_format):
    layer = tf.keras.layers.UpSampling3D([2, 3, 2], data_format=data_format)
    layer_test_helper_flatten_3d(layer, channel_index, data_format=data_format)


def test_delete_channels_zeropadding3d(channel_index, data_format):
    layer = tf.keras.layers.ZeroPadding3D([2, 3, 2], data_format=data_format)
    layer_test_helper_flatten_3d(layer, channel_index, data_format=data_format)


def test_delete_channels_averagepooling3d(channel_index, data_format):
    layer = tf.keras.layers.AveragePooling3D([2, 3, 2], data_format=data_format)
    layer_test_helper_flatten_3d(layer, channel_index, data_format=data_format)


def test_delete_channels_merge_concatenate(channel_index, data_format):
    # This should test that the output is the correct shape so it should pass
    # into a Dense layer rather than a Conv layer.
    # The weighted layer is the previous layer,
    # Create model
    if data_format == 'channels_first':
        axis = 1
    elif data_format == 'channels_last':
        axis = -1
    else:
        raise ValueError

    input_shape = list(random.randint(10, 20, size=3))
    input_1 = tf.keras.layers.Input(shape=input_shape)
    input_2 = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(3, [3, 3], data_format=data_format, name='conv_1')(input_1)
    y = tf.keras.layers.Conv2D(3, [3, 3], data_format=data_format, name='conv_2')(input_2)
    x = tf.keras.layers.Concatenate(axis=axis, name='cat_1')([x, y])
    x = tf.keras.layers.Flatten()(x)
    main_output = tf.keras.layers.Dense(5, name='dense_1')(x)
    model = Model(inputs=[input_1, input_2], outputs=main_output)
    _ = model.get_layer('dense_1').get_weights()

    # Delete channels
    layer = model.get_layer('cat_1')
    del_layer = model.get_layer('conv_1')
    surgeon = Surgeon(model, copy=True)
    surgeon.add_job('delete_channels', del_layer, channels=channel_index)
    new_model = surgeon.operate()
    new_w = new_model.get_layer('dense_1').get_weights()

    # Calculate next layer's correct weights
    flat_sz = np.prod(layer.get_output_shape_at(0)[1:])
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    if data_format == 'channels_first':
        delete_indices = [
            x * flat_sz // 2 // channel_count + i
            for x in channel_index
            for i in range(
                0,
                flat_sz // 2 // channel_count,
            )
        ]
    elif data_format == 'channels_last':
        delete_indices = [
            x + i for i in range(0, flat_sz, channel_count * 2) for x in channel_index
        ]
    else:
        raise ValueError
    correct_w = model.get_layer('dense_1').get_weights()
    correct_w[0] = np.delete(correct_w[0], delete_indices, axis=0)

    assert weights_equal(correct_w, new_w)


def test_delete_channels_merge_others(channel_index, data_format):
    layer_test_helper_merge_2d(tf.keras.layers.Add(), channel_index, data_format)
    layer_test_helper_merge_2d(tf.keras.layers.Multiply(), channel_index, data_format)
    layer_test_helper_merge_2d(tf.keras.layers.Average(), channel_index, data_format)
    layer_test_helper_merge_2d(tf.keras.layers.Maximum(), channel_index, data_format)


def test_delete_channels_advanced_activations(channel_index, data_format):
    layer_test_helper_flatten_2d(tf.keras.layers.LeakyReLU(), channel_index, data_format)
    layer_test_helper_flatten_2d(tf.keras.layers.ELU(), channel_index, data_format)
    layer_test_helper_flatten_2d(tf.keras.layers.ThresholdedReLU(), channel_index, data_format)
    layer_test_helper_flatten_2d(tf.keras.layers.ReLU(), channel_index, data_format)


def test_delete_channels_noise(channel_index, data_format):
    layer_test_helper_flatten_2d(tf.keras.layers.GaussianNoise(0.5), channel_index, data_format)
    layer_test_helper_flatten_2d(tf.keras.layers.GaussianDropout(0.5), channel_index, data_format)
    layer_test_helper_flatten_2d(tf.keras.layers.AlphaDropout(0.5), channel_index, data_format)


def test_delete_channels_simplernn(channel_index):
    layer = tf.keras.layers.SimpleRNN(9, return_sequences=True)
    recursive_test_helper(layer, channel_index)


def test_delete_channels_gru(channel_index):
    layer = tf.keras.layers.GRU(9, return_sequences=True)
    recursive_test_helper(layer, channel_index)


def test_delete_channels_lstm(channel_index):
    layer = tf.keras.layers.LSTM(9, return_sequences=True)
    recursive_test_helper(layer, channel_index)


def test_delete_channels_batchnormalization(channel_index, data_format):
    if data_format == 'channels_first':
        axis = 1
    else:
        axis = -1

    layer = tf.keras.layers.BatchNormalization(axis=axis)
    layer_test_helper_flatten_2d(layer, channel_index, data_format)


def recursive_test_helper(layer: tf.keras.layers.Layer, channel_index):
    main_input = tf.keras.layers.Input(shape=[32, 10])
    x = layer(main_input)
    x = tf.keras.layers.GRU(4, return_sequences=False)(x)
    main_output = tf.keras.layers.Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    del_layer_index = 1
    next_layer_index = 2
    del_layer = model.layers[del_layer_index]
    new_model = operations.delete_channels(model, del_layer, channel_index)
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], channel_index, axis=0)

    assert weights_equal(correct_w, new_w)


def layer_test_helper_1d_global(layer: tf.keras.layers.Layer, channel_index):
    # This should test that the output is the correct shape so it should pass
    # into a Dense layer rather than a Conv layer.
    # The weighted layer is the previous layer,
    # Create model
    main_input = tf.keras.layers.Input(shape=list(random.randint(10, 20, size=2)))
    x = tf.keras.layers.Conv1D(3, 3)(main_input)
    x = layer(x)
    main_output = tf.keras.layers.Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    del_layer_index = 1
    next_layer_index = 3
    del_layer = model.layers[del_layer_index]
    new_model = operations.delete_channels(model, del_layer, channel_index)
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], channel_index, axis=0)

    assert weights_equal(correct_w, new_w)


def layer_test_helper_2d_global(layer: tf.keras.layers.Layer, channel_index, data_format):
    # This should test that the output is the correct shape so it should pass
    # into a Dense layer rather than a Conv layer.
    # The weighted layer is the previous layer,
    # Create model
    main_input = tf.keras.layers.Input(shape=list(random.randint(10, 20, size=3)))
    x = tf.keras.layers.Conv2D(3, [3, 3], data_format=data_format)(main_input)
    x = layer(x)
    main_output = tf.keras.layers.Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    del_layer_index = 1
    next_layer_index = 3
    del_layer = model.layers[del_layer_index]
    new_model = operations.delete_channels(model, del_layer, channel_index)
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], channel_index, axis=0)

    assert weights_equal(correct_w, new_w)


def layer_test_helper_flatten_1d(
    layer: tf.keras.layers.Layer,
    channel_index,
    repeat_layer_input: bool = False,
    should_forward_delete_masks: bool = True,
):
    # This should test that the output is the correct shape so it should pass
    # into a Dense layer rather than a Conv layer.
    # The weighted layer is the previous layer,
    # Create model
    main_input = tf.keras.layers.Input(shape=list(random.randint(10, 20, size=2)))
    x = tf.keras.layers.Conv1D(12, 3)(main_input)
    if repeat_layer_input:
        x = layer(x, x, x)
    else:
        x = layer(x)
    x = tf.keras.layers.Flatten()(x)
    main_output = tf.keras.layers.Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    del_layer_index = 1
    next_layer_index = 4
    del_layer = model.layers[del_layer_index]
    surgeon = Surgeon(model)
    surgeon.add_job('delete_channels', del_layer, channels=channel_index)
    new_model = surgeon.operate()
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    flat_sz = np.prod(layer.get_output_shape_at(0)[1:])
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    delete_indices = (
        [x + i for i in range(0, flat_sz, channel_count) for x in channel_index]
        if should_forward_delete_masks
        else []
    )

    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], delete_indices, axis=0)

    assert weights_equal(correct_w, new_w)


def layer_test_helper_flatten_2d(
    layer: tf.keras.layers.Layer,
    channel_index,
    data_format,
    should_forward_delete_masks: bool = True,
):
    # This should test that the output is the correct shape so it should pass
    # into a Dense layer rather than a Conv layer.
    # The weighted layer is the previous layer,
    # Create model
    main_input = tf.keras.layers.Input(shape=list(random.randint(10, 20, size=3)))
    x = tf.keras.layers.Conv2D(7, [3, 3], data_format=data_format)(main_input)
    x = layer(x)
    x = tf.keras.layers.Flatten()(x)
    main_output = tf.keras.layers.Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    del_layer_index = 1
    next_layer_index = 4
    del_layer = model.layers[del_layer_index]
    surgeon = Surgeon(model)
    surgeon.add_job('delete_channels', del_layer, channels=channel_index)
    new_model = surgeon.operate()
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    flat_sz = np.prod(layer.get_output_shape_at(0)[1:])
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    if data_format == 'channels_first':
        delete_indices = [
            x * flat_sz // channel_count + i
            for x in channel_index
            for i in range(
                0,
                flat_sz // channel_count,
            )
        ]
    elif data_format == 'channels_last':
        delete_indices = [x + i for i in range(0, flat_sz, channel_count) for x in channel_index]
    else:
        raise ValueError
    if not should_forward_delete_masks:
        delete_indices = []
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], delete_indices, axis=0)

    assert weights_equal(correct_w, new_w)


def layer_test_helper_flatten_3d(layer: tf.keras.layers.Layer, channel_index, data_format):
    # This should test that the output is the correct shape so it should pass
    # into a Dense layer rather than a Conv layer.
    # The weighted layer is the previous layer,
    # Create model
    main_input = tf.keras.layers.Input(shape=list(random.randint(10, 20, size=4)))
    x = tf.keras.layers.Conv3D(3, [3, 3, 2], data_format=data_format)(main_input)
    x = layer(x)
    x = tf.keras.layers.Flatten()(x)
    main_output = tf.keras.layers.Dense(5)(x)
    model = Model(inputs=main_input, outputs=main_output)

    # Delete channels
    del_layer_index = 1
    next_layer_index = 4
    del_layer = model.layers[del_layer_index]
    surgeon = Surgeon(model)
    surgeon.add_job('delete_channels', del_layer, channels=channel_index)
    new_model = surgeon.operate()
    new_w = new_model.layers[next_layer_index].get_weights()

    # Calculate next layer's correct weights
    flat_sz = np.prod(layer.get_output_shape_at(0)[1:])
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    if data_format == 'channels_first':
        delete_indices = [
            x * flat_sz // channel_count + i
            for x in channel_index
            for i in range(
                0,
                flat_sz // channel_count,
            )
        ]
    elif data_format == 'channels_last':
        delete_indices = [x + i for i in range(0, flat_sz, channel_count) for x in channel_index]
    else:
        raise ValueError
    correct_w = model.layers[next_layer_index].get_weights()
    correct_w[0] = np.delete(correct_w[0], delete_indices, axis=0)

    assert weights_equal(correct_w, new_w)


def layer_test_helper_merge_2d(layer: tf.keras.layers.Layer, channel_index, data_format):
    # This should test that the output is the correct shape so it should pass
    # into a Dense layer rather than a Conv layer.
    # The weighted layer is the previous layer,
    # Create model
    input_shape = list(random.randint(10, 20, size=3))
    input_1 = tf.keras.layers.Input(shape=input_shape)
    input_2 = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(3, [3, 3], data_format=data_format, name='conv_1')(input_1)
    y = tf.keras.layers.Conv2D(3, [3, 3], data_format=data_format, name='conv_2')(input_2)
    x = layer([x, y])
    x = tf.keras.layers.Flatten()(x)
    main_output = tf.keras.layers.Dense(5, name='dense_1')(x)
    model = Model(inputs=[input_1, input_2], outputs=main_output)

    # Delete channels
    del_layer = model.get_layer('conv_1')
    del_layer_2 = model.get_layer('conv_2')
    surgeon = Surgeon(model)
    surgeon.add_job('delete_channels', del_layer, channels=channel_index)
    surgeon.add_job('delete_channels', del_layer_2, channels=channel_index)
    new_model = surgeon.operate()
    new_w = new_model.get_layer('dense_1').get_weights()

    # Calculate next layer's correct weights
    flat_sz = np.prod(layer.get_output_shape_at(0)[1:])
    channel_count = getattr(del_layer, utils.get_channels_attr(del_layer))
    channel_index = [i % channel_count for i in channel_index]
    if data_format == 'channels_first':
        delete_indices = [
            x * flat_sz // channel_count + i
            for x in channel_index
            for i in range(
                0,
                flat_sz // channel_count,
            )
        ]
    elif data_format == 'channels_last':
        delete_indices = [x + i for i in range(0, flat_sz, channel_count) for x in channel_index]
    else:
        raise ValueError
    correct_w = model.get_layer('dense_1').get_weights()
    correct_w[0] = np.delete(correct_w[0], delete_indices, axis=0)

    assert weights_equal(correct_w, new_w)


def weights_equal(w1, w2):
    if len(w1) != len(w2):
        return False
    return all(np.array_equal(w1[i], w2[i]) for i in range(len(w1)))


def test_delete_layer():
    # Create all model layers
    input_1 = tf.keras.layers.Input(shape=[7, 7, 1])
    conv2d_1 = tf.keras.layers.Conv2D(3, [3, 3], data_format='channels_last')
    conv2d_2 = tf.keras.layers.Conv2D(3, [3, 3], data_format='channels_last')
    flatten_1 = tf.keras.layers.Flatten()
    dense_1 = tf.keras.layers.Dense(3)
    dense_2 = tf.keras.layers.Dense(3)
    dense_3 = tf.keras.layers.Dense(3)
    dense_4 = tf.keras.layers.Dense(1)
    # Create the base model
    x = conv2d_1(input_1)
    x = conv2d_2(x)
    x = flatten_1(x)
    x = dense_1(x)
    x = dense_2(x)
    x = dense_3(x)
    output_1 = dense_4(x)
    model_1 = utils.clean_copy(Model(input_1, output_1))
    # Create the expected modified model
    x = conv2d_1(input_1)
    x = conv2d_2(x)
    x = flatten_1(x)
    x = dense_1(x)
    x = dense_3(x)
    output_2 = dense_4(x)
    model_2_exp = utils.clean_copy(Model(input_1, output_2))
    # Delete layer dense_2
    model_2 = operations.delete_layer(model_1, model_1.get_layer(dense_2.name))
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_delete_layer_reuse():
    # Create all model layers
    input_1 = tf.keras.layers.Input(shape=[3])
    dense_1 = tf.keras.layers.Dense(3)
    dense_2 = tf.keras.layers.Dense(3)
    dense_3 = tf.keras.layers.Dense(3)
    dense_4 = tf.keras.layers.Dense(3)
    # Create the model
    x = dense_1(input_1)
    x = dense_2(x)
    x = dense_3(x)
    x = dense_2(x)
    output_1 = dense_4(x)
    # TODO: use clean_copy once keras issue 4160 has been fixed
    # model_1 = utils.clean_copy(Model(input_1, output_1))
    model_1 = Model(input_1, output_1)
    # Create the expected modified model
    x = dense_1(input_1)
    x = dense_3(x)
    output_2 = dense_4(x)
    # model_2_exp = utils.clean_copy(Model(input_1, output_2))
    model_2_exp = Model(input_1, output_2)
    # Delete layer dense_2
    model_2 = operations.delete_layer(model_1, model_1.get_layer(dense_2.name), copy=False)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_replace_layer():
    # Create all model layers
    input_1 = tf.keras.layers.Input(shape=[7, 7, 1])
    dense_1 = tf.keras.layers.Dense(3)
    dense_2 = tf.keras.layers.Dense(3)
    dense_3 = tf.keras.layers.Dense(3)
    dense_4 = tf.keras.layers.Dense(1)
    # Create the model
    x = dense_1(input_1)
    x = dense_2(x)
    output_1 = dense_4(x)
    model_1 = utils.clean_copy(Model(input_1, output_1))
    # Create the expected modified model
    x = dense_1(input_1)
    x = dense_3(x)
    output_2 = dense_4(x)
    model_2_exp = utils.clean_copy(Model(input_1, output_2))
    # Replace dense_2 with dense_3 in model_1
    model_2 = operations.replace_layer(model_1, model_1.get_layer(dense_2.name), dense_3)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_insert_layer():
    # Create all model layers
    input_1 = tf.keras.layers.Input(shape=[7, 7, 1])
    dense_1 = tf.keras.layers.Dense(3)
    dense_2 = tf.keras.layers.Dense(3)
    dense_3 = tf.keras.layers.Dense(3)
    dense_4 = tf.keras.layers.Dense(1)
    # Create the model
    x = dense_1(input_1)
    x = dense_2(x)
    output_1 = dense_4(x)
    model_1 = utils.clean_copy(Model(input_1, output_1))
    # Create the expected modified model
    x = dense_1(input_1)
    x = dense_2(x)
    x = dense_3(x)
    output_2 = dense_4(x)
    model_2_exp = utils.clean_copy(Model(input_1, output_2))
    # Insert dense_3 before dense_4 in model_1
    model_2 = operations.insert_layer(model_1, model_1.get_layer(dense_4.name), dense_3)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_delete_layer_same_layer_outputs():
    # Create all model layers
    input_1 = tf.keras.layers.Input(shape=(10,))
    dense_1 = tf.keras.layers.Dense(3)
    dense_2 = tf.keras.layers.Dense(3)
    dense_3 = tf.keras.layers.Dense(3)
    dense_4 = tf.keras.layers.Dense(1)
    # Create the base model
    x = dense_1(input_1)
    y = dense_2(x)
    x = dense_3(x)
    output_1 = dense_4(x)
    output_2 = dense_4(y)
    model_1 = utils.clean_copy(Model(input_1, [output_1, output_2]))
    # Create the expected modified model
    x = dense_1(input_1)
    y = dense_2(x)
    output_1 = dense_4(x)
    output_2 = dense_4(y)
    model_2_exp = utils.clean_copy(Model(input_1, [output_1, output_2]))
    # Delete layer dense_3
    model_2 = operations.delete_layer(model_1, model_1.get_layer(dense_3.name), copy=False)
    # Compare the modified model with the expected modified model
    assert compare_models(model_2, model_2_exp)


def test_delete_channels_downstream_sharing():
    # Create all model layers
    input_1 = tf.keras.layers.Input(shape=(5,))
    dense_1 = tf.keras.layers.Dense(4, name='dense_1')
    dense_2 = tf.keras.layers.Dense(4, name='dense_2')
    dense_3 = tf.keras.layers.Dense(3, name='dense_3')
    # Create the base model
    x = dense_1(input_1)
    y = dense_2(input_1)
    output_1 = dense_3(x)
    output_2 = dense_3(y)
    model_1 = utils.clean_copy(Model(input_1, [output_1, output_2]))
    # Delete channels from dense_1 and dense_2
    surgeon = Surgeon(model_1)
    surgeon.add_job('delete_channels', model_1.get_layer(dense_1.name), channels=[0])
    surgeon.add_job('delete_channels', model_1.get_layer(dense_2.name), channels=[1])
    model_2 = surgeon.operate()
    # Create the expected model
    # input_1 = tf.keras.layers.Input(shape=(5,))
    dense_1_exp = tf.keras.layers.Dense(3, name='dense_1')
    dense_2_exp = tf.keras.layers.Dense(3, name='dense_2')
    dense_3_exp = tf.keras.layers.Dense(3, name='dense_3')
    # Create the base model
    x = dense_1_exp(input_1)
    y = dense_2_exp(input_1)
    output_1 = dense_3_exp(x)
    output_2 = dense_3_exp(y)
    model_2_exp = utils.clean_copy(Model(input_1, [output_1, output_2]))

    config_1 = model_2.get_config()
    config_2 = model_2_exp.get_config()
    config_2['name'] = config_1['name']  # make the config names identical
    assert json.dumps(config_1) == json.dumps(config_2)


def test_delete_all_channels_in_branch():
    input_1 = tf.keras.layers.Input(shape=(20, 20, 3))
    conv_1 = tf.keras.layers.Conv2D(2, [3, 3], name='conv_1')
    conv_2 = tf.keras.layers.Conv2D(3, [3, 3], name='conv_2')
    cat_1 = tf.keras.layers.Concatenate(name='cat_1')

    x = conv_1(input_1)
    y = conv_2(input_1)
    output_1 = cat_1([x, y])
    model_1 = utils.clean_copy(Model(input_1, output_1))

    surgeon = Surgeon(model_1, copy=True)
    surgeon.add_job('delete_channels', model_1.get_layer('conv_1'), channels=[0, 1])
    model_2 = surgeon.operate()

    output_1 = conv_2(input_1)
    model_2_exp = utils.clean_copy(Model(input_1, output_1))

    config_1 = model_2.get_config()
    config_2 = model_2_exp.get_config()
    config_2['name'] = config_1['name']  # make the config names identical
    assert json.dumps(config_1) == json.dumps(config_2)


def test_delete_all_channels_in_long_branch():
    input_1 = tf.keras.layers.Input(shape=(20, 20, 3))
    conv_1 = tf.keras.layers.Conv2D(2, [3, 3], name='conv_1')
    conv_2 = tf.keras.layers.Conv2D(3, [3, 3], name='conv_2')
    conv_3 = tf.keras.layers.Conv2D(4, [1, 1], name='conv_3')
    cat_1 = tf.keras.layers.Concatenate(name='cat_1')

    x = conv_1(input_1)
    x = conv_3(x)
    y = conv_2(input_1)
    output_1 = cat_1([x, y])
    model_1 = utils.clean_copy(Model(input_1, output_1))

    surgeon = Surgeon(model_1, copy=True)
    surgeon.add_job('delete_channels', model_1.get_layer('conv_1'), channels=[0, 1])
    model_2 = surgeon.operate()

    output_1 = conv_2(input_1)
    model_2_exp = utils.clean_copy(Model(input_1, output_1))

    config_1 = model_2.get_config()
    config_2 = model_2_exp.get_config()
    config_2['name'] = config_1['name']  # make the config names identical
    assert config_1 == config_2


def compare_models(model_1, model_2):
    config_1 = model_1.get_config()
    config_2 = model_2.get_config()
    config_2['name'] = config_1['name']  # make the config names identical
    # json string dump avoids differences between tuples and lists
    config_match = json.dumps(config_1) == json.dumps(config_2)
    weights_match = all(
        np.array_equal(weight_1, weight_2)
        for (weight_1, weight_2) in zip(model_1.get_weights(), model_2.get_weights())
    )
    return config_match and weights_match


class CustomLayer(tf.keras.layers.Layer, OperableLayerMixin):
    def __init__(self, factor: float, **kwargs):
        super().__init__(**kwargs)
        self.factor = self.add_weight(
            name='factor', shape=(), initializer=tf.keras.initializers.Constant(factor)
        )

    def call(self, inputs, *_, **__):
        return inputs * self.factor

    def get_config(self):
        config = super().get_config()
        config['factor'] = self.factor.numpy()
        return config

    def apply_delete_mask(
        self, inbound_masks: Masks, input_shape  # pylint: disable=unused-argument
    ) -> tuple[tf.keras.layers.Layer, np.ndarray]:
        return self, inbound_masks  # no-op


def test_rebuild_submodel_with_operable_layer():
    inputs = tf.keras.layers.Input((64, 64, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(1, 3, activation='relu')(x)
    x = CustomLayer(2.5)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    model = tf.keras.Model(inputs=[inputs], outputs=x)

    output_nodes = []
    for output in model.outputs:
        # pylint: disable=protected-access
        layer, node_index, _ = output._keras_history
        output_nodes.append(layer.inbound_nodes[node_index])
    surgeon = Surgeon(model)
    # pylint: disable=protected-access
    outputs, _ = surgeon._rebuild_graph(model.inputs, output_nodes)
    new_model = Model(model.inputs, outputs)
    assert compare_models(model, new_model)


if __name__ == '__main__':
    pytest.main([__file__])
