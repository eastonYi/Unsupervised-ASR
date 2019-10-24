import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Bidirectional, \
    Conv2D, LayerNormalization, ReLU, GRU, LSTM, MaxPool1D, MaxPooling2D

from utils.tools import get_tensor_len


def Res_Conv(args):
    num_hidden = args.model.G.encoder.num_hidden

    input_x = Input(shape=[None, args.dim_input], name='encoder_input')
    len_seq = get_tensor_len(input_x)
    x = Dense(num_hidden, use_bias=False, activation='linear', name="encoder/fc_1")(input_x)

    for i in range(3):
        inputs = x
        x = Conv1D(dim_output=num_hidden, kernel_size=5)(x)
        # x = tf.keras.layers.LayerNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(dim_output=num_hidden, kernel_size=5)(x)
        # x = tf.keras.layers.LayerNormalization()(x)
        x = ReLU()(x)
        x = inputs + (0.3*x)
        x = MaxPool1D(pool_size=2, padding='SAME')(x)
        len_seq = tf.cast(tf.math.ceil(tf.cast(len_seq, tf.float32)/2), tf.int32)

    encoded = x
    pad_mask = tf.tile(tf.expand_dims(tf.sequence_mask(len_seq, dtype=tf.float32), -1),
                       [1, 1, num_hidden])
    encoded *= pad_mask

    encoder = tf.keras.Model(input_x, encoded, name='encoder')

    return encoder


def Conv_LSTM(args):
    num_hidden = args.model.G.encoder.num_hidden
    num_filters = args.model.G.encoder.num_filters
    size_feat = args.dim_input

    input_x = Input(shape=[None, args.dim_input], name='encoder_input')
    size_length = tf.shape(input_x)[1]
    size_feat = int(size_feat/3)
    len_feats = get_tensor_len(input_x)
    x = tf.reshape(input_x, [-1, size_length, size_feat, 3])
    # the first cnn layer
    x = normal_conv(
        x=x,
        filter_num=num_filters,
        kernel=(3,3),
        stride=(2,2),
        padding='SAME')
    # x = normal_conv(
    #     x=x,
    #     filter_num=num_filters,
    #     kernel=(3,3),
    #     stride=(1,1),
    #     padding='SAME')
    gates = Conv2D(
        4 * num_filters,
        (3,3),
        padding="SAME",
        dilation_rate=(1, 1))(x)
    g = tf.split(LayerNormalization()(gates), 4, axis=3)
    new_cell = tf.math.sigmoid(g[0]) * x + tf.math.sigmoid(g[1]) * tf.math.tanh(g[3])
    x = tf.math.sigmoid(g[2]) * tf.math.tanh(new_cell)

    size_feat = int(np.ceil(size_feat/2))*num_filters
    size_length = tf.cast(tf.math.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
    len_seq = tf.cast(tf.math.ceil(tf.cast(len_feats, tf.float32)/2), tf.int32)
    x = tf.reshape(x, [-1, size_length, size_feat])

    x = Bidirectional(LSTM(num_hidden//2, return_sequences=True))(x)
    x, len_seq = pooling(x, len_seq, num_hidden, 'HALF')

    x = Bidirectional(LSTM(num_hidden//2, return_sequences=True))(x)
    x, len_seq = pooling(x, len_seq, num_hidden, 'SAME')

    x = Bidirectional(LSTM(num_hidden//2, return_sequences=True))(x)
    x, len_seq = pooling(x, len_seq, num_hidden, 'HALF')

    x = Bidirectional(LSTM(num_hidden//2, return_sequences=True))(x)
    x, len_seq = pooling(x, len_seq, num_hidden, 'SAME')

    encoded = x
    pad_mask = tf.tile(tf.expand_dims(tf.sequence_mask(len_seq, dtype=tf.float32), -1),
                       [1, 1, num_hidden])
    encoded *= pad_mask

    encoder = tf.keras.Model(input_x, encoded, name='encoder')

    return encoder


## building blocks

def normal_conv(x, filter_num, kernel, stride, padding):

    x = Conv2D(filter_num, kernel, stride, padding)(x)
    x = LayerNormalization()(x)
    output = ReLU()(x)

    return output


def pooling(x, len_sequence, num_hidden, type):

    x = tf.expand_dims(x, axis=2)
    x = normal_conv(
        x,
        num_hidden,
        (1, 1),
        (1, 1),
        'SAME')

    if type == 'SAME':
        x = MaxPooling2D((1, 1), (1, 1), 'SAME')(x)
    elif type == 'HALF':
        x = MaxPooling2D((2, 1), (2, 1), 'SAME')(x)
        len_sequence = tf.cast(tf.math.ceil(tf.cast(len_sequence, tf.float32)/2), tf.int32)

    x = tf.squeeze(x, axis=2)

    return x, len_sequence


def Conv1D(dim_output, kernel_size, strides=1, padding='same'):
    conv_op = tf.keras.layers.Conv1D(
        filters=dim_output,
        kernel_size=(kernel_size,),
        strides=strides,
        padding='same',
        use_bias=True)

    return conv_op
