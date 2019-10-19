import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, \
    Conv2D, Input, LayerNormalization, ReLU, GRU, MaxPool1D
from utils.tools import get_tensor_len


def FullyConv(args):
    num_hidden = args.model.G.num_hidden
    dim_output = args.dim_output
    dim_input = args.dim_input

    input_x = Input(shape=[100, dim_input], name='conv_lstm_input')

    len_seq = get_tensor_len(input_x)

    x = Dense(num_hidden, use_bias=False, activation='linear')(input_x)

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

    logits = Dense(dim_output)(x)
    pad_mask = tf.tile(tf.expand_dims(tf.sequence_mask(len_seq, dtype=tf.float32), -1),
                       [1, 1, dim_output])
    logits *= pad_mask
    model = tf.keras.Model(input_x, logits, name='conv_lstm_output')

    return model


def Conv1D(dim_output, kernel_size, strides=1, padding='same'):
    conv_op = tf.keras.layers.Conv1D(
        filters=dim_output,
        kernel_size=(kernel_size,),
        strides=strides,
        padding='same',
        use_bias=True)

    return conv_op
