import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, Embedding, Reshape, Conv2D, Input, MaxPooling2D


def Conv1D(dim_output, kernel_size, strides=1, padding='same'):
    conv_op = tf.keras.layers.Conv1D(
        filters=dim_output,
        kernel_size=(kernel_size,),
        strides=strides,
        padding='same',
        use_bias=True)

    return conv_op


class PhoneClassifier():
    def __init__(self, args):
        self.conv_CIF = Conv1D(args.model.G.num_filters, args.model.G.kernel_size, strides=1, padding='same')
        self.layernorm_CIF = tf.keras.layers.LayerNormalization()
        self.fc_CIF = Dense(1, activation='sigmod')
        self.fc = Dense(args.dim_output, activation='linear')
        self.args = args
        self.threshold = args.model.G.threshold

    def CIF(self, x, len_y):
        x = self.conv_CIF(x)
        x = self.layernorm_CIF(x)
        x = tf.keras.layers.ReLU()(x)
        alpha = self.fc_CIF(x)
        _len_y = tf.reduce_sum(alpha, -1)
        # scaling
        alpha *= (len_y / _len_y)
        # integrate & fire
        integeate = tf.zeros(alpha.shape[0])
        for a in alpha:
            integeate += a
            tf.where(integeate > self.threshold, )
