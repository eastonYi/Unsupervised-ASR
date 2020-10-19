import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Input, MaxPooling2D


def Conv1D(dim_output, kernel_size, strides=1, padding='same'):
    conv_op = tf.keras.layers.Conv1D(
        filters=dim_output,
        kernel_size=(kernel_size,),
        strides=strides,
        padding='same',
        use_bias=True)

    return conv_op


def CLM(args):
    dim_hidden = args.model.D.num_hidden
    x = input = tf.keras.layers.Input(shape=[args.model.D.max_label_len, args.dim_output],
                                      name='discriminator_input_x')

    x = Dense(dim_hidden, use_bias=False, activation='linear')(x)

    for i in range(args.model.D.num_blocks):
        inputs = x
        x = Conv1D(dim_output=dim_hidden, kernel_size=3, strides=1, padding='valid')(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv1D(dim_output=dim_hidden, kernel_size=5, strides=1, padding='valid')(x)
        x = tf.keras.layers.ReLU()(x)

        x = inputs + 1.0*x
        x = tf.keras.layers.MaxPooling1D(padding='same')(x)

    _, time, hidden = x.shape
    x = tf.reshape(x, [-1, time*hidden])
    output = Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=input,
                           outputs=output,
                           name='sequence_discriminator')

    return model
