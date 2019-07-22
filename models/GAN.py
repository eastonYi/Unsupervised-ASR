import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, Embedding, Reshape, Conv2D, Input, MaxPooling2D


def Conv1D(dim_output, kernel_size):
    conv_op = tf.keras.layers.Conv1D(
        filters=dim_output,
        kernel_size=(kernel_size,),
        strides=1,
        padding='same',
        use_bias=True)

    return conv_op


def ResBlock(inputs, list_Conv1D, list_norms):
    output = inputs

    output = list_Conv1D[0](output)
    output = list_norms[0](output)
    output = tf.nn.relu(output)
    # output = tf.keras.layers.LeakyReLU(alpha=0.2)(output)
    output = list_Conv1D[1](output)
    output = list_norms[1](output)
    output = tf.nn.relu(output)
    # output = tf.keras.layers.LeakyReLU(alpha=0.2)(output)

    return inputs + (0.3*output)


def PhoneClassifier(args):
    x = input_x = tf.keras.layers.Input(shape=[None, args.dim_input],
                                        name='generator_input_x')
    if args.model.G.structure == 'fc':
        for _ in range(args.model.G.num_layers):
            x = Dense(args.model.G.num_hidden, activation='relu')(x)
    elif args.model.G.structure == 'lstm':
        for _ in range(args.model.G.num_layers):
            x = LSTM(args.model.G.num_hidden,
                     return_sequences=True)(x)
    elif args.model.G.structure == 'blstm':
        for _ in range(args.model.G.num_layers):
            x = Bidirectional(LSTM(args.model.G.num_hidden,
                                   return_sequences=True))(x)

    logits = Dense(args.dim_output, activation='linear')(x)

    model = tf.keras.Model(inputs=input_x,
                           outputs=logits,
                           name='sequence_generator')

    return model


def PhoneDiscriminator(args):
    dim_hidden = args.model.D.num_hidden

    x = input = tf.keras.layers.Input(shape=[args.max_seq_len, args.dim_output],
                                      name='discriminator_input_x')

    pad_mask = input_mask = tf.keras.layers.Input(shape=[args.max_seq_len],
                                                  name='discriminator_input_mask',
                                                  dtype=tf.bool)
    pad_mask = tf.cast(pad_mask, tf.float32)

    x = Dense(dim_hidden, use_bias=False)(x)

    for i in range(args.model.D.num_blocks):
        inputs = x
        x = Conv1D(dim_output=dim_hidden, kernel_size=5)(x)
        # x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv1D(dim_output=dim_hidden, kernel_size=5)(x)
        # x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = inputs + (0.3*x)

    x = Conv1D(dim_output=1, kernel_size=1)(x)[:, :, 0]

    x *= pad_mask
    output = tf.reduce_sum(x, -1) / tf.reduce_sum(pad_mask, -1)

    model = tf.keras.Model(inputs=[input, input_mask],
                           outputs=output,
                           name='sequence_discriminator')

    return model

#
# class Generator(tf.keras.Model):
#
#     def __init__(self, dim_hidden, seq_len, dim_output):
#         super().__init__()
#         self.dim_hidden = dim_hidden
#         self.seq_len = seq_len
#         self.dim_output = dim_output
#         self.input_layer = Dense(dim_hidden*seq_len, activation=None)
#         self.list_Conv1D = [Conv1D(dim_output=dim_hidden, kernel_size=5)
#                             for _ in range(10)]
#         self.list_norms = [tf.keras.layers.BatchNormalization() for _ in range(10)]
#         self.list_Conv1D.append(Conv1D(dim_output=dim_output, kernel_size=1))
#
#         self.optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
#
#     def call(self, n_samples, SEQ_LEN):
#         output = tf.random.normal(shape=[n_samples, 128])
#         output = self.input_layer(output)
#         output = tf.reshape(output, [n_samples, SEQ_LEN, self.dim_hidden])
#         for i in range(5):
#             output = ResBlock(output, self.list_Conv1D[2*i: 2*i+2], self.list_norms[2*i: 2*i+2])
#         output = self.list_Conv1D[-1](output)
#         output = tf.nn.softmax(output)
#
#         return output
#
#
# class Discriminator(tf.keras.Model):
#
#     def __init__(self, dim_hidden, seq_len):
#         super().__init__()
#         self.dim_hidden = dim_hidden
#         self.seq_len = seq_len
#         self.embedding_layer = Dense(dim_hidden, use_bias=False)
#         self.list_Conv1D = [Conv1D(dim_output=dim_hidden, kernel_size=5)
#                             for _ in range(10)]
#         self.list_norms = [tf.keras.layers.BatchNormalization() for _ in range(10)]
#         # self.input_layer = Conv1D(dim_output=dim_hidden, kernel_size=1)
#         self.final_layer = Dense(1, activation=None)
#         # self.final_layer = Conv1D(dim_output=1, kernel_size=1)
#
#         self.optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)
#
#     def call(self, inputs):
#         batch_size = inputs.shape[0]
#         dim_input = inputs.shape[-1]
#         inputs = tf.image.random_crop(inputs, [batch_size, self.seq_len, dim_input])
#         output = self.embedding_layer(inputs)
#         # output = self.input_layer(inputs)
#         for i in range(5):
#             output = ResBlock(output, self.list_Conv1D[2*i: 2*i+2], self.list_norms[2*i: 2*i+2])
#         output = tf.reshape(output, [batch_size, -1])
#         output = self.final_layer(output)
#         # output = tf.reduce_mean(output[:, :, 0], -1)
#
#         return output
