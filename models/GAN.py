import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, Embedding, Reshape, Conv2D, Input, MaxPooling2D,MaxPool1D


def Conv1D(dim_output, kernel_size, strides=1, padding='same'):
    conv_op = tf.keras.layers.Conv1D(
        filters=dim_output,
        kernel_size=(kernel_size,),
        strides=strides,
        padding='same',
        use_bias=True)

    return conv_op


def PhoneClassifier(args):
    x = input_x = tf.keras.layers.Input(shape=[None, args.dim_input],
                                        name='generator_input_x')
    len_feats = None
    if args.model.G.structure == 'fc':
        x = x[:,::4,:]
        len_feats = tf.reduce_sum(tf.cast(tf.reduce_sum(tf.abs(x), -1) > 0, tf.int32), -1)
        for _ in range(args.model.G.num_layers):
            x = Dense(args.model.G.num_hidden, activation='relu')(x)
    elif args.model.G.structure == 'conv':
        len_feats = tf.reduce_sum(tf.cast(tf.reduce_sum(tf.abs(input_x), -1) > 0, tf.int32), -1)
        x = Dense(args.model.G.num_hidden)(x)
        for i in range(args.model.G.num_layers):
            inputs = x
            x = Conv1D(dim_output=args.model.G.num_hidden, kernel_size=5)(x)
            # x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = Conv1D(dim_output=args.model.G.num_hidden, kernel_size=5)(x)
            # x = tf.keras.layers.LayerNormalization()(x)
            x = tf.keras.layers.ReLU()(x)

            x = inputs + (0.3*x)

            x = MaxPool1D(strides=2, padding='same')(x)
            len_feats = tf.cast(tf.math.ceil(tf.cast(len_feats, tf.float32)/2), tf.int32)

        for i in range(1):
            x = Dense(args.model.G.num_hidden, activation='relu')(x)

    elif args.model.G.structure == 'lstm':
        for _ in range(args.model.G.num_layers):
            x = LSTM(args.model.G.num_hidden,
                     return_sequences=True)(x)
    elif args.model.G.structure == 'gru':
        for _ in range(args.model.G.num_layers):
            x = tf.keras.layers.GRU(args.model.G.num_hidden,
                                    dropout=args.model.G.dropout,
                                    return_sequences=True)(x)
    elif args.model.G.structure == 'blstm':
        for _ in range(args.model.G.num_layers):
            x = Bidirectional(LSTM(args.model.G.num_hidden,
                                   return_sequences=True))(x)
    elif args.model.G.structure == 'bGRU':
        for _ in range(args.model.G.num_layers):
            x = Bidirectional(GRU(int(args.model.G.num_hidden/2),
                                  return_sequences=True))(x)
    elif args.model.G.structure == 'fc+lstm':
        x_1 = input_x
        for _ in range(args.model.G.num_layers):
            x_1 = Dense(args.model.G.num_hidden, activation='relu')(x_1)

        x_2 = x_1
        x_2 = LSTM(64, return_sequences=True, dropout=0.2)(x_2)
        x_2 = tf.keras.layers.ReLU()(x_2)

        x = tf.concat([x_1, x_2], -1)
    elif args.model.G.structure == 'fc+GRU':
        x = input_x
        # for _ in range(args.model.G.num_layers):
        #     x = Dense(args.model.G.num_hidden, activation='relu')(x)
        x = Dense(args.model.G.num_fc_hidden, activation='relu')(x)
        x = Dense(args.model.G.num_hidden, activation='linear')(x)
        x = Bidirectional(GRU(args.model.G.num_cell_hidden, return_sequences=True))(x)
    elif args.model.G.structure == 'self-attention':
        from .SelfAttentionModel import SelfAttention

        mask = tf.cast(tf.reduce_sum(tf.abs(input_x), -1) > 0, tf.float32)
        x = SelfAttention(
            num_layers=args.model.G.num_layers,
            d_model=args.model.G.num_hidden,
            num_heads=args.model.G.num_heads,
            dff=args.model.G.num_hidden,
            input_vocab_size=args.dim_output,
            rate=args.model.G.dropout)(x, mask=mask)

    logits = Dense(args.dim_output, activation='linear')(x)
    if len_feats is not None:
        mask = tf.sequence_mask(len_feats, dtype=tf.float32)
    else:
        mask = tf.cast(tf.reduce_sum(tf.abs(input_x), -1) > 0, tf.float32)
    logits *= mask[:, :, None]
    model = tf.keras.Model(inputs=input_x,
                           outputs=logits,
                           name='sequence_generator')

    return model


def PhoneClassifier2(args):
    x = input_x = tf.keras.layers.Input(shape=[None, args.dim_input],
                                        name='generator_input_x')
    if args.model.G.structure == 'fc':
        for _ in range(args.model.G.num_layers):
            x = Dense(args.model.G.num_hidden, activation='relu')(x)
    elif args.model.G.structure == 'lstm':
        for _ in range(args.model.G.num_layers):
            x = LSTM(args.model.G.num_hidden,
                     return_sequences=True)(x)
    elif args.model.G.structure == 'gru':
        for _ in range(args.model.G.num_layers):
            x = tf.keras.layers.GRU(args.model.G.num_hidden,
                                    dropout=args.model.G.dropout,
                                    return_sequences=True)(x)
    elif args.model.G.structure == 'blstm':
        for _ in range(args.model.G.num_layers):
            x = Bidirectional(LSTM(args.model.G.num_hidden,
                                   return_sequences=True))(x)
    elif args.model.G.structure == 'bGRU':
        for _ in range(args.model.G.num_layers):
            x = Bidirectional(GRU(args.model.G.num_hidden,
                                  return_sequences=True))(x)
    elif args.model.G.structure == 'fc+lstm':
        x_1 = input_x
        for _ in range(args.model.G.num_layers):
            x_1 = Dense(args.model.G.num_hidden, activation='relu')(x_1)

        x_2 = x_1
        x_2 = LSTM(64, return_sequences=True, dropout=0.2)(x_2)
        x_2 = tf.keras.layers.ReLU()(x_2)

        x = tf.concat([x_1, x_2], -1)

    mask = tf.cast(tf.reduce_sum(tf.abs(input_x), -1) > 0, tf.float32)
    logits = Dense(args.dim_output, activation='linear')(x)
    logits *= mask[:, :, None]

    logits2 = Dense(2, activation='linear')(x)
    model = tf.keras.Model(inputs=input_x,
                           outputs=[logits, logits2],
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

    x = Dense(dim_hidden, use_bias=False, activation='linear')(x)

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


def PhoneDiscriminator2(args):
    dim_hidden = args.model.D.num_hidden

    x = input = tf.keras.layers.Input(shape=[args.max_seq_len, args.dim_output],
                                      name='discriminator_input_x')

    x = Dense(dim_hidden, use_bias=False, activation='linear')(x)

    for i in range(args.model.D.num_blocks):
        inputs = x
        x = Conv1D(dim_output=dim_hidden, kernel_size=3, strides=1, padding='valid')(x)
        # x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv1D(dim_output=dim_hidden, kernel_size=5, strides=1, padding='valid')(x)
        # x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = inputs + 1.0*x
        x = tf.keras.layers.MaxPooling1D(padding='same')(x)

    x = Conv1D(dim_output=10, kernel_size=7, strides=2, padding='valid')(x)
    _, time, hidden = x.shape
    x = tf.reshape(x, [-1, time*hidden])
    output = Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=input,
                           outputs=output,
                           name='sequence_discriminator')

    return model


def PhoneDiscriminator3(args):
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
        # x = tf.keras.layers.MaxPooling1D(padding='same')(x)
    _, time, hidden = x.shape
    x = tf.reshape(x, [-1, time*hidden])
    output = Dense(1, activation='linear')(x)

    model = tf.keras.Model(inputs=input,
                           outputs=output,
                           name='sequence_discriminator')

    return model
