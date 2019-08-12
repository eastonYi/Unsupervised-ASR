import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, Input


def GRNN(args):
    x = input_x = Input(shape=[None, args.dim_input], name='input')

    x = Dense(args.model.num_hidden_fc, activation='relu')(x)
    x = GRU(args.model.num_hidden_rnn,
            dropout=args.model.dropout,
            return_sequences=True)(x)
    x = GRU(args.model.num_hidden_rnn,
            dropout=args.model.dropout,
            return_sequences=True)(x)
    _x = Dense(args.dim_input, activation='linear')(x)

    model = tf.keras.Model(inputs=input_x,
                           outputs=_x,
                           name='AE-GRNN')

    return model
