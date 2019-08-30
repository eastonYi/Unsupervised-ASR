import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, Embedding, Reshape, Input


def attentionAssign(args):
    x = input_x = Input(shape=[None, args.dim_input],
                        name='input_x')
    x = Bidirectional(GRU(args.model.G.num_hidden, return_sequences=True))(x)
    alpha = Dense(1, activation='sigmod')(x)

    model = tf.keras.Model(inputs=input_x,
                           outputs=alpha,
                           name='alpha')

    return model
