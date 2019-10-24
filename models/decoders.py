import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, LSTM, Bidirectional, Input
from utils.tools import get_tensor_len


def Fully_Connected(args):
    dim_output = args.dim_output
    dim_input = args.model.G.encoder.num_hidden

    encoded = Input(shape=[None, dim_input], name='encoded')
    len_seq = get_tensor_len(encoded)

    logits = Dense(dim_output, name="decoder/fc")(encoded)
    pad_mask = tf.tile(tf.expand_dims(tf.sequence_mask(len_seq, dtype=tf.float32), -1),
                       [1, 1, dim_output])
    logits *= pad_mask

    decoder = tf.keras.Model(encoded, logits, name='decoder')

    return decoder


def RNN_FC(args):
    dim_input = args.model.G.encoder.num_hidden
    dim_output = args.dim_output
    num_hidden = args.model.G.decoder.num_hidden
    cell_type = args.model.G.decoder.cell_type
    dropout = args.model.G.decoder.dropout

    encoded = Input(shape=[None, dim_input], name='encoded')
    len_seq = get_tensor_len(encoded)

    if cell_type == 'gru':
        x = GRU(num_hidden,
                return_sequences=True,
                dropout=dropout,
                name="decoder/gru")(encoded)
    elif cell_type == 'lstm':
        x = LSTM(num_hidden,
                 return_sequences=True,
                 dropout=dropout,
                 name="decoder/lstm")(encoded)
    elif cell_type == 'bgru':
        x = Bidirectional(GRU(int(num_hidden//2),
                              return_sequences=True,
                              dropout=dropout,
                              name="decoder/gru"))(encoded)
    elif cell_type == 'blstm':
        x = Bidirectional(LSTM(int(num_hidden//2),
                               return_sequences=True,
                               dropout=dropout,
                               name="decoder/lstm"))(encoded)
    logits = Dense(dim_output, name="decoder/fc")(x)
    pad_mask = tf.tile(tf.expand_dims(tf.sequence_mask(len_seq, dtype=tf.float32), -1),
                       [1, 1, dim_output])
    logits *= pad_mask

    decoder = tf.keras.Model(encoded, logits, name='decoder')

    return decoder
