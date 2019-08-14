import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, GRUCell, Input, Reshape


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

    outputs = _x
    model = tf.keras.Model(inputs=input_x,
                           outputs=outputs,
                           name='AE-GRNN')
    return model



# def GRNN_Cell(args):
#     x = input_x = Input(shape=[args.dim_input], name='input')
#     state1 = Input(shape=[args.model.num_hidden_rnn], name='state1')
#     state2 = Input(shape=[args.model.num_hidden_rnn], name='state2')
#
#     x = Dense(args.model.num_hidden_fc, activation='relu')(x)
#     x = GRU(args.model.num_hidden_rnn,
#             dropout=args.model.dropout,
#             return_state=True)(x)
#     # x_fc1, _state1 = GRUCell(args.model.num_hidden_rnn, dropout=args.model.dropout)(inputs=x, states=[state1])
#     # x_cell1, _state2 = GRUCell(args.model.num_hidden_rnn, dropout=args.model.dropout)(inputs=x_fc1, states=[state2])
#     # _x = Dense(args.dim_input, activation='linear')(x_cell1)
#
#     model = tf.keras.Model(inputs=[input_x, state1, state2],
#                            outputs=[x_fc1, x_cell1, _state1, _state2],
#                            name='AE-GRNN-cell')
#     return model


def GRNN_Cell(args):
    x = input_x = Input(shape=[1, args.dim_input], name='input')
    state1 = Input(shape=[args.model.num_hidden_rnn], name='state1')
    state2 = Input(shape=[args.model.num_hidden_rnn], name='state2')

    x = Dense(args.model.num_hidden_fc, activation='relu')(x)
    x_fc = x
    x, h1 = GRU(args.model.num_hidden_rnn,
            dropout=args.model.dropout,
            return_sequences=True,
            return_state=True)(x, initial_state=state1)
    x_cell = x
    x, h2 = GRU(args.model.num_hidden_rnn,
            dropout=args.model.dropout,
            return_sequences=True,
            return_state=True)(x, initial_state=state2)
    _x = Dense(args.dim_input, activation='linear')(x)

    inputs = [input_x, state1, state2]
    outputs = [x_fc, x_cell, h1, h2]
    model = tf.keras.Model(inputs=inputs,
                           outputs=outputs,
                           name='AE-GRNN')
    return model


# def GRNN(args, training=True):
#     x = input_x = Input(shape=[None, args.dim_input], name='input')
#
#     if training:
#         x = Dense(args.model.num_hidden_fc, activation='relu')(x)
#         x = GRU(args.model.num_hidden_rnn,
#                 dropout=args.model.dropout,
#                 return_sequences=True)(x)
#     else:
#         x = fc_output = Dense(args.model.num_hidden_fc, activation='relu')(x)
#         x = gru_output = GRU(args.model.num_hidden_rnn,
#                              dropout=args.model.dropout,
#                              return_state=True,
#                              return_sequences=True)(x)
#     x = GRU(args.model.num_hidden_rnn,
#             dropout=args.model.dropout,
#             return_state=True,
#             return_sequences=True)(x)
#     _x = Dense(args.dim_input, activation='linear')(x)
#
#     if training:
#         outputs = _x
#     else:
#         outputs = [fc_output, gru_output, _x]
#
#     model = tf.keras.Model(inputs=input_x,
#                            outputs=outputs,
#                            name='AE-GRNN')
#
#     return model
