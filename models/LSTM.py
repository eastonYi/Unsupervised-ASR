import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, Embedding, Reshape, Conv2D, Input, MaxPooling2D


# class Embed_LSTM_Model(tf.keras.models.Sequential):
#
#     def __init__(self, args, optimizer, name='lstm_model'):
#         super().__init__([Embedding(args.dim_output, args.data.dim_embedding)]
#                          +
#                          [LSTM(args.model.num_hidden, return_sequences=True) for _ in range(args.model.num_layers)]
#                          +
#                          [Dense(args.dim_output, activation='linear')]
#                          )
#
#         self.tabel_embedding = self.layers[0]
#         self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
#
#         self.compile(optimizer, self.loss_obj)
#         self.build((None, None, args.dim_input))
#
#     def compute_loss(self, logits, labels):
#         loss = self.loss_obj(labels, logits)
#         mask = tf.cast(labels > 0, dtype=tf.float32)
#         loss *= mask
#         loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
#
#         return loss
#
#     def compute_ppl(self, logits, labels):
#         loss = self.loss_obj(labels, logits)
#         mask = tf.cast(labels > 0, dtype=tf.float32)
#         loss *= mask
#         loss_sum = tf.reduce_sum(loss)
#         token_sum = tf.reduce_sum(mask)
#
#         return loss_sum, token_sum
#
#     def compute_fitting_loss(self, logits, aligns):
#         """
#         for the other model (i.e. acoustic model) to fit the fixed LM.
#         """
#         batch, len_label_max = aligns.shape
#         batch_idx = tf.tile(tf.range(batch)[:, None], [1, len_label_max])
#         indices = tf.stack([batch_idx, aligns], -1)
#         _logits = tf.gather_nd(logits, indices)
#
#         label_musk = tf.cast(aligns>0, tf.float32)[:, 1:]
#         preds = tf.argmax(_logits, -1)
#         probs_lm = tf.nn.softmax(self(preds)[:, :-1, :])
#         ce_loss = tf.nn.softmax_cross_entropy_with_logits(
#             labels=probs_lm,
#             logits=_logits[:, 1:, :]) * label_musk
#         ce_loss = tf.reduce_sum(ce_loss) / tf.reduce_sum(label_musk)
#
#         return ce_loss


def Embed_LSTM_Model(args):
    x = input_x = tf.keras.layers.Input(shape=[None], name='token_input_x')
    x = Embedding(args.dim_output, args.data.dim_embedding)(x)
    for _ in range(args.model.num_layers):
        x = LSTM(args.model.num_hidden, return_sequences=True)(x)

    logits = Dense(args.dim_output, activation='linear')(x)

    model = tf.keras.Model(inputs=input_x, outputs=logits, name='logits')

    return model
