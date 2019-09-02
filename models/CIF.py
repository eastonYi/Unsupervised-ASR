import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, Embedding, Reshape, Input
# from utils.tools import pad_list


def attentionAssign(args):
    x = input_x = Input(shape=[None, args.dim_input],
                        name='input_x')
    x = Bidirectional(GRU(args.model.attention.num_hidden, return_sequences=True))(x)
    alpha = Dense(1, activation='sigmoid')(x)[:, :, 0]

    model = tf.keras.Model(inputs=input_x,
                           outputs=alpha,
                           name='alpha')

    return model


# # @tf.function
# def CIF(x, alphas, threshold, max_label_len=100):
#     batch_size, len_time, hidden_size = x.shape
#
#     integrate_init = tf.constant(0.0)
#     frame_init = tf.zeros([hidden_size])
#     frames_init = tf.zeros([0, hidden_size])
#     batch_frames_init = tf.zeros([0, max_label_len, hidden_size])
#
#     def sent(b, batch_frames):
#         def step(b, t, integrate, frame, frames):
#             a = alphas[b, t]
#             integrate += a
#             if integrate > threshold:
#                 integrate -= tf.constant(1.0)
#                 frame += (a - integrate) * x[b, t, :]
#                 frames = tf.concat([frames, frame[None, :]], 0)
#                 frame = integrate * x[b, t, :]
#             else:
#                 frame += integrate * x[b, t, :]
#
#             return b, t+1, integrate, frame, frames
#
#         _, _, _, _, frames = tf.while_loop(
#             cond=lambda b, t, *_: tf.less(t, len_time),
#             body=step,
#             loop_vars=[b, 0, integrate_init, frame_init, frames_init],
#             shape_invariants=[tf.TensorShape([]),
#                               tf.TensorShape([]),
#                               tf.TensorShape([]),
#                               tf.TensorShape([hidden_size]),
#                               tf.TensorShape([None, hidden_size])])
#         pad_frames = tf.zeros([tf.reduce_max([max_label_len-tf.shape(frames)[0]], 0), hidden_size])
#         frames = tf.concat([frames, pad_frames], 0)
#         batch_frames = tf.concat([batch_frames, frames[None, :]], 0)
#
#         return b+1, batch_frames
#
#     _, batch_frames = tf.while_loop(
#     cond=lambda b, *_: tf.less(b, batch_size),
#     body=sent,
#     loop_vars=[0, batch_frames_init],
#     shape_invariants=[tf.TensorShape([]),
#                       tf.TensorShape([None, None, hidden_size])])
#
#     return batch_frames

# # @tf.function
# def CIF(x, alphas, threshold, max_label_len=100):
#     """
#     fires: b x t  (fire in place > thresdhold)
#     integrate: b
#
#     alphas: b x t
#     alpha: b
#
#     frames: b x t x h
#     frame: b x h
#
#     l: t x h
#     ls: b x t x h
#     """
#     batch_size, len_time, hidden_size = x.shape
#
#     fires_init = tf.zeros([batch_size, 1])
#     frame_init = tf.zeros([batch_size, hidden_size])
#     frames_init = tf.zeros([batch_size, 0, hidden_size])
#     ls_init = tf.zeros([0, max_label_len, hidden_size])
#
#     def step(t, fires, frame, frames):
#         alpha = alphas[:, t]
#         integrate = fires[:, -1] + alpha
#         fires = tf.concat([fires, integrate[:, None]], 1)
#
#         integrate = tf.where(integrate > threshold,
#                              x=integrate-tf.ones([batch_size]),
#                              y=integrate)
#         frame += integrate[:, None] * x[:, t, :]
#         frames = tf.concat([frames, frame[:, None, :]], 1)
#
#         frame = (alpha - integrate)[:, None] * x[:, t, :]
#
#         return t+1, fires, frame, frames
#
#     _, fires, _, frames = tf.while_loop(
#         cond=lambda t, *_: tf.less(t, len_time),
#         body=step,
#         loop_vars=[0, fires_init, frame_init, frames_init],
#         shape_invariants=[tf.TensorShape([]),
#                           tf.TensorShape([batch_size, None]),
#                           tf.TensorShape([batch_size, hidden_size]),
#                           tf.TensorShape([batch_size, None, hidden_size])])
#
#     fires = fires[:, 1:]
#
#     def sent(b, ls):
#         fire = fires[b, :]
#         l = tf.gather_nd(frames[b, :, :], tf.where(fire > threshold))
#         pad_l = tf.zeros([tf.reduce_max([max_label_len-tf.shape(l)[0], 0]), hidden_size])
#         l_padded = tf.concat([l, pad_l], 0)[:max_label_len, :]
#         ls = tf.concat([ls, l_padded[None, :]], 0)
#
#         return b+1, ls
#
#     _, ls = tf.while_loop(
#     cond=lambda b, *_: tf.less(b, batch_size),
#     body=sent,
#     loop_vars=[0, ls_init],
#     shape_invariants=[tf.TensorShape([]),
#                       tf.TensorShape([None, None, hidden_size])])
#
#     return ls

# # @tf.function
# def CIF(x, alphas, threshold):
#     """
#     fires: b x t  (fire in place > thresdhold)
#     integrate: b
#
#     alphas: b x t
#     alpha: b
#
#     frames: b x t x h
#     frame: b x h
#
#     l: t x h
#     ls: b x t x h
#     """
#     batch_size, len_time, hidden_size = x.shape
#
#     integrate = tf.zeros([batch_size])
#     list_fires = []
#     frame = tf.zeros([batch_size, hidden_size])
#     list_frames = []
#
#     for t in range(len_time):
#         alpha = alphas[:, t]
#         distribution_completion = tf.ones([batch_size]) - integrate
#         integrate += alpha
#         list_fires.append(integrate)
#
#         fire_place = integrate > threshold
#         integrate = tf.where(fire_place,
#                              x=integrate - tf.ones([batch_size]),
#                              y=integrate)
#         cur = tf.where(fire_place,
#                        x=distribution_completion,
#                        y=alpha)
#         remainds = alpha - cur
#
#         frame += cur[:, None] * x[:, t, :]
#         list_frames.append(frame)
#         frame = remainds[:, None] * x[:, t, :]
#
#     fires = tf.stack(list_fires, 1)
#     frames = tf.stack(list_frames, 1)
#     list_ls = []
#
#     len_labels = tf.cast(tf.round(tf.reduce_sum(alphas, -1)), tf.int32)
#     max_label_len = tf.reduce_max(len_labels)
#     for b, len in zip(range(batch_size), len_labels):
#         fire = fires[b, :]
#         l = tf.gather_nd(frames[b, :, :], tf.where(fire > threshold))
#         pad_l = tf.zeros([max_label_len-len, hidden_size])
#         list_ls.append(tf.concat([l, pad_l], 0))
#
#     return tf.stack(list_ls, 0)


# @tf.function
def CIF(x, alphas, threshold):
    """
    fires: b x t  (fire in place > thresdhold)
    integrate: b

    alphas: b x t
    alpha: b

    frames: b x t x h
    frame: b x h

    l: t x h
    ls: b x t x h
    """
    batch_size, len_time, hidden_size = x.shape

    integrate = tf.zeros([batch_size])
    list_fires = []
    frame = tf.zeros([batch_size, hidden_size])
    list_frames = []

    for t in range(len_time):
        alpha = alphas[:, t]
        distribution_completion = tf.ones([batch_size]) - integrate
        integrate += alpha
        list_fires.append(integrate)

        fire_place = integrate > threshold
        integrate = tf.where(fire_place,
                             x=integrate - tf.ones([batch_size]),
                             y=integrate)
        cur = tf.where(fire_place,
                       x=distribution_completion,
                       y=alpha)
        remainds = alpha - cur

        frame += cur[:, None] * x[:, t, :]
        list_frames.append(frame)
        frame = remainds[:, None] * x[:, t, :]

    fires = tf.stack(list_fires, 1)
    frames = tf.stack(list_frames, 1)
    list_ls = []

    len_labels = tf.cast(tf.round(tf.reduce_sum(alphas, -1)), tf.int32)
    max_label_len = tf.reduce_max(len_labels)
    for b, len in zip(range(batch_size), len_labels):
        fire = fires[b, :]
        l = tf.gather_nd(frames[b, :, :], tf.where(fire > threshold))
        pad_l = tf.zeros([max_label_len-len, hidden_size])
        list_ls.append(tf.concat([l, pad_l], 0))

    return tf.stack(list_ls, 0)
