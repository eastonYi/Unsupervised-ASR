import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, Embedding, Reshape, Conv2D, Input, MaxPooling2D


def EODM_loss(_logits, mask, conv_op, k, py):
    """
    Eq.1
    tau is deterministic (stochastic in paper) and omit the tau loop
    args:
        logits:
        aligns:     segmentations for the batch
        ngram_py:   ngram and its proportion
    """
    mask = tf.tile(tf.cast(mask, dtype=tf.float32)[:, :, None], [1, 1, k])
    px_batch = tf.nn.softmax(_logits)

    # compute p(z) average over the batch (nested batch and sent loops)
    pz = conv_op(px_batch) # pz: [b, t, z]
    pz = tf.reduce_sum(pz * mask[:, :pz.shape[1], :], [0, 1]) / \
         tf.reduce_sum(mask, [0, 1]) # [z]

    loss_z = - py * tf.math.log(pz+1e-15) # batch loss
    loss = tf.reduce_sum(loss_z)

    return loss


def EODM(self, logits, aligns, kernel):
    """
    Eq.1
    tau is deterministic (stochastic in paper) and omit the tau loop
    args:
        logits:
        aligns:     segmentations for the batch
        ngram_py:   ngram and its proportion
    """
    batch, len_label_max = aligns.shape
    k = kernel.shape[-1]
    batch_idx = tf.tile(tf.range(batch)[:, None], [1, len_label_max])
    time_idx = aligns
    indices = tf.stack([batch_idx, time_idx], -1)
    _logits = tf.gather_nd(logits, indices)
    px_batch = tf.nn.softmax(_logits)

    # compute p(z) average over the batch (nested batch and sent loops)
    pz = self.conv_op(kernel, px_batch) # pz: [b, t, z]
    mask = tf.tile(tf.cast(aligns > 0, dtype=tf.float32)[:, :, None], [1, 1, k])[:, :pz.shape[1], :]

    pz = tf.reduce_sum(tf.reduce_sum(pz * mask, 0), 0)
    K = tf.reduce_sum(tf.reduce_sum(mask, 0), 0) # [z]

    return pz, K


def P_Ngram(kernel, args):
    """
    kernel: [args.data.ngram, args.dim_output, args.data.top_k]
    return:
        p: [batch, max_label_len]
    """
    input_x = tf.keras.layers.Input(shape=[None, args.dim_input],
                                    name='input_x')

    x_log = tf.math.log(input_x+1e-15)
    x_conv = tf.keras.layers.Conv1D(filters=args.data.top_k,
                                    kernel_size=(args.data.ngram,),
                                    strides=1,
                                    padding='valid',
                                    use_bias=False,
                                    kernel_initializer=lambda _, dtype: kernel,
                                    trainable=False)(x_log)
    p = tf.exp(x_conv)

    model = tf.keras.Model(inputs=input_x,
                           outputs=p,
                           name='P_ngram')

    return model
