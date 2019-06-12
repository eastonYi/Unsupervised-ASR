import tensorflow as tf
import numpy as np
import logging
from time import time
from tensorflow.keras.layers import Dense, Bidirectional, LSTM
from random import sample


class FC_Model(tf.keras.models.Sequential):

    def __init__(self, args, optimizer, name='fc_model'):
        super().__init__([Dense(args.model.num_hidden,
                                activation='relu',
                                input_shape=(None, args.dim_input),
                                # kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                # activity_regularizer=tf.keras.regularizers.l1(args.l1)
                                )]
                         +
                          [Dense(args.model.num_hidden,
                                 activation='relu',
                                 # kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                 # activity_regularizer=tf.keras.regularizers.l1(args.l1)
                                 ) for _ in range(args.model.num_layers-1)]
                         +
                          [Dense(args.dim_output,
                                 activation='linear',
                                 # kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                 # activity_regularizer=tf.keras.regularizers.l1(args.l1)
                                 )]
        )

        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.conv_op = P_Ngram(N=args.data.ngram,
                               size_input=args.dim_output,
                               num_filter=args.data.top_k)

        self.compile(optimizer, self.loss_obj)
        self.build((None, None, args.dim_input))

    def align_loss(self, logits, labels, aligns, full_align=False):

        _logits, _labels, mask = self.post_process(logits, labels, aligns, full_align)

        loss = self.loss_obj(_labels, _logits)
        loss *= mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

        return loss

    def align_accuracy(self, logits, labels, aligns, full_align=False):

        _logits, _labels, mask = self.post_process(logits, labels, aligns, full_align)

        predicts = tf.argmax(_logits, axis=-1, output_type=tf.int32)
        results = tf.cast(tf.equal(predicts, _labels), tf.float32)

        results *= mask
        acc = tf.reduce_sum(results) / tf.reduce_sum(mask)

        return acc

    @staticmethod
    def post_process(logits, labels, aligns, full_align):
        batch, len_label_max = aligns.shape

        if full_align:
            mask = tf.cast(labels > 0, dtype=tf.float32)
            _logits = logits
            _labels = labels
        else:
            batch_idx = tf.tile(tf.range(batch)[:, None], [1, len_label_max])
            time_idx = aligns
            indices = tf.stack([batch_idx, time_idx], -1)
            _logits = tf.gather_nd(logits, indices)
            _labels = tf.gather_nd(labels, indices)
            mask = tf.cast(aligns > 0, dtype=tf.float32)

        return _logits, _labels, mask

    def get_predicts(self, logits):

        return tf.argmax(logits, axis=-1, output_type=tf.int32)

    def EODM_loss(self, logits, aligns, kernel, py):
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
        mask = tf.tile(tf.cast(aligns > 0, dtype=tf.float32)[:, :, None], [1, 1, k])
        px_batch = tf.nn.softmax(_logits)

        # compute p(z) average over the batch (nested batch and sent loops)
        pz = self.conv_op(kernel, px_batch) # pz: [b, t, z]
        pz = tf.reduce_sum(tf.reduce_sum(pz * mask[:, :pz.shape[1], :], 0), 0) / \
             tf.reduce_sum(tf.reduce_sum(mask, 0), 0) # [z]

        loss_z = - py * tf.math.log(pz+1e-5) # batch loss
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

    def frames_constrain_loss(self, logits, align):
        align += 1 # align means the new phone start time step
        end_time = tf.reduce_max(align, -1)
        batch_size = logits.shape[0]
        px_batch = tf.nn.softmax(logits)
        _frame = None
        loss = tf.zeros([batch_size], tf.float32)
        for i, frame in enumerate(tf.unstack(px_batch, axis=1)):
            if i > 1:
                pad_mask = tf.less(i, end_time)
                update_mask = tf.keras.backend.all(tf.not_equal(align, i), -1)
                mask = tf.cast(tf.logical_and(pad_mask, update_mask), dtype=tf.float32)
                loss += tf.reduce_mean(tf.pow(_frame-frame, 2), 1) * mask
            _frame = frame

        return tf.reduce_sum(loss)


class LSTM_Model(tf.keras.models.Sequential):

    def __init__(self, args, optimizer, name='lstm_model'):
        super().__init__([Bidirectional(LSTM(args.model.num_hidden, return_sequences=True),
                                        input_shape=(None, args.dim_input))]
                         +
                         [Bidirectional(LSTM(args.model.num_hidden, return_sequences=True)) for _ in range(args.model.num_layers-1)]
                         +
                         [Dense(args.dim_output, activation='relu')]
                         )
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.conv_op = P_Ngram(N=args.data.ngram,
                               size_input=args.dim_output,
                               num_filter=args.data.top_k)

        self.compile(optimizer, self.loss_obj)
        self.build((None, None, args.dim_input))

    def align_loss(self, logits, labels, aligns, full_align=False):
        _logits, _labels, mask = self.post_process(logits, labels, aligns, full_align)

        loss = self.loss_obj(_labels, _logits)
        loss *= mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

        return loss

    def align_accuracy(self, logits, labels, aligns, full_align=False):

        _logits, _labels, mask = self.post_process(logits, labels, aligns, full_align)

        predicts = tf.argmax(_logits, axis=-1, output_type=tf.int32)
        results = tf.cast(tf.equal(predicts, _labels), tf.float32)

        results *= mask
        acc = tf.reduce_sum(results) / tf.reduce_sum(mask)

        return acc

    @staticmethod
    def post_process(logits, labels, aligns, full_align):
        batch, len_label_max = aligns.shape

        if full_align:
            mask = tf.cast(labels > 0, dtype=tf.float32)
            _logits = logits
            _labels = labels
        else:
            batch_idx = tf.tile(tf.range(batch)[:, None], [1, len_label_max])
            time_idx = aligns
            indices = tf.stack([batch_idx, time_idx], -1)
            _logits = tf.gather_nd(logits, indices)
            _labels = tf.gather_nd(labels, indices)
            mask = tf.cast(aligns > 0, dtype=tf.float32)

        return _logits, _labels, mask

    def get_predicts(self, logits):

        return tf.argmax(logits, axis=-1, output_type=tf.int32)

    def EODM_loss(self, logits, aligns, kernel, py):
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
        indices = tf.stack([batch_idx, aligns], -1)
        _logits = tf.gather_nd(logits, indices)
        px_batch = tf.nn.softmax(_logits)

        # compute p(z) average over the batch (nested batch and sent loops)
        pz = self.conv_op(kernel, px_batch) # pz: [b, t, z]
        mask = tf.tile(tf.cast(aligns > 0, dtype=tf.float32)[:, :, None], [1, 1, k])[:, :pz.shape[1], :]
        pz = tf.reduce_sum(tf.reduce_sum(pz * mask, 0), 0) / \
             tf.reduce_sum(tf.reduce_sum(mask, 0), 0) # [z]

        loss_z = - py * tf.math.log(pz) # ngram loss
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
        indices = tf.stack([batch_idx, aligns], -1)
        _logits = tf.gather_nd(logits, indices)
        px_batch = tf.nn.softmax(_logits)

        # compute p(z) average over the batch (nested batch and sent loops)
        pz = self.conv_op(kernel, px_batch) # pz: [b, t, z]
        mask = tf.tile(tf.cast(aligns > 0, dtype=tf.float32)[:, :, None], [1, 1, k])[:, :pz.shape[1], :]
        pz = tf.reduce_sum(tf.reduce_sum(pz * mask, 0), 0)
        K = tf.reduce_sum(tf.reduce_sum(mask, 0), 0)

        return pz, K

    def frames_constrain_loss(self, logits, align):
        align += 1 # here align means the new phone start time step
        end_time = tf.reduce_max(align, -1)
        batch_size = logits.shape[0]
        px_batch = tf.nn.softmax(logits)
        _frame = None
        loss = tf.zeros([batch_size], tf.float32)
        for i, frame in enumerate(tf.unstack(px_batch, axis=1)):
            if i > 1:
                pad_mask = tf.less(i, end_time)
                update_mask = tf.keras.backend.all(tf.not_equal(align, i), -1)
                mask = tf.cast(tf.logical_and(pad_mask, update_mask), dtype=tf.float32)
                loss += tf.reduce_mean(tf.pow(_frame-frame, 2), 1) * mask
            _frame = frame

        return tf.reduce_sum(loss)


class P_Ngram():

    def __init__(self, N, size_input, num_filter):
        self.N = N
        self.size_input = size_input
        self.num_filter = num_filter
        conv_op = tf.keras.layers.Conv1D(filters=num_filter, kernel_size=(N,), strides=1,
                    padding='valid', use_bias=False)
        conv_op.build(tf.TensorShape((None, None, size_input)))
        conv_op.trainable = False
        self.conv_op = conv_op

    def __call__(self, kernel, input):
        """
        return:
            p: [batch, max_label_len]
        """
        self.conv_op.set_weights([kernel])

        input_log = tf.math.log(input+1e-5)
        p = tf.exp(self.conv_op(input_log))

        return p
