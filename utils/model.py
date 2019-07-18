import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, GRU, Embedding, Reshape, Conv2D, Input, MaxPooling2D


class FC_Model(tf.keras.models.Sequential):

    def __init__(self, args, optimizer, name='fc_model'):
        super().__init__([Input((None, args.dim_input))]
                         +
                          [Dense(args.model.num_hidden,
                                 activation='relu',
                                 kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
                                 bias_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05)
                                 # kernel_regularizer=tf.keras.regularizers.l2(args.l2),
                                 # activity_regularizer=tf.keras.regularizers.l1(args.l1)
                                 ) for _ in range(args.model.num_layers)]
                         +
                          [Dense(args.dim_output,
                                 activation='linear',
                                 kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05),
                                 bias_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.05)
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

    def align_loss(self, logits, labels, vocab_size, confidence=0.9):
        mask = tf.cast(labels > 0, dtype=tf.float32)

        # loss = self.loss_obj(labels, logits)

        low_confidence = (1.0 - confidence) / tf.cast(vocab_size-1, tf.float32)
        normalizing = -(confidence*tf.math.log(confidence) +
            tf.cast(vocab_size-1, tf.float32) * low_confidence * tf.math.log(low_confidence + 1e-20))
        soft_targets = tf.one_hot(
            tf.cast(labels, tf.int32),
            depth=vocab_size,
            on_value=confidence,
            off_value=low_confidence)

        xentropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=soft_targets)
        loss = xentropy - normalizing

        loss *= mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

        return loss

    def align_accuracy(self, logits, labels):
        mask = tf.cast(labels > 0, dtype=tf.float32)

        predicts = tf.argmax(logits, axis=-1, output_type=tf.int32)
        results = tf.cast(tf.equal(predicts, labels), tf.float32)

        results *= mask
        acc = tf.reduce_sum(results) / tf.reduce_sum(mask)

        return acc

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

        loss_z = - py * tf.math.log(pz+1e-15) # batch loss
        loss = tf.reduce_sum(loss_z)

        return loss

    # @tf.function
    # def __call__(self, *args):
    #     return super().__call__(*args)

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

    def __init__(self, args, optimizer, name='fc_model'):
        super().__init__([LSTM(args.model.num_hidden, return_sequences=True) for _ in range(args.model.num_layers)]
                         +
                         [Dense(args.dim_output, activation='linear')]
                         )

        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        # self.conv_op = P_Ngram(N=args.data.ngram,
        #                        size_input=args.dim_output,
        #                        num_filter=args.data.top_k)

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


class Conv_Model(tf.keras.models.Sequential):
    def __init__(self, args, optimizer, name='conv_model'):
        super().__init__(
            [Reshape((-1, args.dim_input, 1), input_shape=(None, args.dim_input))]
            +
            [Conv2D(filters=args.model.num_filters,
                    kernel_size=args.model.kernel_size,
                    strides=(1, 1),
                    padding='same')]
            +
            [Reshape((-1, args.dim_input * args.model.num_filters))]
            +
            [Dense(args.model.num_hidden, activation='relu') for _ in range(args.model.num_layers-1)]
            +
            [Dense(args.dim_output, activation='linear')]
            )

        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.conv_op = P_Ngram(N=args.data.ngram,
                               size_input=args.dim_output,
                               num_filter=args.data.top_k)

        self.compile(optimizer, self.loss_obj)
        self.build((None, None, args.dim_input))

    def align_loss(self, logits, labels):
        mask = tf.cast(labels > 0, dtype=tf.float32)

        loss = self.loss_obj(labels, logits)
        loss *= mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

        return loss

    def align_accuracy(self, logits, labels):
        mask = tf.cast(labels > 0, dtype=tf.float32)

        predicts = tf.argmax(logits, axis=-1, output_type=tf.int32)
        results = tf.cast(tf.equal(predicts, labels), tf.float32)

        results *= mask
        acc = tf.reduce_sum(results) / tf.reduce_sum(mask)

        return acc

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

        loss_z = - py * tf.math.log(pz+1e-15) # batch loss
        loss = tf.reduce_sum(loss_z)

        return loss

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

        input_log = tf.math.log(input+1e-15)
        p = tf.exp(self.conv_op(input_log))

        return p


class Embed_LSTM_Model(tf.keras.models.Sequential):

    def __init__(self, args, optimizer, name='lstm_model'):
        super().__init__([Embedding(args.dim_output, args.data.dim_embedding)]
                         +
                         [LSTM(args.model.num_hidden, return_sequences=True) for _ in range(args.model.num_layers)]
                         +
                         [Dense(args.dim_output, activation='linear')]
                         )

        self.tabel_embedding = self.layers[0]
        self.loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

        self.compile(optimizer, self.loss_obj)
        self.build((None, None, args.dim_input))

    def compute_loss(self, logits, labels):
        loss = self.loss_obj(labels, logits)
        mask = tf.cast(labels > 0, dtype=tf.float32)
        loss *= mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

        return loss

    def compute_ppl(self, logits, labels):
        loss = self.loss_obj(labels, logits)
        mask = tf.cast(labels > 0, dtype=tf.float32)
        loss *= mask
        loss_sum = tf.reduce_sum(loss)
        token_sum = tf.reduce_sum(mask)

        return loss_sum, token_sum

    def compute_fitting_loss(self, logits, aligns):
        """
        for the other model (i.e. acoustic model) to fit the fixed LM.
        """
        batch, len_label_max = aligns.shape
        batch_idx = tf.tile(tf.range(batch)[:, None], [1, len_label_max])
        indices = tf.stack([batch_idx, aligns], -1)
        _logits = tf.gather_nd(logits, indices)

        label_musk = tf.cast(aligns>0, tf.float32)[:, 1:]
        preds = tf.argmax(_logits, -1)
        probs_lm = tf.nn.softmax(self(preds)[:, :-1, :])
        ce_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=probs_lm,
            logits=_logits[:, 1:, :]) * label_musk
        ce_loss = tf.reduce_sum(ce_loss) / tf.reduce_sum(label_musk)

        return ce_loss


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


class Generator(tf.keras.Model):

    def __init__(self, dim_hidden, seq_len, dim_output):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.seq_len = seq_len
        self.dim_output = dim_output
        self.input_layer = Dense(dim_hidden*seq_len, activation=None)
        self.list_Conv1D = [Conv1D(dim_output=dim_hidden, kernel_size=5)
                            for _ in range(10)]
        self.list_norms = [tf.keras.layers.BatchNormalization() for _ in range(10)]
        self.list_Conv1D.append(Conv1D(dim_output=dim_output, kernel_size=1))

        self.optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    def call(self, n_samples, SEQ_LEN):
        output = tf.random.normal(shape=[n_samples, 128])
        output = self.input_layer(output)
        output = tf.reshape(output, [n_samples, SEQ_LEN, self.dim_hidden])
        for i in range(5):
            output = ResBlock(output, self.list_Conv1D[2*i: 2*i+2], self.list_norms[2*i: 2*i+2])
        output = self.list_Conv1D[-1](output)
        output = tf.nn.softmax(output)

        return output


class Discriminator(tf.keras.Model):

    def __init__(self, dim_hidden, seq_len):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.seq_len = seq_len
        self.embedding_layer = Dense(dim_hidden, use_bias=False)
        self.list_Conv1D = [Conv1D(dim_output=dim_hidden, kernel_size=5)
                            for _ in range(10)]
        self.list_norms = [tf.keras.layers.BatchNormalization() for _ in range(10)]
        # self.input_layer = Conv1D(dim_output=dim_hidden, kernel_size=1)
        self.final_layer = Dense(1, activation=None)
        # self.final_layer = Conv1D(dim_output=1, kernel_size=1)

        self.optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, beta_2=0.9)

    def call(self, inputs):
        batch_size = inputs.shape[0]
        dim_input = inputs.shape[-1]
        inputs = tf.image.random_crop(inputs, [batch_size, self.seq_len, dim_input])
        output = self.embedding_layer(inputs)
        # output = self.input_layer(inputs)
        for i in range(5):
            output = ResBlock(output, self.list_Conv1D[2*i: 2*i+2], self.list_norms[2*i: 2*i+2])
        output = tf.reshape(output, [batch_size, -1])
        output = self.final_layer(output)
        # output = tf.reduce_mean(output[:, :, 0], -1)

        return output


def PhoneClassifier(args):
    x = input_x = tf.keras.layers.Input(shape=[None, args.dim_input],
                                        name='generator_input_x')
    if args.model.G.structure == 'fc':
        for _ in range(args.model.G.num_layers):
            x = Dense(args.model.G.num_hidden, activation='relu')(x)
    elif args.model.G.structure == 'lstm':
        for _ in range(args.model.G.num_layers):
            x = LSTM(args.model.G.num_hidden,
                     return_sequences=True,
                     dropout=args.model.G.dropout)(x)
    elif args.model.G.structure == 'blstm':
        for _ in range(args.model.G.num_layers):
            x = Bidirectional(LSTM(args.model.G.num_hidden,
                                   return_sequences=True,
                                   dropout=args.model.G.dropout))(x)

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
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        x = Conv1D(dim_output=dim_hidden, kernel_size=5)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        x = inputs + (0.3*x)

    x = Conv1D(dim_output=1, kernel_size=1)(x)[:, :, 0]

    x *= pad_mask
    output = tf.reduce_sum(x, -1) / tf.reduce_sum(pad_mask, -1)

    model = tf.keras.Model(inputs=[input, input_mask],
                           outputs=output,
                           name='sequence_discriminator')

    return model
