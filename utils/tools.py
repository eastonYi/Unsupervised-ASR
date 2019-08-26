import tensorflow as tf
import numpy as np
import editdistance as ed
from pathlib import Path
from tqdm import tqdm
from random import shuffle
import hashlib
import math


def mkdirs(filename):
    if not filename.parent.is_dir():
        mkdirs(filename.parent)

    if '.' not in str(filename) and not filename.is_dir():
        filename.mkdir()


class warmup_exponential_decay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, peak, decay_steps):
        super().__init__()
        self.warmup_steps = warmup_steps*1.0
        self.peak = peak*1.0
        self.decay_steps = decay_steps*1.0

    def __call__(self, step):
        warmup_steps, peak, decay_steps = self.warmup_steps, self.peak, self.decay_steps
        # warmup_steps = tf.to_float(warmup_steps)
        global_step = tf.cast(step, tf.float32)

        return tf.where(global_step <= warmup_steps,
                        peak * global_step / warmup_steps,
                        peak * 0.5 ** ((global_step - warmup_steps) / decay_steps))


class TFData:
    """
    test on TF2.0
    """
    def __init__(self, dataset, dir_save, args, size_file=5000000, max_feat_len=3000):
        self.dataset = dataset
        self.max_feat_len = max_feat_len
        self.dir_save = dir_save
        self.args = args
        self.size_file = size_file
        self.dim_feature = dataset[0]['feature'].shape[-1] \
            if dataset else self.read_tfdata_info(dir_save)['dim_feature']

    def save(self, name, length_file='feature_length.txt'):
        num_token = 0
        num_damaged_sample = 0
        fw = open(self.dir_save/length_file, 'w')
        def serialize_example(uttid, feature):
            atts = {
                'uttid': self._bytes_feature(bytes(uttid, 'UTF-8')),
                'feature': self._bytes_feature(feature.tostring())
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=atts))

            return example_proto.SerializeToString()

        def generator():
            nonlocal fw
            for sample, _ in zip(self.dataset, tqdm(range(len(self.dataset)))):
                line = sample['uttid'] + ' ' + str(len(sample['feature']))
                fw.write(line + '\n')
                yield serialize_example(sample['uttid'], sample['feature'])

        dataset_tf = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=tf.string,
            output_shapes=())

        record_file = self.dir_save/'{}.recode'.format(name)
        mkdirs(record_file)
        writer = tf.data.experimental.TFRecordWriter(str(record_file))
        writer.write(dataset_tf)

        with open(str(self.dir_save/'tfdata.info'), 'w') as fw:
            fw.write('data_file {}\n'.format(self.dataset.file))
            fw.write('dim_feature {}\n'.format(self.dim_feature))
            fw.write('num_tokens {}\n'.format(num_token))
            fw.write('size_dataset {}\n'.format(len(self.dataset)-num_damaged_sample))
            fw.write('damaged samples: {}\n'.format(num_damaged_sample))

        return

    def read(self, _shuffle=False):
        """
        the tensor could run unlimitatly
        return a iter
        """
        list_filenames = self.fentch_filelist(self.dir_save)
        if _shuffle:
            shuffle(list_filenames)
        else:
            list_filenames.sort()

        raw_dataset = tf.data.TFRecordDataset(list_filenames)

        def _parse_function(example_proto):
            sample = tf.io.parse_single_example(
                example_proto,
                features={
                    'uttid': tf.io.FixedLenFeature([], tf.string),
                    'feature': tf.io.FixedLenFeature([], tf.string)
                }
            )
            uttid = sample['uttid']
            feature = tf.reshape(tf.io.decode_raw(sample['feature'], tf.float32),
                                 [-1, self.dim_feature])[:self.max_feat_len, :]

            return uttid, feature

        feature = raw_dataset.map(_parse_function)

        return feature

    def __len__(self):
        return self.read_tfdata_info(self.dir_save)['size_dataset']

    @staticmethod
    def fentch_filelist(dir_data):
        p = Path(dir_data)
        assert p.is_dir()

        return [str(i) for i in p.glob('*.recode')]

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a list of string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int_feature(value):
        """Returns a int_list."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def read_tfdata_info(dir_save):
        data_info = {}
        with open(dir_save/'tfdata.info') as f:
            for line in f:
                if 'dim_feature' in line or \
                    'num_tokens' in line or \
                    'size_dataset' in line:
                    line = line.strip().split(' ')
                    data_info[line[0]] = int(line[1])

        return data_info


def build_optimizer(args, lr=0.5, type='adam'):
    if type == 'adam':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            args.opti.peak,
            decay_steps=args.opti.decay_steps,
            decay_rate=0.5,
            staircase=False)
        optimizer = tf.keras.optimizers.Adam(
            lr_schedule,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9)
    elif type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            lr=lr,
            momentum=0.9,
            decay=0.98)

    return optimizer


def sampleFrames(align):
    """
    align:
    return please ignore the value in sample where in align is 0
    """
    align = tf.cast(align, tf.float32)
    pad = tf.zeros([align.shape[0], 1], dtype=tf.float32)
    _align = tf.concat([pad, align[:, :-1]], 1)
    sample = tf.cast((_align + (align-_align)*tf.random.uniform(align.shape))*tf.cast(align > 0, tf.float32), tf.int32)

    return sample


def get_dataset_ngram(text_file, n, k, savefile=None, split=5000):
    """
    Simply concatenate all sents into one will bring in noisy n-gram at end of each sent.
    Here we count ngrams for each sent and sum them up.
    """
    from utils.dataProcess import get_N_gram
    from nltk import FreqDist

    def iter_in_sent(sent):
        for word in sent.split():
            yield word

    print('analysing text ...')

    list_utterances = open(text_file).readlines()

    ngrams_global = FreqDist()
    for i in range(len(list_utterances)//split +1):
        ngrams = FreqDist()
        text = list_utterances[i*split: (i+1)*split]
        for utt in tqdm(text):
            _, seq_label, _ = utt.strip().split(',')
            ngram = get_N_gram(iter_in_sent(seq_label), n)
            ngrams += ngram

        ngrams_global += dict(ngrams.most_common(2*k))

    if savefile:
        with open(savefile, 'w') as fw:
            for ngram,num in ngrams_global.most_common(k):
                line = '{}:{}'.format(ngram,num)
                fw.write(line+'\n')

    return ngrams_global


def read_ngram(top_k, file, token2idx, type='list'):
    """
    """
    total_num = 0
    ngram_py = []
    with open(file) as f:
        for _, line in zip(range(top_k), f):
            ngram, num = line.strip().split(':')
            ngram = tuple(token2idx[i[1:-1]] for i in ngram[1:-1].split(', '))
            ngram_py.append((ngram, int(num)))
            total_num += int(num)

    if type == 'dict':
        dict_ngram_py = {}
        for ngram, num in ngram_py:
            dict_ngram_py[ngram] = num/total_num

        return dict_ngram_py

    elif type == 'list':
        list_ngram_py = []
        for ngram, num in ngram_py:
            list_ngram_py.append((ngram, num/total_num))

        return list_ngram_py, total_num


def align_shrink(align):
    _token = None
    list_tokens = []
    for token in align:
        if _token != token:
            list_tokens.append(token)
            _token = token

    return list_tokens


def batch_cer(preds, reference):
    """
    preds, reference: align type
    result and reference are lists of tokens
    eos_idx is the padding token or eos token
    """
    batch_dist = 0
    batch_len = 0
    for res, ref in zip(preds, reference):
        length = np.sum(ref>0)
        res = align_shrink(res[:length])
        ref = align_shrink(ref[:length])
        batch_dist += ed.eval(res, ref)
        batch_len += len(ref)

    return batch_dist, batch_len


def pertubated_model_weights(w, p, sigma):
    weights_try = []
    for index, i in enumerate(p):
        jittered = sigma*i
        weights_try.append(w[index] + jittered)

    return weights_try


def ngram2kernel(ngram, args):
    kernel = np.zeros([args.data.ngram, args.dim_output, args.data.top_k], dtype=np.float32)
    list_py = []
    for i, (z, py) in enumerate(ngram):
        list_py.append(py)
        for j, token in enumerate(z):
            kernel[j][token][i] = 1.0
    py = np.array(list_py, dtype=np.float32)

    return kernel, py


def get_preds_ngram(preds, len_preds, n):
    """
    Simply concatenate all sents into one will bring in noisy n-gram at end of each sent.
    Here we count ngrams for each sent and sum them up.
    """
    from utils.dataProcess import get_N_gram

    def iter_preds(preds, len_preds):
        for len, utt in zip(len_preds, preds):
            for token in utt[:len]:
                yield token.numpy()
    ngrams = get_N_gram(iter_preds(preds, len_preds), n)

    return ngrams


def gradient_penalty(D, real, fake, mask_real=None, mask_fake=None):
    def _interpolate(a, b):
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.shape)
        return inter

    if mask_real is not None:
        min_len = min(real.shape[1], fake.shape[1])
        x = _interpolate(real[:, :min_len, :], fake[:, :min_len, :])
        mask = tf.logical_and(mask_real[:, :min_len], mask_fake[:, :min_len])
    else:
        assert real.shape == fake.shape
        x = _interpolate(real, fake)
        mask = None

    with tf.GradientTape() as t:
        t.watch(x)
        pred = D(x)
    grad = t.gradient(pred, x)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((norm - 1.)**2)

    return gp


def frames_constrain_loss(logits, align):
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


def align2stamp(align):
    if align is not None:
        list_stamps = []
        label_prev = align[0]
        for i, label in enumerate(align):
            if label_prev != label:
                list_stamps.append(i-1)
            label_prev = label
        list_stamps.append(i)
    else:
        list_stamps = None

    return np.array(list_stamps)


def stamps2indices(stamps, sample='random'):
    """
    aligns:
    return please ignore the value in sample where in aligns is 0
    """
    if sample == 'random':
        stamps = tf.cast(stamps, tf.float32)
        _stamps = tf.pad(stamps, [[0, 0], [1, 0]])[:, :-1]
        sampled_stamps = tf.cast((_stamps + (stamps-_stamps)*
            tf.random.uniform(stamps.shape))*tf.cast(stamps > 0, tf.float32), tf.int32)
    elif sample == 'middle':
        stamps = tf.cast(stamps, tf.float32)
        _stamps = tf.pad(stamps, [[0, 0], [1, 0]])[:, :-1]
        sampled_stamps = tf.cast((_stamps + (stamps-_stamps)*
            0.5*tf.ones_like(stamps))*tf.cast(stamps > 0, tf.float32), tf.int32)
    elif sample == 'end':
        sampled_stamps = stamps
    batch_idx = tf.tile(tf.range(stamps.shape[0])[:, None], [1, stamps.shape[1]])
    indices = tf.stack([batch_idx, sampled_stamps], -1)

    return indices


def str_md5(string):
    encryption = hashlib.md5()
    encryption.update(string)
    return encryption.hexdigest()


def align_accuracy(P_output, labels):
    mask = tf.cast(labels > 0, dtype=tf.float32)

    predicts = tf.argmax(P_output, axis=-1, output_type=tf.int32)
    results = tf.cast(tf.equal(predicts, labels), tf.float32)

    results *= mask
    acc = tf.reduce_sum(results) / tf.reduce_sum(mask)

    return acc


def get_predicts(P_output):

    return tf.argmax(P_output, axis=-1, output_type=tf.int32)


def compute_ppl(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    mask = tf.cast(labels > 0, dtype=tf.float32)
    loss *= mask
    loss_sum = tf.reduce_sum(loss)
    token_sum = tf.reduce_sum(mask)

    return loss_sum, token_sum


def CE_loss(logits, labels, vocab_size, confidence=0.9):
    mask = tf.cast(labels>0, dtype=tf.float32)

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


def get_GRU_activation(layer, cell_inputs, hiddens):
    """
    gru/kernel: h_prev x h
    gru/recurrent_kernel: h x (h*3)
    gru/bias: 2 x (h*3)

    cell_inputs: b x h_prev
    hiddens: b x h
    """
    assert "GRU" in str(layer)
    activation_fn = layer.recurrent_activation
    kernel, recurrent_kernel, bias = layer.get_weights()
    matrix_x = tf.matmul(cell_inputs, kernel)
    matrix_x = tf.add(matrix_x, bias[0])
    x_z, x_r, _ = tf.split(matrix_x, 3, axis=-1)

    matrix_inner = tf.matmul(hiddens, recurrent_kernel)
    matrix_inner = tf.add(matrix_inner, bias[1])
    recurrent_z, recurrent_r, _ = tf.split(matrix_inner, 3, axis=-1)

    z = tf.reduce_sum(activation_fn(x_z + recurrent_z), 1)
    r = tf.reduce_sum(activation_fn(x_r + recurrent_r), 1)

    return z, r


def evaluate(feature, dataset, dev_size, model):
    list_acc = []

    num_processed = 0
    total_cer_dist = 0
    total_cer_len = 0
    for batch in feature:
        uttids, x = batch
        aligns = dataset.get_attrs('align', uttids.numpy())
        trans = dataset.get_attrs('trans', uttids.numpy())
        stamps = dataset.get_attrs('stamps', uttids.numpy())
        logits = model(x)
        # logits, cut_idx, max_idx = model(x)

        acc = align_accuracy(logits, aligns)
        list_acc.append(acc)

        indices = stamps2indices(stamps, 'middle')
        _logits = tf.gather_nd(logits, indices)
        preds = get_predicts(_logits)
        batch_cer_dist, batch_cer_len = batch_cer(preds.numpy(), trans)
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len

        num_processed += len(x)

    cer = total_cer_dist/total_cer_len
    fer = 1-np.mean(list_acc)
    print('dev FER: {:.3f}\t dev PER: {:.3f}\t {} / {}'.format(
           fer, cer, num_processed, dev_size))

    return fer, cer


def monitor(sample, model):
    x = np.array([sample['feature']], dtype=np.float32)
    logits = model(x)
    predicts = get_predicts(logits)
    print('predicts: \n', predicts.numpy()[0])
    print('align: \n', sample['align'])
    print('trans: \n', sample['trans'])


def decode(dataset, model, idx2token, save_file, align=False, log=False):
    """
    decode without stamps
    decode align and shrink it to trans
    """
    with open(save_file, 'w') as fw:
        for sample in tqdm(dataset):
            uttid = sample['uttid']
            x = np.array([sample['feature']], dtype=np.float32)
            logits = model(x)
            _align = get_predicts(logits)[0].numpy()
            list_tokens = []
            token_prev = None
            for token in _align:
                if token_prev == token:
                    continue
                list_tokens.append(token)
                token_prev = token
            if align:
                line = ' '.join(str(token) for token in _align)
            else:
                line = ' '.join(idx2token[token] for token in list_tokens)
            if log:
                print('predicted align: ', _align)
                print('predicted trans: ', line)
            fw.write(uttid + ' ' + line + '\n')


def R_value(res_stamps, ref_stamps, region=2):
    """
    res_stamps, res_stamps: 1-dim np.array
    region: the left and right tolerant region size
    Demo:
        res_stamps = [4,7,10,14,16]
        ref_stamps = [3,5,9,11,15]
        R_value(res_stamps, ref_stamps, region=2)
    """
    print('ref_stamps:', ref_stamps)
    print('res_stamps:', res_stamps)
    N_ref = len(ref_stamps)
    N_f = len(res_stamps)
    N_hit = 0

    _left = 0
    for i, stamp in enumerate(ref_stamps):
        left = max(_left, stamp-region)
        try:
            right = min(stamp+region, (stamp+ref_stamps[i+1])/2 + 0.01)
        except IndexError:
            right = stamp+region
        # print('left:', left, 'right:', right)

        for j, _stamp in enumerate(res_stamps):
            if _stamp < left:
                continue
            elif _stamp > right:
                j = j-1
                break
            else:
                N_hit += 1
                # print('hit:', ref_stamps[i], res_stamps[j])
                break
        res_stamps = res_stamps[j+1:]
        _left = right

    HR = N_hit / N_ref
    OS = N_f / N_ref - 1
    r1 = math.sqrt(math.pow(1-HR, 2) + math.pow(OS, 2))
    r2 = (HR - OS - 1) / 1.414
    R = 1 - (abs(r1) + abs(r2)) / 2
    print('r-value:', R, '\n')

    return R
