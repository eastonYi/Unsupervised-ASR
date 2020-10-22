import tensorflow as tf
import numpy as np
import editdistance as ed
from pathlib import Path
from tqdm import tqdm
from random import shuffle
import hashlib
import math
from time import time
from struct import pack, unpack


def get_tensor_len(tensor):
    if tensor.get_shape().ndims == 3:
        return tf.reduce_sum(tf.cast((tf.reduce_max(tf.abs(tensor), -1) > 0), tf.int32), -1)
    elif tensor.get_shape().ndims == 2:
        return tf.reduce_sum(tf.cast(tf.abs(tensor) > 0, tf.int32), -1)

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


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class TFData:
    """
    test on TF2.0
    save and read tfdata
    """
    def __init__(self, dataset, dir_save, args, size_file=5000000, max_feat_len=3000):
        self.dataset = dataset
        self.max_feat_len = max_feat_len
        self.dir_save = dir_save
        self.args = args
        self.size_file = size_file
        try:
            self.dim_feature = self.read_tfdata_info(dir_save)['dim_feature']
        except:
            self.dim_feature = dataset[0]['feature'].shape[-1]

    def split_save(self, length_file='feature_length.txt', capacity=50000):
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
            nonlocal fw, i, capacity
            # for sample, _ in zip(self.dataset, tqdm(range(len(self.dataset)))):
            for j in tqdm(range(i*capacity, min((i+1)*capacity, len(self.dataset)))):
                sample = self.dataset[j]
                line = sample['uttid'] + ' ' + str(len(sample['feature']))
                fw.write(line + '\n')
                yield serialize_example(sample['uttid'], sample['feature'])

        for i in range(len(self.dataset)//capacity + 1):
        # for i in [2,3,4]:
            dataset_tf = tf.data.Dataset.from_generator(
                generator=generator,
                output_types=tf.string,
                output_shapes=())
            record_file = self.dir_save/'{}.recode'.format(i)
            mkdirs(record_file)
            writer = tf.data.experimental.TFRecordWriter(str(record_file))
            writer.write(dataset_tf)

        with open(str(self.dir_save/'tfdata.info'), 'w') as fw:
            fw.write('data_file {}\n'.format(self.dataset.file))
            fw.write('dim_feature {}\n'.format(self.dataset[0]['feature'].shape[-1]))
            fw.write('num_tokens {}\n'.format(num_token))
            fw.write('size_dataset {}\n'.format(len(self.dataset)-num_damaged_sample))
            fw.write('damaged samples: {}\n'.format(num_damaged_sample))

        return

    def read(self, _shuffle=False, transform=False):
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
            if transform:
                feature = process_raw_feature(feature, self.args)

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


def ctc_shrink(logits):
    batch_size, len_time, dim_output = logits.shape
    blk = dim_output - 1
    tokens = tf.argmax(logits, -1)
    # intermediate vars along time
    list_fires = []
    token_prev = tf.ones((batch_size), tf.int64) * -1
    blk_batch = tf.cast(tf.ones((batch_size), tf.int32) * blk, tf.int64)
    pad_batch = tf.zeros((batch_size), tf.int64)

    for t in tf.range(len_time):
        token = tokens[:, t]
        fire_place = tf.logical_and(token != blk_batch, token != token_prev)
        fire_place = tf.logical_and(fire_place, token != pad_batch)
        list_fires.append(fire_place)
        token_prev = token

    fires = tf.stack(list_fires, 1)
    label_len = tf.reduce_sum(tf.cast(fires, tf.int32), -1)
    max_label_len = tf.reduce_max(label_len)
    list_ls = []
    # len_labels = tf.reduce_sum(tf.cast(fires, tf.int32), -1)
    # max_label_len = tf.reduce_max(len_labels)
    for b in tf.range(batch_size):
        l = tf.gather_nd(logits[b, :, :], tf.where(fires[b]))
        pad_l = tf.zeros([max_label_len-tf.shape(l)[0], dim_output])
        list_ls.append(tf.concat([l, pad_l], 0))

    logits_shrunk = tf.stack(list_ls, 0)
    return logits_shrunk


def pad_to(tensor, length):
    len_tensor = tensor.shape[1]
    max_len = tf.reduce_max([length, len_tensor])
    pad_shape = list(tensor.shape)
    pad_shape[1] = max_len - len_tensor
    pad = tf.zeros(pad_shape, dtype=tensor.dtype)
    tensor_padded = tf.concat([tensor, pad], 1)

    return tensor_padded


def batch_cer(preds, reference):
    """
    preds, reference: align type
    result and reference are lists of tokens
    eos_idx is the padding token or eos token
    """
    batch_dist = 0
    batch_len = 0
    batch_res_len = 0
    for res, ref in zip(preds, reference):
        res = align_shrink(res[res>0])
        ref = align_shrink(ref[ref>0])
        # print(len(res)/len(ref))
        batch_dist += ed.eval(res, ref)
        batch_len += len(ref)
        batch_res_len += len(res)

    return batch_dist, batch_len, batch_res_len


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
        # assert real.shape == fake.shape
        x = _interpolate(real, fake)

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


def align2bound(align):
    if align is not None:
        list_stamps = []
        label_prev = align[0]
        for label in align:
            list_stamps.append(1 if label_prev != label else 0)
            label_prev = label
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


def bounds2stamps(bounds):
    lengths = tf.reduce_sum(bounds, 1)
    max_len = tf.reduce_max(lengths)
    batch_size, len_time = bounds.shape
    list_stamps = []
    for b in range(batch_size):
        stamps = tf.where(bounds[b, :] > 0)[:, 0]
        stamps = tf.concat([stamps, tf.zeros([max_len - lengths[b]], tf.int64)], 0)
        list_stamps.append(stamps)

    return tf.stack(list_stamps, 0)


def str_md5(string):
    encryption = hashlib.md5()
    encryption.update(string)
    return encryption.hexdigest()


def align_accuracy(preds, labels):
    mask = tf.cast(labels > 0, dtype=tf.float32)

    if preds.ndim == 3:
        preds = tf.argmax(preds, axis=-1, output_type=tf.int32)
    else:
        assert preds.ndim == 2
    results = tf.cast(tf.equal(preds, labels), tf.float32)

    results *= mask
    acc = tf.reduce_sum(results) / tf.reduce_sum(mask)

    return acc


def get_predicts(P_output):
    musk = tf.cast(tf.reduce_sum(tf.abs(P_output), -1) > 0, tf.float32)
    res = tf.argmax(P_output * musk[:, :, None], axis=-1, output_type=tf.int32)

    return res

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

def ctc_loss(logits, len_logits, labels, len_labels):
    """
    No valid path found: It is possible that no valid path is found if the
    activations for the targets are zero.
    """
    ctc_loss = tf.nn.ctc_loss(
        labels,
        logits,
        label_length=len_labels,
        logit_length=len_logits,
        logits_time_major=False,
        blank_index=-1)

    return ctc_loss


def evaluate(feature, dataset, dev_size, model, beam_size=0, with_stamp=True, wfst=None):
    list_acc = []

    num_processed = 0
    total_cer_dist = 0
    total_cer_len = 0
    total_res_len = 0
    for batch in feature:
        uttids, x = batch
        logits = model(x)
        # logits, logits_bounds = model(x)
        aligns = dataset.get_attrs('align', uttids.numpy())
        trans = dataset.get_attrs('trans', uttids.numpy())

        if not wfst:
            if with_stamp:
                stamps = dataset.get_attrs('stamps', uttids.numpy())
                indices = stamps2indices(stamps, 'middle')
                _logits = tf.gather_nd(logits, indices)
                preds = get_predicts(_logits)
                acc = align_accuracy(logits, aligns)
            else:
                if beam_size > 0:
                    preds = beam_search_MAP(logits, beam_size) # actually is align
                    acc = align_accuracy(preds, aligns)
                else:
                    preds = get_predicts(logits)
                    acc = align_accuracy(logits, aligns)
                # bounds = get_predicts(logits_bounds)
                # stamps = bounds2stamps(bounds)
                # indices = stamps2indices(stamps, 'middle')
                # _logits = tf.gather_nd(logits, indices)
                # preds = get_predicts(_logits)
                # acc = align_accuracy(logits, aligns)
        else:
            lens = tf.reduce_sum(tf.cast(tf.reduce_sum(x, -1) > 0, tf.int32), -1)
            distribution = tf.nn.softmax(logits)
            list_decoded = []
            for distrib, l in zip(distribution, lens):
                decoded = wfst.decode(distribution)
                list_decoded.append(decoded[:l])
            preds = list_pad(list_decoded)
            # acc = align_accuracy(decoded, aligns)
            acc = 0

        list_acc.append(acc)

        batch_cer_dist, batch_cer_len, batch_res_len = batch_cer(preds.numpy(), trans)
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len
        total_res_len += batch_res_len

        num_processed += len(x)

    cer = total_cer_dist/total_cer_len
    fer = 1-np.mean(list_acc)
    over_fire_rate = total_res_len/total_cer_len
    print('with ground stamps: {} dev FER: {:.3f}\t dev PER: {:.3f}\t over_fire_rate: {:.2f}\t{} / {}'.format(
           with_stamp, fer, cer, over_fire_rate, num_processed, dev_size))

    return fer, cer


def monitor(sample, model):
    x = np.array([sample['feature']], dtype=np.float32)
    logits = model(x)
    # logits, _ = model(x)
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


# def R_value(res_stamps, ref_stamps, region=2):
#     """
#     res_stamps, res_stamps: 1-dim np.array
#     region: the left and right tolerant region size
#     Demo:
#         res_stamps = [4,7,10,14,16]
#         ref_stamps = [3,5,9,11,15]
#         R_value(res_stamps, ref_stamps, region=2)
#     """
#     print('ref_stamps:', ref_stamps)
#     print('res_stamps:', res_stamps)
#     N_ref = len(ref_stamps)
#     N_f = len(res_stamps)
#     N_hit = 0
#
#     _left = 0
#     for i, stamp in enumerate(ref_stamps):
#         left = max(_left, stamp-region)
#         try:
#             right = min(stamp+region, (stamp+ref_stamps[i+1])/2 + 0.01)
#         except IndexError:
#             right = stamp+region
#         # print('left:', left, 'right:', right)
#
#         for j, _stamp in enumerate(res_stamps):
#             if _stamp < left:
#                 continue
#             elif _stamp > right:
#                 j = j-1
#                 break
#             else:
#                 N_hit += 1
#                 # print('hit:', ref_stamps[i], res_stamps[j])
#                 break
#         res_stamps = res_stamps[j+1:]
#         _left = right
#
#     HR = N_hit / N_ref
#     OS = N_f / N_ref - 1
#     r1 = math.sqrt(math.pow(1-HR, 2) + math.pow(OS, 2))
#     r2 = (HR - OS - 1) / 1.414
#     R = 1 - (abs(r1) + abs(r2)) / 2
#     print('r-value:', R, '\n')
#
#     return R


def P_align(bounds, seg_bnds, tolerance_window=2):
    #Precision
    hit = 0.0
    for bound in seg_bnds:
        for l in range(tolerance_window + 1):
            if (bound + l in bounds) and (bound + l > 0):
                hit += 1
                break
            elif (bound - l in bounds) and (bound - l > 0):
                hit += 1
                break
    return hit / (len(seg_bnds))


def R_align(bounds, seg_bnds, tolerance_window=2):
    #Recall
    hit = 0.0
    for bound in bounds:
        for l in range(tolerance_window + 1):
            if (bound + l in seg_bnds) and (bound + l > 0):
                hit += 1
                break
            elif (bound - l in seg_bnds) and (bound - l > 0):
                hit += 1
                break

    return hit / (len(bounds))


def R_value(ref_stamps, res_stamps, region=2):
    u_p = P_align(bounds=ref_stamps, seg_bnds=res_stamps, tolerance_window=region)
    u_r = R_align(bounds=ref_stamps, seg_bnds=res_stamps, tolerance_window=region)

    if u_r * u_p == 0:
        R = -1.
    else:
        u_os = u_r/u_p - 1
        r1 = math.fabs(math.sqrt((1-u_r)*(1-u_r) + math.pow(u_os, 2)))
        r2 = math.fabs( (u_r - 1 - u_os)/math.sqrt(2))
        R = 1 - (r1 + r2) / 2

    return u_r, u_os, R


def beam_search_MAP(logits, beam_size=20, lp=50.0):
    """
    beam search for better preds (not alignment) with  MAP.
    We add length penalty to overcome poor precision,
    one error predict within the boundry will bring two edit errors.

    logits: b x t x v
    marks_token: B * beam_size, T
    """
    inf = 1e10
    distribution = tf.nn.softmax(logits)
    B, T, V = distribution.shape
    aligns = tf.zeros([B * beam_size, 0], tf.int32)
    scores = tf.constant([0.0] + [-inf]*(beam_size-1), dtype=tf.float32) # [beam_size]
    scores = tf.tile(scores, multiples=[B]) # [B x beam_size]
    base_indices = tf.reshape(tf.tile(tf.range(B)[:, None], multiples=[1, beam_size]), [-1])
    preds_prev = -1 * tf.ones([B * beam_size, beam_size], tf.int32)
    lengths = tf.zeros([B * beam_size], tf.int32)
    # marks_token = tf.zeros([B * beam_size, 0], tf.int32)
    prev = time()
    for t in range(T):
        p_prior = tf.ones([B*beam_size, V]) / V
        p_past = tf.ones([B*beam_size, V]) / V
        p_cur = tf.reshape(tf.tile(distribution[:, t, :], [1, beam_size]), [B*beam_size, V])
        p_log = tf.math.log(p_past) + tf.math.log(p_cur) - tf.math.log(p_prior)

        scores_cur, preds_cur = tf.nn.top_k(p_log, k=beam_size, sorted=True)

        # current scores
        scores = scores[:, None] + scores_cur # [B x beam_size, beam_size]
        scores = tf.reshape(scores, [B, beam_size ** 2])

        # current predicts
        marks_cur = tf.cast(tf.not_equal(preds_cur, preds_prev), tf.int32)

        # length penalty
        lengths = lengths[:, None] + marks_cur
        lp_score = tf.reshape(tf.pow((5+tf.cast(lengths, tf.float32))/6, lp), [B, beam_size ** 2])
        # lp_score = 1.0
        scores /= lp_score

        # pruning
        _, k_indices = tf.nn.top_k(scores, k=beam_size)
        k_indices = base_indices * beam_size * beam_size + tf.reshape(k_indices, [-1]) # [B x beam_size]

        # # update marks_token
        # marks_cur = tf.reshape(marks_cur, [-1])
        # marks_cur = tf.gather(marks_cur, k_indices)
        # marks_token = tf.gather(marks_token, k_indices // beam_size)
        # marks_token = tf.concat([marks_token, marks_cur[:, None]], 1)

        # update lengths
        lengths = tf.reshape(lengths, [-1])
        lengths = tf.gather(lengths, k_indices)

        # print('lengths:', (lengths - tf.reduce_sum((marks_token), -1)).numpy())

        # Update scores
        scores = tf.reshape(scores, [-1])
        scores = tf.gather(scores, k_indices)

        # update preds
        preds_prev = preds_cur
        preds_cur = tf.reshape(preds_cur, [-1])
        preds_cur = tf.gather(preds_cur, k_indices)
        # k_indices: [0~B x beam_size x beam_size], preds: [0~B x beam_size]
        aligns = tf.gather(aligns, k_indices // beam_size)
        aligns = tf.concat([aligns, preds_cur[:, None]], -1)

    print(time() - prev, 's')
    prev = time()

    aligns = aligns[::beam_size, :]
    # marks_token = marks_token[::beam_size, :]
    # lengths = lengths[::beam_size]
    # max_len = tf.reduce_max(lengths)
    # predicts = []
    # for b in range(B):
    #     predict = tf.reshape(tf.gather(aligns[b, :], tf.where(marks_token[b, :]>0)), [-1])
    #     pad = tf.zeros([max_len - lengths[b]], tf.int32)
    #     predicts.append(tf.concat([predict, pad], 0))
    # tf.stack(predicts, 0)

    return aligns


def store_2d(array, fw):
    fw.write(pack('I', len(array)))
    for i, distrib in enumerate(array):
        for p in distrib:
            p = pack('f', p)
            fw.write(p)


class ArkReader(object):
    '''
    Class to read Kaldi ark format. Each time, it reads one line of the .scp
    file and reads in the corresponding features into a numpy matrix. It only
    supports binary-formatted .ark files. Text and compressed .ark files are not
    supported. The inspiration for this class came from pdnn toolkit (see
    licence at the top of this file) (https://github.com/yajiemiao/pdnn)
    '''

    def __init__(self, scp_path):
        '''
        ArkReader constructor

        Args:
            scp_path: path to the .scp file
        '''

        self.scp_position = 0
        fin = open(scp_path, "r", errors='ignore')
        self.dict_scp = {}
        line = fin.readline()
        while line != '' and line != None:
            uttid, path_pos = line.replace('\n', '').split(' ')
            path, pos = path_pos.split(':')
            self.dict_scp[uttid] = (path, pos)
            line = fin.readline()

        fin.close()

    def read_utt_data(self, uttid):
        '''
        read data from the archive

        Args:
            index: index of the utterance that will be read

        Returns:
            a numpy array containing the data from the utterance
        '''
        ark_read_buffer = open(self.dict_scp[uttid][0], 'rb')
        ark_read_buffer.seek(int(self.dict_scp[uttid][1]), 0)
        header = unpack('<xcccc', ark_read_buffer.read(5))
        if header[0] != b"B":
            print("Input .ark file is not binary")
            exit(1)
        if header == (b'B', b'C', b'M', b' '):
            # print('enter BCM')
            g_min_value, g_range, g_num_rows, g_num_cols = unpack('ffii', ark_read_buffer.read(16))
            utt_mat = np.zeros([g_num_rows, g_num_cols], dtype=np.float32)
            #uint16 percentile_0; uint16 percentile_25; uint16 percentile_75; uint16 percentile_100;
            per_col_header = []
            for i in range(g_num_cols):
                per_col_header.append(unpack('HHHH', ark_read_buffer.read(8)))
                #print per_col_header[i]

            tmp_mat = np.frombuffer(ark_read_buffer.read(g_num_rows * g_num_cols), dtype=np.uint8)

            pos = 0
            for i in range(g_num_cols):
                p0 = float(g_min_value + g_range * per_col_header[i][0] / 65535.0)
                p25 = float(g_min_value + g_range * per_col_header[i][1] / 65535.0)
                p75 = float(g_min_value + g_range * per_col_header[i][2] / 65535.0)
                p100 = float(g_min_value + g_range * per_col_header[i][3] / 65535.0)

                d1 = float((p25 - p0) / 64.0)
                d2 = float((p75 - p25) / 128.0)
                d3 = float((p100 - p75) / 63.0)
                for j in range(g_num_rows):
                    c = tmp_mat[pos]
                    if c <= 64:
                        utt_mat[j][i] = p0 + d1 * c
                    elif c <= 192:
                        utt_mat[j][i] = p25 + d2 * (c - 64)
                    else:
                        utt_mat[j][i] = p75 + d3 * (c - 192)
                    pos += 1
        elif header == (b'B', b'F', b'M', b' '):
            # print('enter BFM')
            m, rows = unpack('<bi', ark_read_buffer.read(5))
            n, cols = unpack('<bi', ark_read_buffer.read(5))
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=np.float32)
            utt_mat = np.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

        return utt_mat


def list_pad(list_t):
    list_len = [len(t) for t in list_t]
    max_len = max(list_len)
    list_paded = []
    for t in list_t:
        list_paded.append(tf.concat([t, [0]*(max_len-len(t))], 0))

    return tf.stack(list_paded)


def process_raw_feature(features, args):

    left_num = args.data.left_context
    right_num = args.data.right_context
    rate = args.data.downsample
    shape = tf.shape(features)
    splices = []
    pp = tf.pad(features, [[left_num, right_num], [0, 0]])
    for i in range(left_num + right_num + 1):
        splices.append(tf.slice(pp, [i, 0], shape))
    splices = tf.concat(axis=1, values=splices)

    return splices[::rate]


def save_varibales(model):

    return {i.name: i.numpy() for i in model.trainable_variables}

def load_values(model, values):
    for i in model.trainable_variables:
        i.assign(values[i.name])
