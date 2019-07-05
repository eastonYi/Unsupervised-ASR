import tensorflow as tf
import numpy as np
import editdistance as ed
from pathlib import Path
from tqdm import tqdm
from random import shuffle
import sys

from utils.dataProcess import get_N_gram


class warmup_exponential_decay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, peak, decay_steps):
        super().__init__()
        self.warmup_steps = warmup_steps*1.0
        self.peak = peak*1.0
        self.decay_steps = decay_steps*1.0
    # def __call__(self, step):
    #     arg1 = tf.math.rsqrt(step)
    #     arg2 = step * (self.warmup_steps ** -1.5)
    #
    #     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    def __call__(self, step):
        warmup_steps, peak, decay_steps = self.warmup_steps, self.peak, self.decay_steps
        # warmup_steps = tf.to_float(warmup_steps)
        global_step = tf.cast(step, tf.float32)

        return tf.where(global_step <= warmup_steps,
                        peak * global_step / warmup_steps,
                        peak * 0.5 ** ((global_step - warmup_steps) / decay_steps))


class TFData:
    """
    test on TF2.0-alpha
    """
    def __init__(self, dataset, dataAttr, dir_save, args, size_file=5000000, max_feat_len=3000):
        self.dataset = dataset
        self.dataAttr =  dataAttr # ['feature', 'label', 'align']
        self.max_feat_len = max_feat_len
        self.dir_save = dir_save
        self.args = args
        self.size_file = size_file
        self.dim_feature = dataset[0]['feature'].shape[-1] \
            if dataset else self.read_tfdata_info(dir_save)['dim_feature']

    def save(self, name):
        num_token = 0
        num_damaged_sample = 0

        def serialize_example(feature, label, align):
            atts = {
                'feature': self._bytes_feature(feature.tostring()),
                'label': self._bytes_feature(label.tostring()),
                'align': self._bytes_feature(align.tostring()),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=atts))

            return example_proto.SerializeToString()

        def generator():
            for features, _ in zip(self.dataset, tqdm(range(len(self.dataset)))):
                # print(features['feature'].shape)
                yield serialize_example(features['feature'], features['label'], features['align'])

        dataset_tf = tf.data.Dataset.from_generator(
            generator=generator,
            output_types=tf.string,
            output_shapes=())

        writer = tf.data.experimental.TFRecordWriter(str(self.dir_save/'{}.recode'.format(name)))
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
        """
        list_filenames = self.fentch_filelist(self.dir_save)
        if _shuffle:
            shuffle(list_filenames)
        else:
            list_filenames.sort()

        raw_dataset = tf.data.TFRecordDataset(list_filenames)

        def _parse_function(example_proto):
            features = tf.io.parse_single_example(
                example_proto,
                # features={attr: tf.io.FixedLenFeature([], tf.string) for attr in self.dataAttr}
                features={
                    'feature': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.string),
                    'align': tf.io.FixedLenFeature([], tf.string)
                }
            )
            feature = tf.reshape(tf.io.decode_raw(features['feature'], tf.float32),
                                 [-1, self.dim_feature])[:self.max_feat_len, :]
            label = tf.io.decode_raw(features['label'], tf.int32)
            align = tf.io.decode_raw(features['align'], tf.int32)

            return feature, label, align

        features = raw_dataset.map(_parse_function)

        return features

    def get_bucket_size(self, idx_init, reuse=False):

        f_len_hist = './dataset_len_hist.txt'
        list_len = []

        if not reuse:
            dataset = self.read(_shuffle=False)
            for sample in tqdm(dataset):
                feature, *_ = sample
                list_len.append(len(feature))

            hist, edges = np.histogram(list_len, bins=(max(list_len)-min(list_len)+1))

            # save hist
            with open(f_len_hist, 'w') as fw:
                for num, edge in zip(hist, edges):
                    fw.write('{}: {}\n'.format(str(num), str(int(np.ceil(edge)))))
                    print(str(num), str(int(np.ceil(edge))))

            list_num = []
            list_length = []
            with open(f_len_hist) as f:
                for line in f:
                    num, length = line.strip().split(':')
                    list_num.append(int(num))
                    list_length.append(int(length))

        def next_idx(idx, energy):
            for i in range(idx, len(list_num), 1):
                if list_length[i]*sum(list_num[idx+1:i+1]) > energy:
                    return i-1
            return

        M = self.args.num_batch_tokens
        b0 = int(M / list_length[idx_init])
        k = b0/sum(list_num[:idx_init+1])
        energy = M/k

        list_batchsize = [b0]
        list_boundary = [list_length[idx_init]]

        idx = idx_init
        while idx < len(list_num):
            idx = next_idx(idx, energy)
            if not idx:
                break
            if idx == idx_init:
                print('enlarge the idx_init!')
                sys.exit()
            list_boundary.append(list_length[idx])
            list_batchsize.append(int(M / list_length[idx]))

        list_boundary.append(list_length[-1])
        list_batchsize.append(int(M/list_length[-1]))

        print('suggest boundaries: \n{}'.format(','.join(map(str, list_boundary))))
        print('corresponding batch size: \n{}'.format(','.join(map(str, list_batchsize))))

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


def size_variables(model):
    total_size = 0
    all_weights = {v.name: v for v in model.trainable_variables}
    for v_name in sorted(list(all_weights)):
        v = all_weights[v_name]
        v_size = int(np.prod(np.array(v.shape.as_list())))
        print("Weight    %s\tshape    %s\tsize    %d" % (v.name[:-2].ljust(80), str(v.shape).ljust(20), v_size))
        total_size += v_size
    print("Total trainable variables size: %d" % total_size)


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
    for res, ref in zip(preds, reference.numpy()):
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
        pred = D(x, mask)
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
