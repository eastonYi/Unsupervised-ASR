import tensorflow as tf
import numpy as np
import logging
from collections import defaultdict
import editdistance as ed
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


def build_optimizer(learning_rate, type='adam'):
    if type == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9)
    elif type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(
            lr=learning_rate,
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


def get_model_weights(w, p, sigma):
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
