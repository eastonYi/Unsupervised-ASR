import numpy as np
from python_speech_features import mfcc, logfbank
from collections import defaultdict


def load_vocab(path, vocab_size=None):
    with open(path, encoding='utf8') as f:
        vocab = [line.strip().split()[0] for line in f]
    vocab = vocab[:vocab_size] if vocab_size else vocab
    token2idx = defaultdict(lambda: 0)
    idx2token = {}
    token2idx.update({token: idx for idx, token in enumerate(vocab)})
    idx2token.update({idx: token for idx, token in enumerate(vocab)})

    assert len(token2idx) == len(idx2token)

    return token2idx, idx2token


def load_wavfile(wavfile):
    """
    Read a wav file using scipy.io.wavfile
    """
    if wavfile.endswith('.wav'):
        import scipy.io.wavfile as wav
        rate, sig = wav.read(wavfile)
    elif wavfile.endswith('.flac'):
        import soundfile
        sig, rate = soundfile.read(wavfile, dtype='int16')
    else:
        raise IOError('NOT support file type or not a filename: {}'.format(wavfile))
    # data_name = os.path.splitext(os.path.basename(wavfile))[0]
    return rate, sig


def get_duration(wavfile):
    """
    The duration is frames/second
    """
    rate, sig = load_wavfile(wavfile)

    return len(sig)/rate


def audio2vector(audio_filename, dim_feature, method='fbank'):
    '''
    Turn an audio file into feature representation.
    16k wav, size 283K -> len 903
    '''
    rate, sig = load_wavfile(audio_filename)
    # print('len of sig: ', len(sig))
    # Get fbank coefficients. numcep is the feature size
    if method == 'fbank':
        org_inputs = logfbank(sig, samplerate=rate, nfilt=dim_feature).astype(np.float32)
    elif method == 'mfcc':
        org_inputs = mfcc(sig, samplerate=rate, numcep=dim_feature).astype(np.float32)
    else:
        raise IOError('NOT support feature type: {}'.format(method))

    org_inputs = (org_inputs - np.mean(org_inputs)) / np.std(org_inputs)

    return org_inputs


def get_audio_length(audio_filename, dim_feature, method='fbank'):
    rate, sig = load_wavfile(audio_filename)

    return len(sig)


def add_delt(feature):
    def fea_delt1(features):
        feats_padded = np.pad(features, [[1, 1], [0, 0]], "symmetric")
        feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")

        row, col = np.shape(features)
        l2 = feats_padded[:row]
        l1 = feats_padded[1: row + 1]
        r1 = feats_padded[3: row + 3]
        r2 = feats_padded[4: row + 4]
        delt1 = (r1 - l1) * 0.1 + (r2 - l2) * 0.2

        return delt1

    def fea_delt2(features):
        feats_padded = np.pad(features, [[1, 1], [0, 0]], "symmetric")
        feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")
        feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")
        feats_padded = np.pad(feats_padded, [[1, 1], [0, 0]], "symmetric")

        row, col = np.shape(features)
        l4 = feats_padded[:row]
        l3 = feats_padded[1: row + 1]
        l2 = feats_padded[2: row + 2]
        l1 = feats_padded[3: row + 3]
        c = feats_padded[4: row + 4]
        r1 = feats_padded[5: row + 5]
        r2 = feats_padded[6: row + 6]
        r3 = feats_padded[7: row + 7]
        r4 = feats_padded[8: row + 8]

        delt2 = - 0.1 * c - 0.04 * (l1 + r1) + 0.01 * (l2 + r2) + 0.04 * (l3 + l4 + r4 + r3)

        return delt2

    fb = []
    fb.append(feature)
    delt1 = fea_delt1(feature)
    # delt1 = np_fea_delt(feature)
    fb.append(delt1)
    # delt2 = np_fea_delt(delt1)
    delt2 = fea_delt2(feature)
    fb.append(delt2)
    fb = np.concatenate(fb, 1)

    return fb


def splice(features, left_num, right_num):
    """
    [[1,1,1], [2,2,2], [3,3,3], [4,4,4], [5,5,5], [6,6,6], [7,7,7]]
    left_num=0, right_num=2:
        [[1 1 1 2 2 2 3 3 3]
         [2 2 2 3 3 3 4 4 4]
         [3 3 3 4 4 4 5 5 5]
         [4 4 4 5 5 5 6 6 6]
         [5 5 5 6 6 6 7 7 7]
         [6 6 6 7 7 7 0 0 0]
         [7 7 7 0 0 0 0 0 0]]
    """
    dtype = features.dtype
    len_time, dim_raw_feat = features.shape
    stacked_feat = [1]*len_time
    pad_slice = [0.0] * dim_raw_feat
    pad_left = pad_right = []
    for time in range(len_time):
        idx_left = (time-left_num) if time-left_num>0 else 0
        stacked_feat[time] = features[idx_left: time+right_num+1].tolist()
        if left_num - time > 0:
            pad_left = [pad_slice] * (left_num - time)
            stacked_feat[time] = np.concatenate(pad_left+stacked_feat[time], 0)
        elif right_num > (len_time - time - 1):
            pad_right = [pad_slice] * (right_num - len_time + time + 1)
            stacked_feat[time] = np.concatenate(stacked_feat[time]+pad_right, 0)
        else:
            stacked_feat[time] = np.concatenate(stacked_feat[time], 0)

    return np.asarray(stacked_feat, dtype=dtype)


def down_sample(features, rate):

    return features[::rate]


def process_raw_feature(fea, args):
    # 1-D, 2-D
    if args.data.add_delta:
        fea = add_delt(fea)

    # Splice
    # fea = splice(fea, left_num=0, right_num=args.data.num_context)
    fea = splice(fea, left_num=args.data.left_context, right_num=args.data.right_context)

    # downsample
    fea = down_sample(fea, rate=args.data.downsample)

    return fea


def get_alignRate(wav_file, args):
    len_sig = get_audio_length(wav_file, args.data.dim_raw_input)
    feat = audio2vector(wav_file, args.data.dim_raw_input, method=args.data.featType)
    feat = process_raw_feature(feat, args)
    len_feat = len(feat)

    return len_sig / len_feat


def get_N_gram(iterator, n):
    """
    return :
        [(('ih', 'sil', 'k'), 1150),
         (('ih', 'n', 'sil'), 1067),
         ...],
         num of all the n-gram, i.e. num of tokens
    """
    from nltk import ngrams, FreqDist

    _n_grams = FreqDist(ngrams(iterator, n))

    return _n_grams
