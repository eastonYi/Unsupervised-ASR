# coding=utf-8

import numpy as np
import logging
from collections import defaultdict, Counter
from random import shuffle
import multiprocessing
import tensorflow as tf
from pathlib import Path
from abc import ABCMeta, abstractmethod

from .dataProcess import audio2vector, process_raw_feature, splice, down_sample
from .tools import align2stamp, align2bound

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class DataSet:
    __metaclass__ = ABCMeta
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx):
        """
        """

    @abstractmethod
    def __len__(self):
        """
        """
    def __call__(self, idx):
        return self.__getitem__(idx)


class ASRDataSet(DataSet):
    def __init__(self,file,args,_shuffle,transform):
        self.file = file
        self.args = args
        self.transform = transform
        self._shuffle = _shuffle
        self.token2idx,self.idx2token = args.token2idx,args.idx2token

    @staticmethod
    def gen_utter_list(file):

        return list(open(file).readlines())

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def __len__(self):
        return len(self.list_utterances)


class ASR_align_DataSet(ASRDataSet):
    """
    for dataset with alignment, i.e. TIMIT
    needs:
        vocab.txt remains the index of phones in phones.txt !!
        - align_file
            uttid phone_id phone_id ...
        - trans_file

        - uttid2wav.txt
            uttid wav
        - vocab.txt (used for model output)
            phone
        -
    """
    def __init__(self, trans_file, align_file, uttid2wav, feat_len_file, args, _shuffle, transform):
        super().__init__(align_file, args, _shuffle, transform)
        self.dict_wavs = self.load_uttid2wav(uttid2wav)
        self.list_uttids = list(self.dict_wavs.keys())
        self.dict_trans = self.load_trans(trans_file) if trans_file else None
        self.dict_aligns = self.load_aligns(align_file, feat_len_file) if align_file else None

        if _shuffle:
            shuffle(self.list_uttids)

    def __getitem__(self, id):
        uttid = self.list_uttids[id]
        wav = self.dict_wavs[uttid]

        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        try:
            trans = self.dict_trans[uttid]
        except:
            trans = None
        try:
            align = self.dict_aligns[uttid]
            stamps = align2stamp(align)
        except:
            align = None
            stamps = None

        sample = {'uttid': uttid,
                  'feature': feat,
                  'trans': trans,
                  'align': align,
                  'stamps': stamps}

        return sample

    def get_attrs(self, attr, uttids, max_len=None):
        """
        length serves for the align attr to ensure the align's length same as feature
        """
        list_res = []
        list_len = []
        for uttid in uttids:
            if type(uttid) == bytes:
                uttid = uttid.decode('utf-8')
            if attr == 'wav':
                wav = self.dict_wavs[uttid]
                res = wav
            elif attr == 'feature':
                wav = self.dict_wavs[uttid]
                feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
                if self.transform:
                    feat = process_raw_feature(feat, self.args)
                res = feat
            elif attr == 'trans':
                trans = self.dict_trans[uttid]
                res = trans
            elif attr == 'align':
                align = self.dict_aligns[uttid]
                res = align
            elif attr == 'stamps':
                align = self.dict_aligns[uttid]
                stamps = align2stamp(align)
                res = stamps
            elif attr == 'bounds':
                align = self.dict_aligns[uttid]
                bounds = align2bound(align)
                res = bounds
            else:
                raise KeyError
            list_res.append(res)
            list_len.append(len(res))

        if attr in ('trans', 'align', 'stamps', 'bounds'):
            max_len = max(list_len) if not max_len else max_len
            list_padded = []
            for res in list_res:
                list_padded.append(np.concatenate([res, [0]*(max_len-len(res))])[: max_len])
            list_res = np.array(list_padded, np.int32)

        return list_res

    def load_uttid2wav(self, uttid2wav):
        dict_wavs = {}
        with open(uttid2wav) as f:
            for line in f:
                uttid, wav = line.strip().split()
                dict_wavs[uttid] = wav

        return dict_wavs

    def load_aligns(self, align_file, feat_len_file):
        dict_aligns = defaultdict(lambda: np.array([0]))
        dict_feat_len = {}

        with open(feat_len_file) as f:
            for line in f:
                uttid, len_feature = line.strip().split()
                dict_feat_len[uttid] = int(len_feature)
        align_rate = self.get_alignRate(align_file)

        with open(align_file) as f:
            for line in f:
                uttid, align = line.strip().split(maxsplit=1)
                len_feat = dict_feat_len[uttid]
                align = [int(i) for i in align.split()] + [1]
                # assert len(align) == len_feat + 1
                dict_aligns[uttid] = np.array(align[::align_rate][:len_feat])

        return dict_aligns


    def load_trans(self, trans_file):
        dict_trans = defaultdict(lambda: None)
        with open(trans_file) as f:
            for line in f:
                uttid, load_trans = line.strip().split(maxsplit=1)
                dict_trans[uttid] = np.array([self.token2idx[i] for i in load_trans.split()])

        return dict_trans

    def __len__(self):
        return len(self.list_uttids)

    def get_alignRate(self, align_file):
        with open(align_file) as f:
            uttid, align = f.readline().strip().split(maxsplit=1)

        wav = self.dict_wavs[uttid]
        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        align = align.split()

        return int(np.round(len(align)/len(feat)))


class ASR_align_ArkDataSet(ASRDataSet):
    """
    for dataset with alignment, i.e. TIMIT
    needs:
        vocab.txt remains the index of phones in phones.txt !!
        - align_file
            uttid phone_id phone_id ...
        - trans_file

        - uttid2wav.txt
            uttid wav
        - vocab.txt (used for model output)
            phone
        -
        adjust_trans: 'add_sos', 'add_eos', 'add_sos_eos'
    """
    def __init__(self, scp_file, trans_file, align_file, feat_len_file, args,
                 _shuffle, transform, adjust_trans=None):
        super().__init__(align_file, args, _shuffle, transform)
        from .tools import ArkReader
        self.reader = ArkReader(scp_file)
        self.list_uttids = list(self.reader.dict_scp.keys())
        self.dict_trans = self.load_trans(trans_file, adjust_trans) if trans_file else None
        self.dict_aligns = self.load_aligns(align_file, feat_len_file) if align_file else None

        if _shuffle:
            shuffle(self.list_uttids)

    def __getitem__(self, id):
        uttid = self.list_uttids[id]

        feat = self.reader.read_utt_data(uttid)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        try:
            trans = self.dict_trans[uttid]
        except:
            trans = None
        try:
            align = self.dict_aligns[uttid]
            stamps = align2stamp(align)
        except:
            align = None
            stamps = None

        sample = {'uttid': uttid,
                  'feature': feat,
                  'trans': trans,
                  'align': align,
                  'stamps': stamps}

        return sample

    def get_attrs(self, attr, uttids, max_len=None):
        """
        length serves for the align attr to ensure the align's length same as feature
        """
        list_res = []
        list_len = []
        for uttid in uttids:
            if type(uttid) == bytes:
                uttid = uttid.decode('utf-8')
            if attr == 'wav':
                wav = self.dict_wavs[uttid]
                res = wav
            elif attr == 'feature':
                wav = self.dict_wavs[uttid]
                feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
                if self.transform:
                    feat = process_raw_feature(feat, self.args)
                res = feat
            elif 'trans' in attr:
                '''
                trans, trans_sos, trans_eos, trans_sos_eos
                '''
                try:
                    trans = self.dict_trans[uttid]
                    if 'sos' in attr:
                        trans = np.concatenate([[self.token2idx['<sos>']], trans], 0)
                    if 'eos' in attr:
                        trans = np.concatenate([trans, [self.token2idx['<eos>']]], 0)
                except:
                    print('Not found {}'.format(uttid))
                    trans = np.array([4])
                res = trans
            elif attr == 'align':
                align = self.dict_aligns[uttid]
                res = align
            elif attr == 'stamps':
                align = self.dict_aligns[uttid]
                stamps = align2stamp(align)
                res = stamps
            elif attr == 'bounds':
                align = self.dict_aligns[uttid]
                bounds = align2bound(align)
                res = bounds
            else:
                raise KeyError
            list_res.append(res)
            list_len.append(len(res))

        if attr not in ('wav', 'feature'):
            max_len = max(list_len) if not max_len else max_len
            list_padded = []
            for res in list_res:
                list_padded.append(np.concatenate([res, [0]*(max_len-len(res))])[: max_len])
            list_res = np.array(list_padded, np.int32)

        return list_res

    def load_aligns(self, align_file, feat_len_file):
        dict_aligns = defaultdict(lambda: np.array([0]))
        dict_feat_len = {}

        with open(feat_len_file) as f:
            for line in f:
                uttid, len_feature = line.strip().split()
                dict_feat_len[uttid] = int(len_feature)
        align_rate = self.get_alignRate(align_file)

        with open(align_file) as f:
            for line in f:
                uttid, align = line.strip().split(maxsplit=1)
                len_feat = dict_feat_len[uttid]
                align = [int(i) for i in align.split()] + [1]
                # assert len(align) == len_feat + 1
                dict_aligns[uttid] = np.array(align[::align_rate][:len_feat])

        return dict_aligns

    def load_trans(self, trans_file, adjust_trans):
        # dict_trans = defaultdict(lambda: None)
        dict_trans = {}
        with open(trans_file) as f:
            for line in f:
                uttid, load_trans = line.strip().split(maxsplit=1)
                dict_trans[uttid] = np.array([self.token2idx[i] for i in load_trans.split()])

        return dict_trans

    def __len__(self):
        return len(self.list_uttids)

    def get_alignRate(self, align_file):
        with open(align_file) as f:
            uttid, align = f.readline().strip().split(maxsplit=1)

        wav = self.dict_wavs[uttid]
        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        align = align.split()

        return int(np.round(len(align)/len(feat)))


class ASR_classify_DataSet(ASRDataSet):

    def __init__(self, dir_wavs, class_file, args, _shuffle, transform):
        super().__init__(class_file, args, _shuffle, transform)
        self.dict_wavs = self.load_wav(dir_wavs)
        self.dict_y, self.dict_class = self.load_y(class_file)
        self.list_uttids = list(self.dict_y.keys())

        if _shuffle:
            shuffle(self.list_uttids)

    def __getitem__(self, id):
        uttid = self.list_uttids[id]
        wav = self.dict_wavs[uttid]
        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        y = self.dict_y[uttid]

        sample = {'uttid': uttid,
                  'feature': feat,
                  'class': y}

        return sample

    def load_y(self, class_file):
        dict_y = {}
        dict_class = {}
        with open(class_file) as f:
            for line in f:
                uttid, y = line.strip().split()
                if y not in dict_class.keys():
                    dict_class[y] = len(dict_class)
                dict_y[uttid] = dict_class[y]

        return dict_y, dict_class

    def get_y(self, uttids):
        list_y = []
        for uttid in uttids:
            if type(uttid) == bytes:
                uttid = uttid.decode('utf-8')
            y = self.dict_y[uttid]
            list_y.append(y)

        return np.array(list_y, np.int32)

    def load_wav(self, dir_wavs):
        dict_wavs = {}
        wav_path = Path(dir_wavs)
        for wav_file in wav_path.glob('*.wav'):
            uttid = str(wav_file.name)[:-4]
            dict_wavs[uttid] = str(wav_file)

        return dict_wavs

    def __len__(self):
        return len(self.list_uttids)


class ASR_classify_ArkDataSet(ASRDataSet):

    def __init__(self, scp_file, class_file, args, _shuffle):
        super().__init__(class_file, args, _shuffle, transform=False)
        from .tools import ArkReader
        self.reader = ArkReader(scp_file)
        self.dict_y, self.dict_class = self.load_y(class_file)
        self.list_uttids = list(self.dict_y.keys())
        if _shuffle:
            shuffle(self.list_uttids)

    def __getitem__(self, id):
        uttid = self.list_uttids[id]
        feat = self.reader.read_utt_data(uttid)
        # feat = down_sample(splice(feat, 2, 0), 3)
        y = self.dict_y[uttid]

        sample = {'uttid': uttid,
                  'feature': feat,
                  'class': y}

        return sample

    def load_y(self, class_file):
        dict_y = {}
        dict_class = {}
        with open(class_file) as f:
            for line in f:
                uttid, y = line.strip().split()
                if y not in dict_class.keys():
                    dict_class[y] = len(dict_class)
                dict_y[uttid] = dict_class[y]

        return dict_y, dict_class

    def get_y(self, uttids):
        list_y = []
        for uttid in uttids:
            if type(uttid) == bytes:
                uttid = uttid.decode('utf-8')
            y = self.dict_y[uttid]
            list_y.append(y)

        return np.array(list_y, np.int32)

    def __len__(self):
        return len(self.list_uttids)


class LMDataSet(DataSet):
    """
    dataset for language model. Refer to the PTB dataset
    """
    def __init__(self, list_files, args, _shuffle):
        self.list_files = list_files
        self.args = args
        self._shuffle = _shuffle
        self.token2idx, self.idx2token = args.token2idx, args.idx2token
        self.start_id = 1
        if _shuffle:
            shuffle(self.list_files)
        self.size_dataset = self.get_size()

    def __getitem__(self, idx):
        pass

    def __call__(self):
        return self.__iter__()

    def __len__(self):
        return self.size_dataset

    def get_size(self):
        num_lines = 0
        for filename in self.list_files:
            num_lines += sum(1 for line in open(filename))

        return num_lines

    def __iter__(self):
        """
        (Pdb) i[0]
        [1,18,2,36,1,17,7,9,9,6,25,28,3,5,14,1,11,32,24,16,26,22,3,1,16,15,1,18,8,3,1,4]
        (Pdb) i[1]
        [18,2,36,1,17,7,9,9,6,25,28,3,5,14,1,11,32,24,16,26,22,3,1,16,15,1,18,8,3,1,4,1]
        """
        for filename in self.list_files:
            with open(filename) as f:
                for line in f:
                    line = line.strip()
                    if len(line) > self.args.model.D.max_label_len:
                        continue
                    text_ids = [self.token2idx[word] for word in line]
                    src_ids = text_ids[:-1]
                    tar_ids = text_ids[1:]
                    # sample = {'src': src_ids, 'tar': tar_ids}

                    yield src_ids, tar_ids


class PTB_LMDataSet(LMDataSet):
    """
    dataset for language model. Refer to the PTB dataset
    """
    def __init__(self, list_files, args, _shuffle):
        super().__init__(list_files, args, _shuffle)
        self.start_id = args.token2idx['<sos>']
        self.end_id = args.token2idx['<eos>']

    def __iter__(self):
        for filename in self.list_files:
            with open(filename) as f:
                for line in f:
                    line = line.strip().split()
                    if len(line) > self.args.list_bucket_boundaries[-1]:
                        continue
                    text_ids = [self.token2idx[word] for word in line]
                    src_ids = [self.start_id] + text_ids
                    tar_ids = text_ids + [self.end_id]

                    yield src_ids, tar_ids


class TextDataSet(LMDataSet):

    def __iter__(self):
        """
        (Pdb) i
        [1,18,2,36,1,17,7,9,9,6,25,28,3,5,14,1,11,32,24,16,26,22,3,1,16,15,1,18,8,3,1,4,1]
        """
        for filename in self.list_files:
            with open(filename) as f:
                for line in f:
                    line = line.strip().split()
                    if len(line) > self.args.model.D.max_label_len:
                        continue
                    text_ids = [self.token2idx[word] for word in line]

                    yield text_ids


def load_dataset(max_length, max_n_examples, max_vocab_size=2048, data_dir=''):
    print("loading dataset...")
    lines = []
    finished = False

    for i in range(99):
        path = data_dir+("/training-monolingual.tokenized.shuffled/news.en-{}-of-00100".format(str(i+1).zfill(5)))
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = tuple(line[:-1])

                if len(line) > max_length:
                    line = line[:max_length]

                lines.append(line + ( ("`",)*(max_length-len(line)) ) )

                if len(lines) == max_n_examples:
                    finished = True
                    break
        if finished:
            break

    np.random.shuffle(lines)

    counts = Counter(char for line in lines for char in line)

    charmap = {'unk':0}
    inv_charmap = ['unk']

    for char,count in counts.most_common(max_vocab_size-1):
        if char not in charmap:
            charmap[char] = len(inv_charmap)
            inv_charmap.append(char)

    filtered_lines = []
    for line in lines:
        filtered_line = []
        for char in line:
            if char in charmap:
                filtered_line.append(char)
            else:
                filtered_line.append('unk')
        filtered_lines.append(tuple(filtered_line))

    for i in range(10):
        print(''.join(filtered_lines[i]))

    print("loaded {} lines in dataset".format(len(lines)))

    return filtered_lines, charmap, inv_charmap


def make_32x32_dataset(dataset, batch_size, drop_remainder=True, shuffle=True, repeat=1):

    if dataset == 'mnist':
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'fashion_mnist':
        (train_images, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        train_images.shape = train_images.shape + (1,)
    elif dataset == 'cifar10':
        (train_images, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    else:
        raise NotImplementedError

    @tf.function
    def _map_fn(img):
        img = tf.image.resize(img, [32, 32])
        img = tf.clip_by_value(img, 0, 255)
        img = img / 127.5 - 1
        return img

    dataset = memory_data_batch_dataset(train_images,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)
    img_shape = (32, 32, train_images.shape[-1])
    len_dataset = len(train_images) // batch_size

    return dataset, img_shape, len_dataset


def batch_dataset(dataset,
                  batch_size,
                  drop_remainder=True,
                  n_prefetch_batch=1,
                  filter_fn=None,
                  map_fn=None,
                  n_map_threads=None,
                  filter_after_map=False,
                  shuffle=True,
                  shuffle_buffer_size=None,
                  repeat=None):
    # set defaults
    if n_map_threads is None:
        n_map_threads = multiprocessing.cpu_count()
    if shuffle and shuffle_buffer_size is None:
        shuffle_buffer_size = max(batch_size * 128, 2048)  # set the minimum buffer size as 2048

    # [*] it is efficient to conduct `shuffle` before `map`/`filter` because `map`/`filter` is sometimes costly
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)

    if not filter_after_map:
        if filter_fn:
            dataset = dataset.filter(filter_fn)

        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

    else:  # [*] this is slower
        if map_fn:
            dataset = dataset.map(map_fn, num_parallel_calls=n_map_threads)

        if filter_fn:
            dataset = dataset.filter(filter_fn)

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset = dataset.repeat(repeat).prefetch(n_prefetch_batch)

    return dataset


def memory_data_batch_dataset(memory_data,
                              batch_size,
                              drop_remainder=True,
                              n_prefetch_batch=1,
                              filter_fn=None,
                              map_fn=None,
                              n_map_threads=None,
                              filter_after_map=False,
                              shuffle=True,
                              shuffle_buffer_size=None,
                              repeat=None):
    """Batch dataset of memory data.

    Parameters
    ----------
    memory_data : nested structure of tensors/ndarrays/lists

    """
    dataset = tf.data.Dataset.from_tensor_slices(memory_data)
    dataset = batch_dataset(dataset,
                            batch_size,
                            drop_remainder=drop_remainder,
                            n_prefetch_batch=n_prefetch_batch,
                            filter_fn=filter_fn,
                            map_fn=map_fn,
                            n_map_threads=n_map_threads,
                            filter_after_map=filter_after_map,
                            shuffle=shuffle,
                            shuffle_buffer_size=shuffle_buffer_size,
                            repeat=repeat)
    return dataset


def get_batch(iterator, batch_size, length):
    list_samples = []
    for _ in range(batch_size):
        sample = next(iterator)
        sample += [0] * length

        list_samples.append(sample[:length])
        if len(list_samples) >= batch_size:
            return np.array(list_samples, dtype=np.int32)


def get_bucket(length_file, num_batch_tokens, idx_init=150):
    """
    enlarge idx_init can shrink the num of buckets
    """
    print('get the dataset info')
    list_len = []
    with open(length_file) as f:
        for line in f:
            length = int(line.strip().split()[1])
            list_len.append(length)

    hist, edges = np.histogram(list_len, bins=(max(list_len)-min(list_len)+1))
    list_num = []
    list_length = []
    for num, edge in zip(hist, edges):
        list_num.append(int(num))
        list_length.append(int(np.ceil(edge)))

    def next_idx(idx, energy):
        for i in range(idx, len(list_num), 1):
            if list_length[i]*sum(list_num[idx+1:i+1]) > energy:
                return i-1
        return

    M = num_batch_tokens
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
            break
        list_boundary.append(list_length[idx])
        list_batchsize.append(int(M / list_length[idx]))

    list_boundary.append(list_length[-1])
    list_batchsize.append(int(M/list_length[-1]))

    print('suggest boundaries: \n{}'.format(','.join(map(str, list_boundary))))
    print('corresponding batch size: \n{}'.format(','.join(map(str, list_batchsize))))
