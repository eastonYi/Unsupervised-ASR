# coding=utf-8

import os
import numpy as np
import logging
from collections import defaultdict, Counter
from tqdm import tqdm
from random import shuffle, random, randint
import multiprocessing
import tensorflow as tf

from .dataProcess import audio2vector, get_audio_length, process_raw_feature
from eastonCode.dataProcessing.dataHelper import DataSet, SimpleDataLoader
# from utils.dataProcess import get_alignRate

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


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


# class ASR_align_DataSet(ASRDataSet):
#     """
#     for dataset with alignment,i.e. TIMIT
#     csv: path/to/wav.wav,a b c d e,1000 2000 3000
#     """
#     def __init__(self, file, args, _shuffle, transform):
#         super().__init__(file, args, _shuffle, transform)
#         self.list_utterances = self.gen_utter_list(file)
#         self.align_rate = get_alignRate(self.list_utterances[0].split(',')[0], self.args)
#
#         if _shuffle:
#             self.shuffle_utts()
#
#     def __getitem__(self,idx):
#         utterance = self.list_utterances[idx]
#         wav, seq_label, stamps = utterance.strip().split(',')
#         feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
#         if self.transform:
#             feat = process_raw_feature(feat, self.args)
#
#         seq_label = np.array(
#             [self.token2idx.get(word, self.token2idx['sil'])
#              for word in seq_label.split(' ')],
#             dtype=np.int32)
#
#         stamps = np.array([int(int(stamp)/self.align_rate) for stamp in stamps.split(' ')], dtype=np.int32)
#
#         align = self.get_align(seq_label, stamps, len(feat))
#
#         sample = {'id': wav, 'feature': feat, 'align': align, 'stamps': stamps}
#
#         return sample
#
#     def get_align(self, seq_label, align, len_feature=None):
#         time = -1
#         align = []
#         for label, now in zip(seq_label, align):
#             duration = now - time
#             align.extend([label]*duration)
#             time = now
#
#         if len_feature:
#             pad = abs(len_feature - len(align))
#             align.extend([label]*pad)
#             align = align[:len_feature]
#
#         return np.array(align, dtype=np.int32)


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
    def __init__(self, trans_file, align_file, uttid2wav, args, _shuffle, transform):
        super().__init__(align_file, args, _shuffle, transform)
        self.dict_wavs = self.load_uttid2wav(uttid2wav)
        self.dict_trans = self.load_trans(trans_file)
        self.dict_aligns = self.load_aligns(align_file)
        self.list_uttids = list(self.dict_wavs.keys())
        self.align_rate = self.get_alignRate() if align_file else None

        if _shuffle:
            shuffle(self.list_uttids)

    def __getitem__(self, id):
        uttid = self.list_uttids[id]
        wav = self.dict_wavs[uttid]

        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        trans = self.dict_trans[uttid]

        try:
            align = self.dict_aligns[uttid][::self.align_rate]
            stamps = self.align2stamp(align)
            # print('align rate:', np.round(len(align)/len(feat)))
        except:
            align = None
            stamps = None

        sample = {'uttid': uttid,
                  'feature': feat,
                  'trans': trans,
                  'align': align,
                  'stamps': stamps}

        return sample

    def get_attrs(self, attr, uttids, length=None):
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
            elif attr == 'trans':
                trans = self.dict_trans[uttid]
                res = trans
            elif attr == 'align':
                try:
                    _align = self.dict_aligns[uttid]
                    align = _align[::self.align_rate]
                    if len(_align) % 2 == 0:
                        align = np.concatenate([align, [align[-1]]])
                    stamps = self.align2stamp(align)
                except:
                    align = None
                    stamps = None
                res = align
            elif attr == 'stamps':
                try:
                    align = self.dict_aligns[uttid][::self.align_rate]
                    stamps = self.align2stamp(align)
                except:
                    stamps = None
                res = stamps
            else:
                raise KeyError
            list_res.append(res)
            try:
                list_len.append(len(res))
            except:
                pass
        try:
            max_len = max(list_len)
            list_padded = []
            for res in list_res:
                list_padded.append(np.concatenate([res, [0]*(max_len-len(res))]))
            list_res = np.array(list_padded, np.int32)
        except:
            pass

        return list_res

    def load_uttid2wav(self, uttid2wav):
        dict_wavs = {}
        with open(uttid2wav) as f:
            for line in f:
                uttid, wav = line.strip().split()
                dict_wavs[uttid] = wav

        return dict_wavs

    @staticmethod
    def align2stamp(align):
        if align is not None:
            list_stamps = []
            label_prev = align[0]
            for i, label in enumerate(align):
                if label_prev != label:
                    list_stamps.append(i-1)
                label_prev = label
        else:
            list_stamps = None

        return np.array(list_stamps)

    @staticmethod
    def load_aligns(align_file):
        dict_aligns = defaultdict(lambda: None)
        if align_file:
            with open(align_file) as f:
                for line in f:
                    uttid, align = line.strip().split(maxsplit=1)
                    dict_aligns[uttid] = np.array([int(i) for i in align.split()])

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

    def get_alignRate(self):
        uttid = self.list_uttids[0]

        wav = self.dict_wavs[uttid]
        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        align = self.dict_aligns[uttid]

        return int(np.round(len(align)/len(feat)))


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
                    if len(line) > self.args.max_seq_len:
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
                    if len(line) > self.args.max_seq_len:
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
