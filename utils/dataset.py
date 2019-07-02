# coding=utf-8

import os
import numpy as np
import logging
import collections
from tqdm import tqdm
from random import shuffle,random
import multiprocessing
import tensorflow as tf

from .dataProcess import audio2vector,get_audio_length,process_raw_feature
from eastonCode.dataProcessing.dataHelper import DataSet,SimpleDataLoader
from utils.dataProcess import get_alignRate

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class ASRDataSet(DataSet):
    def __init__(self,file,args,_shuffle,transform):
        self.file = file
        self.args = args
        self.transform = transform
        self._shuffle = _shuffle
        self.token2idx,self.idx2token = args.token2idx,args.idx2token

    @staticmethod
    def gen_utter_list(list_files):
        list_utter = []
        for file in list_files:
            with open(file) as f:
                list_utter.extend(f.readlines())

        return list_utter

    def shuffle_utts(self):
        shuffle(self.list_utterances)

    def __len__(self):
        return len(self.list_utterances)


class ASR_align_DataSet(ASRDataSet):
    """
    for dataset with alignment,i.e. TIMIT
    csv: path/to/wav.wav,a b c d e,1000 2000 3000
    """
    def __init__(self,file,args,_shuffle,transform):
        super().__init__(file,args,_shuffle,transform)
        self.list_utterances = self.gen_utter_list(file)
        self.align_rate = get_alignRate(self.list_utterances[0].split(',')[0], self.args)

        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self,idx):
        utterance = self.list_utterances[idx]
        wav,seq_label,alignment = utterance.strip().split(',')
        feat = audio2vector(wav,self.args.data.dim_raw_input,method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat,self.args)

        seq_label = np.array(
            [self.token2idx.get(word,self.token2idx['sil'])
             for word in seq_label.split(' ')],
            dtype=np.int32)

        alignment = np.array([int(int(align)/self.align_rate) for align in alignment.split(' ')], dtype=np.int32)

        seq_label = self.get_full_align(seq_label,alignment,len(feat))

        sample = {'id': wav, 'feature': feat, 'label': seq_label, 'align': alignment}

        return sample

    def get_full_align(self,seq_label,align,len_feature=None):
        time = -1
        full_align = []
        for label,now in zip(seq_label,align):
            duration = now - time
            full_align.extend([label]*duration)
            time = now

        if len_feature:
            pad = abs(len_feature - len(full_align))
            full_align.extend([label]*pad)
            full_align = full_align[:len_feature]

        return np.array(full_align,dtype=np.int32)

    def get_dataset_ngram(self,n,k,savefile=None,split=5000):
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

        ngrams_global = FreqDist()
        for i in range(len(self.list_utterances)//split +1):
            ngrams = FreqDist()
            text = self.list_utterances[i*split: (i+1)*split]
            for utt in tqdm(text):
                _,seq_label,_ = utt.strip().split(',')
                ngram = get_N_gram(iter_in_sent(seq_label), n)
                ngrams += ngram

            ngrams_global += dict(ngrams.most_common(2*k))

        if savefile:
            with open(savefile, 'w') as fw:
                for ngram,num in ngrams_global.most_common(k):
                    line = '{}:{}'.format(ngram,num)
                    fw.write(line+'\n')

        return ngrams_global


class LMDataSet(DataSet):
    """
    dataset for language model. Refer to the PTB dataset
    """
    def __init__(self,list_files,args,_shuffle):
        self.list_files = list_files
        self.args = args
        self._shuffle = _shuffle
        self.token2idx,self.idx2token = args.token2idx,args.idx2token
        self.start_id = 1
        if _shuffle:
            shuffle(self.list_files)
        self.size_dataset = self.get_size()

    def __getitem__(self,idx):
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
                    line = line.strip().split(',')[1].split()
                    if len(line) > self.args.list_bucket_boundaries[-1]:
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
                    if len(line) > self.args.list_bucket_boundaries[-1]:
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

    counts = collections.Counter(char for line in lines for char in line)

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
