import os
import numpy as np
import logging
import collections
from tqdm import tqdm
from random import shuffle, random

from .dataProcess import audio2vector, get_audio_length, process_raw_feature
from eastonCode.dataProcessing.dataHelper import DataSet, SimpleDataLoader
from utils.dataProcess import get_alignRate

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class ASRDataSet(DataSet):
    def __init__(self, file, args, _shuffle, transform):
        self.file = file
        self.args = args
        self.transform = transform
        self._shuffle = _shuffle
        self.token2idx, self.idx2token = args.token2idx, args.idx2token

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
    for dataset with alignment, i.e. TIMIT
    csv: path/to/wav.wav,a b c d e,1000 2000 3000
    """
    def __init__(self, file, args, _shuffle, transform):
        super().__init__(file, args, _shuffle, transform)
        self.list_utterances = self.gen_utter_list(file)
        self.align_rate = get_alignRate(self.list_utterances[0].split(',')[0], self.args)

        if _shuffle:
            self.shuffle_utts()

    def __getitem__(self, idx):
        utterance = self.list_utterances[idx]
        wav, seq_label, alignment = utterance.strip().split(',')
        feat = audio2vector(wav, self.args.data.dim_raw_input, method=self.args.data.featType)
        if self.transform:
            feat = process_raw_feature(feat, self.args)

        seq_label = np.array(
            [self.token2idx.get(word, self.token2idx['sil'])
             for word in seq_label.split(' ')],
            dtype=np.int32)

        alignment = np.array([int(int(align)/self.align_rate) for align in alignment.split(' ')], dtype=np.int32)

        seq_label = self.get_full_align(seq_label, alignment, len(feat))

        sample = {'id': wav, 'feature': feat, 'label': seq_label, 'align': alignment}

        return sample

    def get_full_align(self, seq_label, align, len_feature=None):
        time = 0
        full_align = []
        for label, now in zip(seq_label, align):
            duration = now - time
            full_align.extend([label]*duration)
            time = now

        if len_feature:
            pad = abs(len_feature - len(full_align))
            full_align.extend([label]*pad)
            full_align = full_align[:len_feature]

        return np.array(full_align, dtype=np.int32)

    def get_dataset_ngram(self, n, k, savefile=None, split=5000):
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
                _, seq_label, _ = utt.strip().split(',')
                ngram = get_N_gram(iter_in_sent(seq_label), n)
                ngrams += ngram

            ngrams_global += dict(ngrams.most_common(2*k))

        if savefile:
            with open(savefile, 'w') as fw:
                for ngram, num in ngrams_global.most_common(k):
                    line = '{}:{}'.format(ngram, num)
                    fw.write(line+'\n')

        return ngrams_global
