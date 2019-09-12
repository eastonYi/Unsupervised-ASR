#!/usr/bin/env python
# coding=utf-8
"""
author: easton
date: 19-01-22

limitations:
    - the wfst would output all the decoded words each calling time. Need to clip
    the repeat part each time. sor the wfst decode need to fix.
"""

import time
import logging
import tensorflow as tf
import numpy as np
import collections
from queue import Queue
import threading
from ctypes import c_char
from collections import defaultdict
import struct
from pathlib import Path
import sys
import random
import os

np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.nan)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')

cmd2gen_scp = lambda feats_processing_dir, tmp_dir, wav_scp, feat_scp, feat_conf: '''cd {0}
./compute-fbank-feats --config=./{4} scp:../{1}/{2} ark:- | \
./add-deltas --delta-order=2 ark:- ark:- | \
./apply-cmvn --print-args=false --norm-means=true --norm-vars=true cmvn_glob.mat ark:- ark:- | \
./copy-feats ark:- ark,scp:../{1}/feat.ark,../{1}/_feat.scp
cd ../{1}
cat _feat.scp | awk '{{print $1" '$PWD'/"$2}}' > {3}'''.format(\
feats_processing_dir, tmp_dir, wav_scp, feat_scp, feat_conf)


def load_vocab(path, vocab_size=None):
    vocab = [line.split('\n')[0].split()[0] for line in open(path)]
    vocab = vocab[:vocab_size] if vocab_size else vocab
    id_unk = vocab.index('<unk>')
    token2idx = defaultdict(lambda: id_unk)
    idx2token = defaultdict(lambda: '<unk>')
    token2idx.update({token: idx for idx, token in enumerate(vocab)})
    idx2token.update({idx: token for idx, token in enumerate(vocab)})
    if '<blk>' in vocab:
        idx2token[token2idx['<blk>']] = ''
    if '<pad>' in vocab:
        idx2token[token2idx['<pad>']] = ''

    assert len(token2idx) == len(idx2token)

    return token2idx, idx2token


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    demo:
    a ={'length': 10, 'shape': (2,3)}
    config = AttrDict(a)
    config.length #10

    for i in read_tfdata_info(args.dirs.train.tfdata).items():
        args.data.train.i[0] = i[1]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


class WFST_Decoder:

    def __init__(self,
                 decode_outs,
                 len_decode_max=200,
                 max_wfst_num=64,
                 fcdll="/Users/easton/Projects/wfst_decode/bin/libctc_wfst_lib.dylib",
                 fcfg="/Users/easton/Projects/wfst_decode/config/cfg_hkust_pinyin.json"):
        from chj_cdll import cls_speed

        start = time.time()
        self.speed = cls_speed()
        self.speed.load_cdll(fcdll)
        self.speed.check_type=0
        fcfg_=(c_char * 100)().value = bytes(fcfg, encoding="ascii")
        self.speed.cdll.CHJ_CTC_LIB_init(fcfg_)
        self.len_decode_max = len_decode_max
        self.decode_outs=decode_outs
        self.speed.set_mp("decode_outs", self.decode_outs)
        #the dict maintains the coming distribution pieces from the model output
        #{uttid: {'idx': None, 'waitFor': 1, 'pool':[(piece, pos), (piece, pos), ...]}, 'time': 0;     uttid:}
        self.dict_utt = defaultdict(lambda: {'idx': None, 'waitFor':1, 'pool':[], 'birth_time': 0.0})
        self.list_free_wfst_uttid = list(range(max_wfst_num))
        self.max_wfst_num = max_wfst_num
        logging.info('loading wfst graph with {}s'.format(time.time() - start))

    def decode(self, distribution_log, uttid, pos):
        """
        wfst_uttid: int
        flag: 0:start, 1: middle, 2:end
        """
        # add the output from acoustic model and sort the data pool
        flag = 1
        if uttid not in self.dict_utt.keys():
            try:
                # a new uttid fentches a wfst_id from list_free_wfst_uttid
                self.dict_utt[uttid]['idx'] = self.list_free_wfst_uttid[0]
                self.dict_utt[uttid]['birth_time'] = time.time()
                del self.list_free_wfst_uttid[0]
                flag = 0
            except:
                raise('the wfst maintains too much utts!')

        self.dict_utt[uttid]['pool'].append((distribution_log, pos))
        self.dict_utt[uttid]['pool'].sort(key=lambda x: int(x[1].split('_')[0]))

        # fetch all the pieces that in order
        # e.g. for [1,2,3,5,6,7], we would process [1,2,3], remain the [5,6,7]
        list_pieces = []
        for (piece, pos) in self.dict_utt[uttid]['pool']:
            if int(pos.split('_')[0]) == self.dict_utt[uttid]['waitFor']:
                list_pieces.append(piece)
                wfst_uttid = self.dict_utt[uttid]['idx']
                self.dict_utt[uttid]['waitFor'] +=1
                # if the eos piece will be decoded, the utt would be delected, and recycle the wfst_id as a free one
                if 'eos' in pos:
                    self.list_free_wfst_uttid.append(wfst_uttid)
                    # self.list_free_wfst_uttid.sort()
                    del self.dict_utt[uttid]

                    flag = 2

        # del the decoded data
        if uttid in self.dict_utt.keys():
            for _ in list_pieces:
                del self.dict_utt[uttid]['pool'][0]

        # start to decode in wfst
        if list_pieces:
            distribution_log = np.concatenate(list_pieces, 0)
            dims = np.array([distribution_log.shape[0],
                             self.len_decode_max,
                             wfst_uttid,
                             flag], dtype=np.int32)
            self.speed.set_mp("dims", dims)
            self.speed.set_mp("net_outs", distribution_log)
            self.speed.cdll.CHJ_CTC_LIB_run_one_sentence()
            len_decode = self.decode_outs[0]
            decoded = self.decode_outs[1: 1+len_decode]
        else:
            decoded = ''

        return decoded, flag


class ArkReader(object):
    def __init__(self, scp_path):
        self.scp_position = 0
        fin = open(str(scp_path), "r", errors='ignore')
        self.utt_ids = []
        self.scp_data = []
        line = fin.readline()
        while line != '' and line != None:
            utt_id, path_pos = line.replace('\n', '').split(' ')
            path, pos = path_pos.split(':')
            self.utt_ids.append(utt_id)
            self.scp_data.append((path, pos))
            line = fin.readline()

        fin.close()

    def read_utt_data(self, index):
        ark_read_buffer = open(self.scp_data[index][0], 'rb')
        ark_read_buffer.seek(int(self.scp_data[index][1]), 0)
        header = struct.unpack('<xcccc', ark_read_buffer.read(5))
        if header[0] != b"B":
            print("Input .ark file is not binary")
            exit(1)
        if header == (b'B', b'C', b'M', b' '):
            # print('enter BCM')
            g_min_value, g_range, g_num_rows, g_num_cols = struct.unpack('ffii', ark_read_buffer.read(16))
            utt_mat = np.zeros([g_num_rows, g_num_cols], dtype=np.float32)
            #uint16 percentile_0; uint16 percentile_25; uint16 percentile_75; uint16 percentile_100;
            per_col_header = []
            for i in range(g_num_cols):
                per_col_header.append(struct.unpack('HHHH', ark_read_buffer.read(8)))
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
            m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
            n, cols = struct.unpack('<bi', ark_read_buffer.read(5))
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=np.float32)
            utt_mat = np.reshape(tmp_mat, (rows, cols))

        ark_read_buffer.close()

        return utt_mat


class ASR_DataSet(object):
    def __init__(self, f_scp, use_online):
        self.reader = ArkReader(f_scp)
        self.use_online = use_online

    def __getitem__(self, idx):
        sample = {}
        sample['feature'] = self.reader.read_utt_data(idx)
        sample['id'] = self.reader.utt_ids[idx]

        return sample

    def __len__(self):
        return len(self.reader.utt_ids)
        # return 500

    def __iter__(self):
        '''
        '1', '2', 'eos'
        'eos'
        '1', 'eos'
        '''
        for idx in range(len(self)):
            sample = self[idx]
            piece = {'id': sample['id'], 'feature': sample['feature'], 'pos': 'eos'}
            yield piece
        # end_sample = {'id': '*finish*', 'feature': np.random.rand(100, 87), 'pos': 'eos'}
        # yield end_sample

    @staticmethod
    def random_split(array):
        rate_long = 0.7
        idx_cur = 1
        list_idx = []
        while True:
            if np.random.rand() > rate_long:
                rand_len = np.random.randint(idx_cur, idx_cur+5)
            else:
                rand_len = np.random.randint(idx_cur+5, idx_cur+100)

            idx_cur += rand_len
            if idx_cur >= len(array)-1:
                break

            list_idx.append(idx_cur)
        if not list_idx:
            list_idx = [len(array)]
        pos = list(map(str, range(len(list_idx)+1)))
        pos[-1] = 'eos'
        # for x in np.split(array, list_idx):
        #     print(x.shape)

        return zip(np.split(array, list_idx), pos)


class Data_Loader(object):
    def __init__(self,
                 feats_processing_dir,
                 tmp_dir,
                 feat_conf,
                 feat_scp=None,
                 wav_dir=None,
                 cmd2gen_scp=cmd2gen_scp,
                 use_online=False,
                 args=None,
                 size_queue=99999):
        if feat_scp:
            self.feat_scp = feat_scp
        else:
            assert feats_processing_dir and tmp_dir and wav_dir
            self.wav_dir = wav_dir if type(wav_dir) == type(Path('.')) else Path(wav_dir)
            self.tmp_dir = tmp_dir if type(tmp_dir) == type(Path('.')) else Path(tmp_dir)
            self.wav_scp = self.tmp_dir/'wav.scp'
            self.feat_scp = self.tmp_dir/'feat.scp'
            self.gen_feat_scp(cmd2gen_scp(feats_processing_dir.name, tmp_dir.name, self.wav_scp.name, self.feat_scp.name, feat_conf))
        self.dataset = ASR_DataSet(self.feat_scp, use_online)
        self.args = args
        if use_online:
            self.dict_utts = defaultdict(lambda: {'data': np.zeros([self.args.len_past_frames, self.args.size_feature],
                                                                   dtype=np.float32),
                                                  'num':0})
        else:
            # once break within the process_piece
            self.args.len_middle_frames = 9999
            self.dict_utts = defaultdict(lambda: {'data': np.zeros([0, self.args.size_feature],
                                                                   dtype=np.float32),
                                                  'num':0})
        self.queue_input = Queue(maxsize=size_queue)
        self.thread_queue_put = threading.Thread(target=self.feed_queue)
        self.thread_queue_put.daemon = True

    def gen_feat_scp(self, cmd):
        with open(str(self.wav_scp), 'w') as fw:
            for x in self.wav_dir.iterdir():
                if x.name.endswith('.wav'):
                    line = x.name + ' ' + str(x.absolute())
                    fw.write(line+'\n')
        logging.info('generate the wav.scp!')
        os.system(cmd)
        logging.info('generate the feat.scp!')

    def feed_queue(self):
        logging.info('enter the feed queue thread!')

        for i, piece in enumerate(self.dataset):
            self.process_piece(piece)

    def process_piece(self, coming_piece):
        '''
        for online decode
        '''
        dict_utts = self.dict_utts
        seq_id, seq_features, pos = coming_piece['id'], coming_piece['feature'], coming_piece['pos']
        dict_utts[seq_id]['data'] = np.concatenate([dict_utts[seq_id]['data'], seq_features], 0)
        # print(seq_features.shape, pos, 'utt len:', len(dict_utts[seq_id]))
        for idx in range(0, len(dict_utts[seq_id]['data']), self.args.len_middle_frames):
            # simulate the random arrival of input frames
            # time.sleep(random.random()/100)
            if idx + self.args.len_past_frames + self.args.len_middle_frames > len(dict_utts[seq_id]['data']):
                # there would not be middle frames next time so it is the end frames piece
                break
            elif idx + self.args.len_input_frames > len(dict_utts[seq_id]['data']):
                # there would be not complete middle frames next time
                len_pad = idx + self.args.len_input_frames - len(dict_utts[seq_id]['data'])
                pad = np.zeros([len_pad, self.args.size_feature])
                dict_utts[seq_id]['data'] = np.concatenate([dict_utts[seq_id]['data'], pad], 0)
            # print(idx, idx+len_input_frames)
            dict_utts[seq_id]['num'] += 1
            sample_piece = {'id': seq_id,
                            'feature': dict_utts[seq_id]['data'][idx:idx+self.args.len_input_frames],
                            'pos':str(dict_utts[seq_id]['num'])}
            self.queue_input.put(sample_piece)
        dict_utts[seq_id]['data'] = dict_utts[seq_id]['data'][idx:]
        if pos == 'eos':
            # pop all the rest frames in the utt and del it
            pad = np.zeros([self.args.len_post_frames, self.args.size_feature])
            feature = np.concatenate([dict_utts[seq_id]['data'], pad], 0)
            sample_piece = {'id': seq_id, 'feature': feature, 'pos':str(dict_utts[seq_id]['num']+1)+'_'+pos}
            # print('put', seq_id, 'eos')
            self.queue_input.put(sample_piece)
            del dict_utts[seq_id]


def ctc_reduce_map(align, blank):
    sent = []
    for token in align:
        if token != blank:
            sent.append(token)
    return sent


def load_graph(frozen_graph_filename):

    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            return_elements=None)

        op_input = graph.get_tensor_by_name('import/split:0')
        op_output = graph.get_tensor_by_name('import/mul_2:0')

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        config.log_device_placement = False
        sess = tf.Session(config=config, graph=graph)

    return sess, (op_input, op_output)


def maybe_finish_one_utt(uttid, dict_decoded, fw):
    # after recieving the end decoded piece and wait 10 sec, we assume the utt is finished. 10_eos
    num_utt_piece = [int(x[1].split('_')[0]) for x in dict_decoded[uttid] if 'eos' in x[1]]
    # if there is end piece and all the pieces has been in sict
    is_finished = (num_utt_piece != []) and (len(dict_decoded[uttid]) >= num_utt_piece[0])

    if is_finished:
        # post processing of finishing one utt
        dict_decoded[uttid].sort(key=lambda x: int(x[1].split('_')[0]))
        decode_line = ''.join(x[0] for x in dict_decoded[uttid])
        line = uttid + '\t' + decode_line
        print('finish one sent: ', line)
        fw.write(line+'\n')
        del dict_decoded[uttid]

    return is_finished
