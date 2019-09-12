#!/usr/bin/env python
# coding=utf-8
"""
author: easton
date: 19-01-22
performance:
    tcnt offline wfst rmsil: 5.24  |  python3 infer.py --model python_test/export_1203/frozen_model.pb --output decoded_offline_wfst_rmsil.txt --feat_scp all_feats.scp.rmsil  --mode offline --use_wfst
    tcnt offline rmsil: 7.71 | python3 infer.py --model python_test/export_1203/frozen_model.pb --output decoded_offline_rmsil.txt --feat_scp all_feats.scp.rmsil  --mode offline
    tcnt offline :  9.67 | python3 infer.py --model python_test/export_1203/frozen_model.pb --output decoded_offline_rmsil.txt --feat_scp tcnt_org/final_feats.scp  --mode offline
description:
    - support greedy or wfst decode
    - support decode directly form a feat scp file
limitations:
    - the wfst would output all the decoded words each calling time. Need to clip
    the repeat part each time. sor the wfst decode need to fix.
    --use_gpu is much slow than not use gpu
"""

import time
import logging
import numpy as np
from queue import Queue
import threading
from collections import defaultdict
from pathlib import Path
import sys

from infer_tools import WFST_Decoder, load_vocab, Data_Loader, ctc_reduce_map, load_graph, maybe_finish_one_utt, AttrDict

np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=np.nan)
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')
# put all the tmp file into it, including the feats scp
tmp_dir = Path('tmp')
# the feats processing related files are put here
feats_processing_dir = Path('pipeline_for_ztspeech')
len_past_frames = 64 # the past frames that need to concatenate in front of the comming frames
len_middle_frames = 128 # the minimal frame length that could be put into the network
len_post_frames = 64 # the past frames that need to concatenate in front of the comming frames
len_input_frames = len_past_frames+len_middle_frames+len_post_frames
rate_stack = 8
size_feature = 29*3
# num_threads = 10
args = {'size_feature': 29*3,
        'len_past_frames': 64,
        'len_middle_frames': 128,
        'len_post_frames': 64,
        'len_input_frames': len_input_frames}
args = AttrDict(args)

def check():
    assert feats_processing_dir.is_dir()

    tuple_files_in_feats = ('compute-fbank-feats',
                            'add-deltas',
                            'copy-feats',
                            'apply-cmvn',
                            'cmvn_glob.mat')
    for file in tuple_files_in_feats:
        try:
            assert (feats_processing_dir/file).is_file()
        except:
            raise IOError(feats_processing_dir/file)
    # assert all( (python_dir/file).is_file() for file in tuple_files_in_python )

    if tmp_dir.is_dir():
        pass
        # os.system('rm -r '+tmp_dir.name)
    else:
        tmp_dir.mkdir()


def thread_session(sess, thread_id, queue_input, queue_output, list_op, mode):
    # input_feature = graph.get_tensor_by_name('import/split:0')
    # distribution_T = graph.get_tensor_by_name('import/mul_2:0')
    input_feature, distribution_T = list_op

    logging.info('thread_{} is waiting to run....'.format(thread_id))
    while True:
        sample = queue_input.get()
        feature = sample['feature']
        if np.any(np.isinf(sample['feature'])):
            logging.warning('There are inf numbers in feat in {} !'.format(sample['id']))
            feature = feature[:, 1-np.any(np.isinf(feature), 0)>0]

        len_fea = len(feature)
        dict_feed = {input_feature: feature.reshape([1, len_fea, int(size_feature / 3), 3])}
        # start = time.time()
        distribution = sess.run(distribution_T, feed_dict=dict_feed)
        # print('use: {:.2f}'.format(time.time()-start))
        distribution = distribution[0]
        distribution = distribution[np.sum(distribution, -1)>0.99]
        output_model = {'distribution': distribution, 'id': sample['id'], 'pos': sample['pos']}
        queue_output.put(output_model)


def infer(frozen_graph_filename, wav_dir, output_file, use_wfst, feat_conf, feat_scp, num_threads):
    # load data and put pieces into input queue
    data_loader = Data_Loader(
        feats_processing_dir=feats_processing_dir,
        tmp_dir=tmp_dir,
        feat_conf=feat_conf,
        feat_scp=feat_scp,
        wav_dir=wav_dir,
        args=args
        )
    data_loader.thread_queue_put.start()

    # prepare to create sessions
    queue_output = Queue(maxsize=100)

    sess, list_op = load_graph(frozen_graph_filename)

    # run sessions and comsume the input queue
    for id in range(num_threads):
        thread = threading.Thread(
            target=thread_session,
            args=(sess, id, data_loader.queue_input, queue_output, list_op))
        thread.daemon = True
        thread.start()

    # decode preparation
    if use_wfst:
        # WFTS config
        len_decode_max=200
        decode_outs=np.zeros((len_decode_max), dtype=np.int32)
        wfst = WFST_Decoder(
            len_decode_max=len_decode_max,
            decode_outs=decode_outs,
            fcdll="bin/libctc_wfst_lib.so",
            fcfg="cfg.json")
        vocab_file = 'words.txt'
    else:
        # greedy mode
        vocab_file = 'vocab.txt'

    token2id, idx2token = load_vocab(vocab_file)

    # consume the output queue and decode
    num_recognized = 0
    dict_decoded = defaultdict(lambda: [])
    with open(output_file, 'w') as fw:
        while True:
            try:
                output_model = queue_output.get(block=True, timeout=200)
            except:
                for x in dict_decoded.items():
                    print('rest dict_decoded')
                    print(x)
                    # print('uttid:', uttid, 'decoded:', decoded, 'pos:', output_model['pos'])
                print('finished! recognized {}'.format(num_recognized))
                sys.exit()

            distribution, uttid, pos = output_model['distribution'], output_model['id'], output_model['pos']
            distribution_log = np.log(distribution)
            if use_wfst:
                # the pieces of one sentence are not coming in order, but they must be decoded in order
                decoded, flag = wfst.decode(distribution_log, uttid, pos)
                if len(decoded) > len(dict_decoded[uttid]):
                    dict_decoded[uttid] = decoded

                if flag == 2:
                    decode_line = ''.join(idx2token[idx] for idx in decoded)
                    line = uttid + '\t' + decode_line
                    used_time = time.time() - wfst.dict_utt[uttid]['birth_time']
                    logging.info('finish one sent in {:.3f}s: {}'.format(used_time, line))
                    fw.write(line+'\n')
                    del dict_decoded[uttid]
                    num_recognized +=1
            else:
                decoded = np.argmax(distribution_log, -1)
                decoded = ctc_reduce_map(decoded, token2id['<blk>'])
                decoded = ''.join(idx2token[idx] for idx in decoded)
                dict_decoded[uttid].append((decoded, pos))
                num_recognized += maybe_finish_one_utt(uttid, dict_decoded, fw)

            # print(decoded, uttid+'_'+pos)


if __name__ == '__main__':
    '''
    python3 infer.py --model python_test/export_1203/frozen_model.pb --output decoded_offline.txt --wav_dir --wav_dir wav_test --use_wfst False
    '''
    import os
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--num_threads', type=int, dest='num_threads', default=10)
    parser.add_argument('--model', type=str, dest='model', default=None)
    parser.add_argument('--output', type=str, dest='output', default=None)
    parser.add_argument('--wav_dir', type=str, dest='wav_dir', default=None)
    parser.add_argument('--feat_scp', type=str, dest='feat_scp', default=None)
    parser.add_argument('--use_wfst', action='store_true', default=False, help='''False: greedy, True: wfst, default greedy''')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='''False: not use gpu, True: wfst, use gpu0''')
    parser.add_argument('--feat_conf', type=str, dest='feat_conf', default='fbank_16k.conf')

    param = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0' if param.use_gpu else ''

    check()
    infer(param.model, param.wav_dir, param.output, param.mode, param.use_wfst, param.feat_conf, param.feat_scp, param.num_threads)
