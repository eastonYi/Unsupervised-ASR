#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
tf.config.gpu.set_per_process_memory_growth(True)
import numpy as np
from random import sample
import threading
from queue import Queue
from collections import Counter

from eastonCode.tfTools.tfData import TFData

from utils.arguments import args
from utils.dataset import ASR_align_DataSet
from utils.tools import build_optimizer, sampleFrames, read_ngram, batch_cer, pertubated_model_weights, ngram2kernel


def evaluation(tfdata_dev, model):
    list_loss = []
    list_acc = []

    start_time = time()
    num_processed = 0
    progress = 0
    total_cer_dist = 0
    total_cer_len = 0
    for batch in tfdata_dev:
        x, y, aligns = batch
        logits = model(x, training=False)
        loss = model.align_loss(logits, y, args.dim_output, confidence=0.9)
        acc = model.align_accuracy(logits, y)
        list_loss.append(loss)
        list_acc.append(acc)
        preds = model.get_predicts(logits)
        batch_cer_dist, batch_cer_len = batch_cer(preds.numpy(), y)
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len
        num_processed += len(x)
        progress = num_processed / args.data.dev_size

    cer = total_cer_dist/total_cer_len
    print('dev loss: {:.3f}\t dev FER: {:.3f}\t dev PER: {:.3f}\t {:.2f}min {} / {}'.format(
            np.mean(list_loss), 1-np.mean(list_acc), cer, (time()-start_time)/60, num_processed, args.data.dev_size))

    return 1-np.mean(list_acc)


def train(Model):
    # create dataset and dataloader
    with tf.device("/cpu:0"):
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_train = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_train = tfdata_train.repeat().shuffle(500).\
            padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=3)
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

    # get dataset ngram
    ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')

    # create model paremeters
    opti_adam = build_optimizer(args, type='adam')
    model = Model(args, optimizer=opti_adam, name='fc')

    queue_input = Queue(maxsize=9999)
    queue_output = Queue(maxsize=9999)

    def thread_session(thread_id, queue_input, queue_output):
        gpu = args.list_gpus[thread_id]
        with tf.device(gpu):
            opti_adam = build_optimizer(args, type='adam')
            model = Model(args, optimizer=opti_adam, name='fc'+str(thread_id))
            print('thread_{} is waiting to run on {}....'.format(thread_id, gpu))
            while True:
                # s = time()
                x, y, weights = queue_input.get()
                model.set_weights(weights)
                # t = time()
                logits = model(x, training=False)
                loss = model.align_loss(logits, y, args.dim_output, confidence=0.9)
                queue_output.put((loss, thread_id))
                # print('{} {:.3f}|{:.3f}s'.format(gpu, t-s, time()-s))

    # args.list_gpus = ['/gpu:0', '/gpu:1', '/gpu:2']
    for id in range(4):
        thread = threading.Thread(
            target=thread_session,
            args=(id, queue_input, queue_output))
        thread.daemon = True
        thread.start()

    start_time = datetime.now()
    end_time = 0

    size_population = 100
    for batch in tfdata_train:
        x, y, aligns = batch
        start = time()

        # explore
        population_time = time()
        weights_model = model.get_weights()
        for i in range(size_population):
            epsilon = [np.random.randn(*w.shape) for w in weights_model]
            weights_pert = pertubated_model_weights(weights_model, epsilon, sigma=0.5)
            queue_input.put((x, y, weights_pert))
        # queue_input.put((x, y, seed))
        gen_population_time = time()
        sign = [queue_output.get()[1] for _ in range(size_population)]
        analyse = '|'.join(map(str, Counter(sign).values()))
        evaluate_time = time()

        print('getr data time: {:.2f}, gen population time: {:.2f}s, evaluate time: {:.2f}s, {}'.\
              format(start-end_time, gen_population_time-population_time, evaluate_time-gen_population_time, analyse))
        end_time = time()

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)

    if args.model.structure == 'fc':
        from utils.model import FC_Model as Model
    elif args.model.structure == 'lstm':
        from utils.model import LSTM_Model as Model

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print('enter the TRAINING phrase')
        train(Model)
        # train_mul(Model)

        # python main_es.py --gpu 1 -c configs/timit_es.yaml
