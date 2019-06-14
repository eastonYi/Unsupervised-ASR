#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
tf.config.gpu.set_per_process_memory_growth(True)
import numpy as np
from tqdm import tqdm
from random import sample
from nltk import FreqDist
import threading
from queue import Queue

from eastonCode.tfTools.tfData import TFData

from utils.arguments import args
from utils.dataset import ASR_align_DataSet
from utils.tools import sampleFrames, read_ngram, batch_cer, get_model_weights, ngram2kernel


def train(Model):
    # create dataset and dataloader
    with tf.device("/cpu:0"):
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_monitor = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_monitor = tfdata_monitor.cache().repeat().shuffle(500).padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=5)
        tfdata_monitor = iter(tfdata_monitor)
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

    # get dataset ngram
    ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')

    # create model paremeters
    optimizer = tf.keras.optimizers.SGD(args.opti.lr_fs)
    model = Model(args, optimizer=optimizer, name='fc')
    model.summary()

    def get_rewards(list_models):
        list_pz = [[] for _ in range(len(list_models))]
        list_K = [[] for _ in range(len(list_models))]

        num_processed = 0
        for _ in range(args.batch_multi):
            batch = next(tfdata_monitor)
            x, y, aligns = batch
            num_processed += len(x)
            aligns_sampled = sampleFrames(aligns)
            ngram_sampled = sample(ngram_py, args.data.top_k)
            kernel, py = ngram2kernel(ngram_sampled, args)
            for i, weights in enumerate(list_models):
                model.set_weights(weights)
                logits = model(x, training=False)
                pz, K = model.EODM(logits, aligns_sampled, kernel)
                list_pz[i].append(pz)
                list_K[i].append(K)

        pz = tf.reduce_sum(list_pz, 1) / tf.reduce_sum(list_K, 1)
        rewards = tf.reduce_sum(py * tf.math.log(1e-11+pz), -1) # ngram loss

        return rewards

    # save & reload
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    if args.dirs.restore:
        latest_checkpoint = tf.train.CheckpointManager(ckpt, args.dirs.restore, max_to_keep=1).latest_checkpoint
        ckpt.restore(latest_checkpoint)
        print('{} restored!!'.format(latest_checkpoint))

    best_rewards = -999
    start_time = datetime.now()
    lr = args.opti.lr
    sigma = args.opti.sigma
    size_population = args.opti.population

    for global_step in range(99999):
        run_model_time = time()
        # build compute model on-the-fly
        population = []
        weights_model = model.get_weights()

        # explore
        weights_population = []
        for i in range(size_population):
            x = []
            for w in weights_model:
                x.append(np.random.randn(*w.shape))
            population.append(x)
            weights_population.append(get_model_weights(weights_model, population[i], sigma))

        # analyse feedbacks
        rewards = get_rewards(weights_population)
        _rewards = (rewards - np.mean(rewards)) / (np.std(rewards)+1e-11)

        # merge weights
        weights_new = []
        for index, w in enumerate(weights_model):
            A = tf.cast(tf.stack([p[index] for p in population], 0), 'float32')
            shape_out = A.shape[1:]
            _rewards = tf.reshape(_rewards, [1, size_population])
            A = tf.reshape(A, [size_population, -1])
            weights_new.append(w + lr/(size_population*sigma) * tf.reshape(tf.matmul(_rewards, A), shape_out))
        model.set_weights(weights_new)
        FER = wipe_off(next(tfdata_monitor), model, optimizer)

        used_time = time()-run_model_time
        if global_step % 1 == 0:
            print('full training loss: {:.3f}, FER: {:.3f} spend {:.2f}s step {}'.format(np.mean(-rewards), FER, used_time, global_step))

        if global_step % args.dev_step == 0:
            evaluation(tfdata_dev, model)
        if global_step % args.fs_step == 0:
            fs_constrain(next(tfdata_monitor), model, optimizer)
            # sigma *= 0.9
            # lr *= 0.9
        if global_step % args.decode_step == 0:
            decode(model)
        if global_step % args.save_step == 0:
            ckpt_manager.save()

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))

aligns_sampled = kernel = None

def train_mul(Model):

    lr = args.opti.lr
    sigma = args.opti.sigma
    size_population = args.opti.population

    # create dataset and dataloader
    with tf.device("/cpu:0"):
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_monitor = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_monitor = tfdata_monitor.cache().repeat().shuffle(500).padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=5)
        tfdata_monitor = iter(tfdata_monitor)
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

    # get dataset ngram
    ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')

    # create model paremeters
    optimizer = tf.keras.optimizers.SGD(args.opti.lr_fs)
    model = Model(args, optimizer=optimizer, name='fc')
    model.summary()

    queue_input = Queue(maxsize=9999)
    queue_output = Queue(maxsize=9999)

    def thread_session(thread_id, queue_input, queue_output):
        global aligns_sampled, kernel
        model = Model(args, optimizer=optimizer, name='fc'+str(thread_id))
        print('thread_{} is waiting to run....'.format(thread_id))
        while True:
            x, weights = queue_input.get()
            model.set_weights(weights)
            logits = model(x, training=False)
            pz, K = model.EODM(logits, aligns_sampled, kernel)
            queue_output.put((pz, K))

    for id in range(args.num_threads):
        thread = threading.Thread(
            target=thread_session,
            args=(id, queue_input, queue_output))
        thread.daemon = True
        thread.start()

    def get_rewards(list_models):
        global aligns_sampled, kernel
        list_pz = [[] for _ in range(len(list_models))]
        list_K = [[] for _ in range(len(list_models))]

        num_processed = 0
        for _ in range(args.batch_multi):
            batch = next(tfdata_monitor)
            x, y, aligns = batch
            num_processed += len(x)
            aligns_sampled = sampleFrames(aligns)
            ngram_sampled = sample(ngram_py, args.data.top_k)
            kernel, py = ngram2kernel(ngram_sampled, args)

            [queue_input.put((x, list_models[i])) for i in range(size_population)]

            for i in range(size_population):
                pz, K = queue_output.get()
                list_pz[i].append(pz)
                list_K[i].append(K)

        pz = tf.reduce_sum(list_pz, 1) / tf.reduce_sum(list_K, 1)
        rewards = tf.reduce_sum(py * tf.math.log(1e-11+pz), -1) # ngram loss

        return rewards

    # save & reload
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    if args.dirs.restore:
        latest_checkpoint = tf.train.CheckpointManager(ckpt, args.dirs.restore, max_to_keep=1).latest_checkpoint
        ckpt.restore(latest_checkpoint)
        print('{} restored!!'.format(latest_checkpoint))

    best_rewards = -999
    start_time = datetime.now()

    for global_step in range(99999):
        start = time()
        # build compute model on-the-fly
        population = []
        weights_model = model.get_weights()

        # explore
        weights_population = []
        for i in range(size_population):
            x = []
            for w in weights_model:
                x.append(np.random.randn(*w.shape))
            population.append(x)
            weights_population.append(get_model_weights(weights_model, population[i], sigma))

        # analyse feedbacks
        model_time = time()
        rewards = get_rewards(weights_population)
        run_model_time = time() - model_time
        _rewards = (rewards - np.mean(rewards)) / (np.std(rewards)+1e-11)

        # merge weights
        weights_new = []
        for index, w in enumerate(weights_model):
            A = tf.cast(tf.stack([p[index] for p in population], 0), 'float32')
            shape_out = A.shape[1:]
            _rewards = tf.reshape(_rewards, [1, size_population])
            A = tf.reshape(A, [size_population, -1])
            weights_new.append(w + lr/(size_population*sigma) * tf.reshape(tf.matmul(_rewards, A), shape_out))
        model.set_weights(weights_new)

        used_time = time()-start
        print('full training loss: {:.3f}, spend {:.2f}|{:.2f}s \tstep {}'.format(np.mean(-rewards), run_model_time, used_time, global_step))

        if global_step % args.dev_step == 0:
            evaluation(tfdata_dev, model)
        if global_step % args.fs_step == 0:
            fs_constrain(tfdata_monitor, model, optimizer)
            # sigma *= 0.9
            # lr *= 0.9
        if global_step % args.decode_step == 0:
            decode(model)
        if global_step % args.save_step == 0:
            ckpt_manager.save()

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


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
        loss = model.align_loss(logits, y, aligns, full_align=args.data.full_align)
        acc = model.align_accuracy(logits, y, aligns, full_align=args.data.full_align)
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


def decode(model):
    dataset = ASR_align_DataSet(
        file=[args.dirs.dev.data],
        args=args,
        _shuffle=False,
        transform=True)
    sample = dataset[0]
    x = np.array([sample['feature']], dtype=np.float32)
    logits = model(x, training=False)
    predits = model.get_predicts(logits)
    print('predict: \n', predits.numpy()[0])
    print('label: \n', sample['label'])
    print('align: ', sample['align'])


def fs_constrain(batch, model, optimizer):
    x, y, aligns = batch
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = model.frames_constrain_loss(logits, aligns)
    print('fs loss: ', loss.numpy())
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def wipe_off(batch, model, optimizer):
    x, y, aligns = batch
    # build compute model on-the-fly
    logits = model(x, training=False)
    acc = model.align_accuracy(logits, y, aligns, full_align=args.data.full_align)
    if 1-acc > 0.85:
        y = tf.ones_like(y) * y[0][0]
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = model.align_loss(logits, y, aligns, full_align=args.data.full_align)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        decode(model)

    return 1-acc

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
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        print('enter the TRAINING phrase')
        train(Model)
        # train_mul(Model)

        # python main_es.py --gpu 1 -c configs/timit_es.yaml
