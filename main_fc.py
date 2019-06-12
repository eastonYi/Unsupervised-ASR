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

from eastonCode.tfTools.tfData import TFData

from utils.arguments import args
from utils.dataset import ASR_align_DataSet
from utils.tools import build_optimizer, warmup_exponential_decay, size_variables, \
sampleFrames, read_ngram, batch_cer, get_model_weights, ngram2kernel, get_preds_ngram


def train(Model):
    # create dataset and dataloader
    dataset_train = ASR_align_DataSet(
        file=[args.dirs.train.data],
        args=args,
        _shuffle=True,
        transform=True)

    with tf.device("/cpu:0"):
        tfdata_train = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=True)
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_monitor = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=False)
    tfdata_train = tfdata_train.cache().repeat().shuffle(500).padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=5)
    tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))
    tfdata_monitor = tfdata_monitor.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

    # get dataset ngram
    # ngram_py = dataset_train.get_dataset_ngram(n=args.data.ngram, k=1000)
    ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')

    # build optimizer
    warmup = warmup_exponential_decay(
        warmup_steps=args.opti.warmup_steps,
        peak=args.opti.peak,
        decay_steps=args.opti.decay_steps)
    optimizer = build_optimizer(warmup, args.opti.type)

    # create model paremeters
    model = Model(args, optimizer=optimizer, name='fc')
    model.summary()

    # save & reload
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    if args.dirs.restore:
        latest_checkpoint = tf.train.CheckpointManager(ckpt, args.dirs.restore, max_to_keep=1).latest_checkpoint
        ckpt.restore(latest_checkpoint)
        print('{} restored!!'.format(latest_checkpoint))

    start_time = datetime.now()
    get_data_time = 0
    num_processed = 0
    progress = 0

    for global_step, batch in enumerate(tfdata_train):
        x, y, aligns = batch
        aligns_sampled = sampleFrames(aligns)
        ngram_sampled = sample(ngram_py, args.data.top_k)
        kernel = np.zeros([args.data.ngram, args.dim_output, args.data.top_k], dtype=np.float32)
        list_py = []
        for i, (z, py) in enumerate(ngram_sampled):
            list_py.append(py)
            for j, token in enumerate(z):
                kernel[j][token][i] = 1.0
        py = np.array(list_py, dtype=np.float32)
        run_model_time = time()
        # build compute model on-the-fly
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            # loss = model.align_loss(logits, y[:, :max_len], aligns, full_align=args.data.full_align)
            loss_fs = model.frames_constrain_loss(logits, aligns)
            # loss_fs = 0
            loss_EODM = model.EODM_loss(logits, aligns_sampled, kernel, py)
            loss = loss_EODM + args.lambda_fs * loss_fs

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc = model.align_accuracy(logits, y, aligns, full_align=args.data.full_align)

        num_processed += len(x)
        get_data_time = run_model_time - get_data_time
        run_model_time = time() - run_model_time

        progress = num_processed / args.data.train_size
        if global_step % 1 == 0:
            print('loss: {:.3f}|{:.3f}\t FER: {:.3f}\t batch: {} lr:{:.6f} time: {:.2f}|{:.2f} s {:.3f}% step: {}'.format(
                   loss_EODM, loss_fs, 1-acc, x.shape, warmup(global_step*1.0).numpy(), get_data_time, run_model_time, progress*100.0, global_step))
        get_data_time = time()

        if global_step % args.dev_step == 0:
            evaluation(tfdata_dev, model)
            # monitor_EODM_loss(tfdata_monitor, model, ngram_py)
        # if global_step % args.decode_step == 0:
        #     decode(model)
        # if global_step % args.save_step == 0:
        #     ckpt_manager.save()

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def train_ES(Model):
    from nltk import FreqDist
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
        transformation_func = tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda x,*y: tf.shape(x)[0],
            bucket_boundaries=args.list_bucket_boundaries,
            bucket_batch_sizes=args.list_batch_size,
            padded_shapes=([None, args.dim_input], [None], [None]))
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))
        tfdata_monitor = tfdata_monitor.cache().apply(transformation_func).prefetch(buffer_size=10)

    # get dataset ngram
    ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')

    # create model paremeters
    model = Model(args, optimizer=tf.keras.optimizers.Adam(), name='fc')
    model.summary()

    def get_rewards(list_models):
        list_pz = [[] for _ in range(len(list_models))]
        list_K = [[] for _ in range(len(list_models))]

        num_processed = 0
        for batch in tfdata_monitor:
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
                # print('\t\tan individual use time {:.2f}|{:.2f}s'.format(time()-s1, time()-s))
            # print('\tprocessed {:.3f}%'.format(100*num_processed/args.data.train_size))

        pz = tf.reduce_sum(list_pz, 1) / tf.reduce_sum(list_K, 1)
        rewards = tf.reduce_sum(py * tf.math.log(1e-9+pz), -1) # ngram loss

        return rewards

    def get_discrete_rewards(list_models):
        ngrams_preds = [[] for _ in range(len(list_models))]

        num_processed = 0
        for batch in tfdata_monitor:
            x, y, aligns = batch
            len_aligns = tf.reduce_sum(tf.cast(aligns>0, tf.int32), -1)
            num_processed += len(x)
            aligns_sampled = sampleFrames(aligns)
            for i, weights in enumerate(list_models):
                s = time()
                model.set_weights(weights)
                logits = model(x, training=False)
                s1 = time()
                pred_aligns = model.get_predicts(logits)
                # indices
                batch, len_label_max = aligns.shape
                batch_idx = tf.tile(tf.range(batch)[:, None], [1, len_label_max])
                indices = tf.stack([batch_idx, aligns_sampled], -1)
                preds = tf.gather_nd(pred_aligns, indices)
                _ngrams_preds = get_preds_ngram(preds, len_aligns, args.data.ngram)
                ngrams_preds[i].append(_ngrams_preds)
                print('\t\tan individual use time {:.2f}|{:.2f}s'.format(time()-s1, time()-s))
            print('\tprocessed {:.3f}%'.format(100*num_processed/args.data.train_size))

        rewards = np.zeros(len(list_models), dtype=np.float32)
        for i, list_ngrams in enumerate(ngrams_preds):
            one = FreqDist()
            for ngrams in list_ngrams:
                one += ngrams
            for key, value in ngram_py:
                r = value * tf.math.log(1e-9+one[key]/total_num)
                rewards[i] = r

        return rewards

    # save & reload
    ckpt = tf.train.Checkpoint(model=model, optimizer=tf.keras.optimizers.Adam())
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    if args.dirs.restore:
        latest_checkpoint = tf.train.CheckpointManager(ckpt, args.dirs.restore, max_to_keep=1).latest_checkpoint
        ckpt.restore(latest_checkpoint)
        print('{} restored!!'.format(latest_checkpoint))

    start_time = datetime.now()

    for global_step in range(99999):
        run_model_time = time()
        # build compute model on-the-fly
        population = []
        weights_model = model.get_weights()

        # explore
        weights_population = []
        for i in range(args.opti.population):
            x = []
            for w in weights_model:
                x.append(np.random.randn(*w.shape))
            population.append(x)
            weights_population.append(get_model_weights(weights_model, population[i], args.opti.sigma))

        # analyse feedbacks
        rewards = get_rewards(weights_population)
        print('full training loss: ', np.mean(-rewards))
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards)+1e-9)

        # merge weights
        weights_new = []
        for index, w in enumerate(weights_model):
            A = tf.cast(tf.stack([p[index] for p in population], 0), 'float32')
            shape_out = A.shape[1:]
            rewards = tf.reshape(rewards, [1, args.opti.population])
            A = tf.reshape(A, [args.opti.population, -1])
            weights_new.append(w + args.opti.lr/(args.opti.population*args.opti.sigma) * tf.reshape(tf.matmul(rewards, A), shape_out))
        model.set_weights(weights_new)

        run_model_time = time() - run_model_time

        if global_step % args.dev_step == 0:
            evaluation(tfdata_dev, model)
        # if global_step % args.decode_step == 0:
        #     decode(model)
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
        _shuffle=True,
        transform=True)
    sample = dataset[0]
    x = np.array([sample['feature']], dtype=np.float32)
    logits = model(x, training=False)
    predits = model.get_predicts(logits)
    print('predits: ', predits)
    print('label: ', sample['label'])
    print('align: ', sample['align'])


def monitor_EODM_loss(tfdata_train, model, ngram_py):
    list_pz = []
    list_K = []
    list_mini_EODM = []

    start_time = time()
    num_processed = 0
    for batch in tfdata_train:
        x, y, aligns = batch

        aligns_sampled = sampleFrames(aligns)
        ngram_sampled = sample(ngram_py, args.data.top_k)
        kernel = np.zeros([args.data.ngram, args.dim_output, args.data.top_k], dtype=np.float32)
        list_py = []
        for i, (z, py) in enumerate(ngram_sampled):
            list_py.append(py)
            for j, token in enumerate(z):
                kernel[j][token][i] = 1.0
        py = np.array(list_py, dtype=np.float32)

        logits = model(x, training=False)
        pz, K = model.EODM(logits, aligns_sampled, kernel)
        list_pz.append(pz)
        list_K.append(K)
        list_mini_EODM.append(model.EODM_loss(logits, aligns_sampled, kernel, py).numpy())

        num_processed += len(x)

    pz = tf.reduce_sum(list_pz, 0) / tf.reduce_sum(list_K, 0)
    loss_EODM = tf.reduce_sum(- py * tf.math.log(pz)) # ngram loss
    print('Full-batch EODM loss: {:.3f}\t mini-batch EODM loss: {:.3f}\t {:.2f}min {} / {}'.format(
           loss_EODM, np.mean(list_mini_EODM), (time()-start_time)/60, num_processed, args.data.train_size))

    return loss_EODM


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
        # train_ES(Model)

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
