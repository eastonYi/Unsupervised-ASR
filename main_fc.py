#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
tf.config.gpu.set_per_process_memory_growth(True)
import numpy as np
from random import sample

from eastonCode.tfTools.tfData import TFData

from utils.arguments import args
from utils.dataset import ASR_align_DataSet, LMDataSet
from utils.tools import build_optimizer, warmup_exponential_decay, sampleFrames,\
read_ngram, batch_cer, ngram2kernel


# def train(Model):
#     # create dataset and dataloader
#     dataset_train = ASR_align_DataSet(
#         file=[args.dirs.train.data],
#         args=args,
#         _shuffle=True,
#         transform=True)
#
#     with tf.device("/cpu:0"):
#         tfdata_train = TFData(dataset=None,
#                         dataAttr=['feature', 'label', 'align'],
#                         dir_save=args.dirs.train.tfdata,
#                         args=args).read(_shuffle=True)
#         tfdata_dev = TFData(dataset=None,
#                         dataAttr=['feature', 'label', 'align'],
#                         dir_save=args.dirs.dev.tfdata,
#                         args=args).read(_shuffle=False)
#         # tfdata_monitor = TFData(dataset=None,
#         #                 dataAttr=['feature', 'label', 'align'],
#         #                 dir_save=args.dirs.train.tfdata,
#         #                 args=args).read(_shuffle=False)
#     tfdata_train = tfdata_train.cache().repeat().shuffle(500).padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=5)
#     tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))
#     # tfdata_monitor = tfdata_monitor.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))
#
#     # get dataset ngram
#     ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')
#
#     # build optimizer
#     warmup = warmup_exponential_decay(
#         warmup_steps=args.opti.warmup_steps,
#         peak=args.opti.peak,
#         decay_steps=args.opti.decay_steps)
#     optimizer = build_optimizer(warmup, args.opti.type)
#
#     # create model paremeters
#     model = Model(args, optimizer=optimizer, name='fc')
#     model.summary()
#
#     # save & reload
#     ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
#     ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
#     if args.dirs.restore:
#         latest_checkpoint = tf.train.CheckpointManager(ckpt, args.dirs.restore, max_to_keep=1).latest_checkpoint
#         ckpt.restore(latest_checkpoint)
#         print('{} restored!!'.format(latest_checkpoint))
#
#     start_time = datetime.now()
#     get_data_time = 0
#     num_processed = 0
#     progress = 0
#
#     for global_step, batch in enumerate(tfdata_train):
#         x, y, aligns = batch
#         aligns_sampled = sampleFrames(aligns)
#         ngram_sampled = sample(ngram_py, args.data.top_k)
#         kernel, py = ngram2kernel(ngram_sampled, args)
#         run_model_time = time()
#         # build compute model on-the-fly
#         with tf.GradientTape() as tape:
#             logits = model(x, training=True)
#             # loss = model.align_loss(logits, y, aligns, full_align=args.data.full_align)
#             # loss_fs = model.frames_constrain_loss(logits, aligns)
#             loss = model.EODM_loss(logits, aligns_sampled, kernel, py)
#
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         acc = model.align_accuracy(logits, y, aligns, full_align=args.data.full_align)
#
#         num_processed += len(x)
#         get_data_time = run_model_time - get_data_time
#         run_model_time = time() - run_model_time
#
#         progress = num_processed / args.data.train_size
#         if global_step % 10 == 0:
#             print('loss: {:.2f}\t FER: {:.3f}\t batch: {} lr:{:.6f} time: {:.2f}|{:.2f} s {:.3f}% step: {}'.format(
#                    loss, 1-acc, x.shape, warmup(global_step*1.0).numpy(), get_data_time, run_model_time, progress*100.0, global_step))
#         get_data_time = time()
#
#         if global_step % args.dev_step == 0:
#             evaluation(tfdata_dev, model)
#         if global_step % args.decode_step == 0:
#             decode(model)
#         if global_step % args.fs_step == 0:
#             fs_constrain(batch, model, optimizer)
#         if 1-acc > 0.80:
#             head_tail_constrain(batch, model, optimizer)
#         if global_step % args.save_step == 0:
#             ckpt_manager.save()
#
#     print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))
def train(Model):
    # load external LM
    with tf.device("/cpu:0"):
        tfdata_train = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=True)
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)

        tfdata_train = tfdata_train.repeat().shuffle(1000).\
            padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=100)
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

    # get dataset ngram
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
        kernel, py = ngram2kernel(ngram_sampled, args)
        run_model_time = time()
        # build compute model on-the-fly
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.variables)
            logits = model(x, training=True)
            loss_EODM = model.EODM_loss(logits, aligns_sampled, kernel, py)
            loss = loss_EODM
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc = model.align_accuracy(logits, y, aligns, full_align=args.data.full_align)

        num_processed += len(x)
        get_data_time = run_model_time - get_data_time
        run_model_time = time() - run_model_time

        progress = num_processed / args.data.train_size
        if global_step % 10 == 0:
            print('EODM loss: {:.2f}\t FER: {:.3f}\t batch: {} lr:{:.6f} time: {:.2f}|{:.2f} s {:.3f}% step: {}'.format(
                   loss_EODM, 1-acc, x.shape, warmup(global_step*1.0).numpy(), get_data_time, run_model_time, progress*100.0, global_step))
        get_data_time = time()

        if global_step % args.dev_step == 0:
            evaluation(tfdata_dev, model)
        if global_step % args.decode_step == 0:
            decode(model)
        if global_step % args.fs_step == 0:
            fs_constrain(batch, model, optimizer)
        if 1-acc > 0.83:
            head_tail_constrain(batch, model, optimizer)
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
    print('predits: \n', predits.numpy()[0])
    print('label: \n', sample['label'])
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
        kernel, py = ngram2kernel(ngram_sampled, args)
        logits = model(x, training=False)
        pz, K = model.EODM(logits, aligns_sampled, kernel)
        list_pz.append(pz)
        list_K.append(K)
        list_mini_EODM.append(model.EODM_loss(logits, aligns_sampled, kernel, py).numpy())

        num_processed += len(x)

    pz = tf.reduce_sum(list_pz, 0) / tf.reduce_sum(list_K, 0)
    loss_EODM = tf.reduce_sum(- py * tf.math.log(pz+1e-17)) # ngram loss
    print('Full-batch EODM loss: {:.3f}\t mini-batch EODM loss: {:.3f}\t {:.2f}min {} / {}'.format(
           loss_EODM, np.mean(list_mini_EODM), (time()-start_time)/60, num_processed, args.data.train_size))

    return loss_EODM


def fs_constrain(batch, model, optimizer):
    x, y, aligns = batch
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = model.frames_constrain_loss(logits, aligns)
        print('fs loss: ', loss.numpy())
        loss *= args.lambda_fs

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def head_tail_constrain(batch, model, optimizer):
    x, y, aligns = batch
    # build compute model on-the-fly
    y = tf.ones_like(y) * y[0][0]
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = model.align_loss(logits, y, aligns, full_align=args.data.full_align)
        loss *= 50

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def lm_assistant(Model, Model_LM):
    # load external LM
    with tf.device("/cpu:0"):
        tfdata_train = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=True)
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)

        # transformation_func = tf.data.experimental.bucket_by_sequence_length(
        #     element_length_func=lambda x,*y: tf.shape(x)[0],
        #     bucket_boundaries=args.list_bucket_boundaries,
        #     bucket_batch_sizes=args.list_batch_size,
        #     padded_shapes=([None, args.dim_input], [None], [None]))

        tfdata_train = tfdata_train.repeat().shuffle(100).\
            padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=1000)
        # tfdata_train = tfdata_train.repeat().shuffle(500).apply(transformation_func).prefetch(buffer_size=5)
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

    # # get dataset ngram
    # ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')

    # build optimizer
    warmup = warmup_exponential_decay(
        warmup_steps=args.opti.warmup_steps,
        peak=args.opti.peak,
        decay_steps=args.opti.decay_steps)
    optimizer = build_optimizer(warmup, args.opti.type)

    # create model paremeters
    model_lm = Model_LM(args.args_lm, optimizer=tf.keras.optimizers.Adam(), name='lstm')
    model = Model(args, optimizer=optimizer, name='fc')
    model.summary()
    model_lm.summary()

    # save & reload
    ckpt_lm = tf.train.Checkpoint(model=model_lm, optimizer=model_lm.optimizer)
    latest_checkpoint = tf.train.CheckpointManager(ckpt_lm, args.args_lm.dirs.restore, max_to_keep=1).latest_checkpoint
    assert latest_checkpoint
    ckpt_lm.restore(latest_checkpoint)
    print('LM {} restored!!'.format(latest_checkpoint))
    lm_dev(model_lm)

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
        # ngram_sampled = sample(ngram_py, args.data.top_k)
        # kernel, py = ngram2kernel(ngram_sampled, args)
        run_model_time = time()
        # build compute model on-the-fly
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(model.variables)
            logits = model(x, training=True)
            loss = model.align_loss(logits, y, aligns, full_align=args.data.full_align)
            # loss_EODM = model.EODM_loss(logits, aligns_sampled, kernel, py)
            loss_LM = model_lm.compute_fitting_loss(logits, aligns_sampled)
            # loss_LM = 0
            loss_EODM = loss
            # loss = loss_EODM + loss_LM
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        acc = model.align_accuracy(logits, y, aligns, full_align=args.data.full_align)

        num_processed += len(x)
        get_data_time = run_model_time - get_data_time
        run_model_time = time() - run_model_time

        progress = num_processed / args.data.train_size
        if global_step % 10 == 0:
            print('EODM loss: {:.2f}\tlm loss: {:.2f}\t FER: {:.3f}\t batch: {} lr:{:.6f} time: {:.2f}|{:.2f} s {:.3f}% step: {}'.format(
                   loss_EODM, loss_LM, 1-acc, x.shape, warmup(global_step*1.0).numpy(), get_data_time, run_model_time, progress*100.0, global_step))
        get_data_time = time()

        if global_step % args.dev_step == 0:
            evaluation(tfdata_dev, model)
        if global_step % args.decode_step == 0:
            decode(model)
        if global_step % args.fs_step == 0:
            fs_constrain(batch, model, optimizer)
        # if 1-acc > 0.80:
        #     head_tail_constrain(batch, model, optimizer)
        if global_step % args.save_step == 0:
            ckpt_manager.save()

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def lm_dev(model):
    # evaluate
    dataset_dev = LMDataSet(
        list_files=[args.dirs.dev.data],
        args=args,
        _shuffle=False)
    tfdata_lm = tf.data.Dataset.from_generator(
        dataset_dev,
        (tf.int32, tf.int32),
        (tf.TensorShape([None]), tf.TensorShape([None]))).\
        padded_batch(args.batch_size, ([None], [None]))
    start_time = time()
    num_processed = 0
    loss_sum = 0
    num_tokens = 0
    for batch in tfdata_lm:
        x, y = batch
        logits = model(x, training=False)
        loss, num_batch_tokens = model.compute_ppl(logits, y)
        loss_sum += loss
        num_tokens += num_batch_tokens
        num_processed += len(x)
    ppl = tf.exp(loss_sum/num_tokens)
    print('lm dev ppl: {:.3f}\t {:.2f}min {} / {}'.format(
            ppl, (time()-start_time)/60, num_processed, args.data.dev_size))


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

    from utils.model import Embed_LSTM_Model as Model_LM

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        print('enter the TRAINING phrase')
        # train(Model)
        lm_assistant(Model, Model_LM)

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
