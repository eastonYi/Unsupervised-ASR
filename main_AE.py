#!/usr/bin/env python

from datetime import datetime
from time import time
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.arguments import args
from utils.dataset import ASR_align_DataSet
from utils.tools import TFData, get_GRU_activation
from models.AE_GRNN import GRNN, GRNN_Cell
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def train():
    with tf.device("/cpu:0"):
        dataset_train = ASR_align_DataSet(
            uttid2wav=args.dirs.train.wav_scp,
            trans_file=None,
            align_file=None,
            feat_len_file=None,
            args=args,
            _shuffle=True,
            transform=True)
        dataset_dev = ASR_align_DataSet(
            uttid2wav=args.dirs.dev.wav_scp,
            trans_file=None,
            align_file=None,
            feat_len_file=None,
            args=args,
            _shuffle=False,
            transform=True)
        feature_train = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args).read()
        feature_dev = TFData(dataset=dataset_dev,
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read()

        feature_train = feature_train.cache().repeat().shuffle(500).padded_batch(
            args.batch_size, ((), [None, args.dim_input])).prefetch(buffer_size=5)
        feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    # create model paremeters
    model = GRNN(args)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(args.opti.lr, beta_1=0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=10)
    step = 0

    if args.dirs.checkpoint:
        _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
        ckpt.restore(_ckpt_manager.latest_checkpoint)
        print('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))
        step = int(_ckpt_manager.latest_checkpoint.split('-')[-1])

    start_time = datetime.now()
    num_processed = 0
    progress = 0

    for batch in feature_train:
        start = time()

        uttids, x = batch
        loss = train_step(x, model, optimizer)

        num_processed += len(x)
        progress = num_processed / args.data.train_size
        if step % 10 == 0:
            print('loss: {:.3f}\tbatch: {}\tused: {:.3f}\t {:.3f}% step: {}'.format(
                  loss, x.shape, time()-start, progress*100.0, step))
            with writer.as_default():
                tf.summary.scalar("losses/loss", loss, step=step)
        if step % args.dev_step == 0:
            loss_dev = evaluate(feature_dev, model)
            with writer.as_default():
                tf.summary.scalar("losses/loss_dev", loss_dev, step=step)
        if step % args.save_step == 0:
            save_path = ckpt_manager.save(step)
            print('save model {}'.format(save_path))

        step += 1

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


@tf.function
def train_step(x, model, optimizer):
    musk = tf.cast(tf.reduce_max(tf.abs(x), -1) > 0, tf.float32)
    params = model.trainable_variables
    with tf.GradientTape() as tape:
        _x = model(x, training=True)
        loss_mat = tf.reduce_mean(tf.pow(x-_x, 2), -1) * musk
        loss_batch = tf.reduce_sum(loss_mat, -1) / tf.reduce_sum(musk, -1)
        loss = tf.reduce_mean(loss_batch)

    gradients = tape.gradient(loss, params)
    optimizer.apply_gradients(zip(gradients, params))

    return loss


def evaluate(feature, model):
    list_loss = []
    for batch in feature:
        uttids, x = batch
        musk = tf.cast(tf.reduce_max(tf.abs(x), -1) > 0, tf.float32)
        _x = model(x, training=True)
        loss_mat = tf.reduce_mean(tf.pow(x-_x, 2), -1) * musk
        loss_batch = tf.reduce_sum(loss_mat, -1) / tf.reduce_sum(musk, -1)
        loss = tf.reduce_mean(loss_batch)
        list_loss.append(loss)

    return tf.reduce_mean(list_loss)


def Decode(save_file):
    dataset = ASR_align_DataSet(
        trans_file=None,
        align_file=None,
        uttid2wav=args.dirs.train.wav_scp,
        feat_len_file=args.dirs.train.feat_len,
        args=args,
        _shuffle=False,
        transform=True)
    dataset_dev = ASR_align_DataSet(
        trans_file=args.dirs.dev.trans,
        align_file=args.dirs.dev.align,
        uttid2wav=args.dirs.dev.wav_scp,
        feat_len_file=args.dirs.dev.feat_len,
        args=args,
        _shuffle=False,
        transform=True)

    feature_train = TFData(dataset=dataset,
                    dir_save=args.dirs.train.tfdata,
                    args=args).read()
    feature_dev = TFData(dataset=dataset_dev,
                    dir_save=args.dirs.dev.tfdata,
                    args=args).read()

    feature_train = feature_train.cache().repeat().shuffle(500).padded_batch(
        args.batch_size, ((), [None, args.dim_input])).prefetch(buffer_size=5)
    feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    # model = GRNN(args)
    model = GRNN_Cell(args)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(1e-4)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('checkpoint {} restored!!'.format(ckpt_manager.latest_checkpoint))

    # plot_align(model, dataset_dev, 0)

    for batch in feature_train:
        uttids, x = batch
        batch_size = len(x)
        h1 = h2 = tf.zeros([batch_size, args.model.num_hidden_rnn])
        for x_t in tf.split(x, x.shape[1], axis=1):
    
            x_fc, x_cell, h1, h2 = model([x_t, h1, h2], training=False)
            z, r = get_GRU_activation(model.layers[3], cell_inputs=x_fc[:, 0, :], hiddens=h1)


def plot_align(model, dataset_dev, i):
    uttid = dataset_dev.list_uttids[i]
    x = dataset_dev.get_attrs('feature', [uttid])
    x = x[0][None, :, :]
    stamps = dataset_dev.get_attrs('stamps', [uttid])

    lists = [[], []]
    h1 = h2 = tf.zeros([1, args.model.num_hidden_rnn])
    _z = 0
    for x_t in tf.split(x, x.shape[1], axis=1):
        x_fc, x_cell, h1, h2 = model([x_t, h1, h2], training=False)
        # z, r = get_GRU_activation(model.layers[3], cell_inputs=x_fc[:, 0, :], hiddens=h1)
        z, r = get_GRU_activation(model.layers[5], cell_inputs=x_cell[:, 0, :], hiddens=h2)
        lists[0].append(z[0].numpy())
        lists[1].append(z[0].numpy()-_z)
        _z = z[0].numpy()
    lists[1][0] = lists[1][1]

    fig = plt.figure(figsize=(40, 10))
    ax = fig.add_subplot(111)
    ax.set_title('alignment')
    ax.plot(lists[0]-np.mean(lists[0]), label='org')
    ax.plot(lists[1]-np.mean(lists[1]), label='delta')

    ax.legend(loc='upper right')
    [ax.axvline(stamp, c='purple') for stamp in stamps[0]]
    fig.savefig('output/' + uttid+ '.png')
    print('plot', 'output/'+uttid+'.png')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', param.gpu)
    # os.environ["CUDA_VISIBLE_DEVICES"] = ''
    os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

    if param.name:
        args.dir_exps = args.dir_exps /  param.name
        args.dir_log = args.dir_exps / 'log'
        args.dir_checkpoint = args.dir_exps / 'checkpoint'
        if not args.dir_exps.is_dir(): args.dir_exps.mkdir()
        if not args.dir_log.is_dir(): args.dir_log.mkdir()
        if not args.dir_checkpoint.is_dir(): args.dir_checkpoint.mkdir()
        with open(args.dir_exps / 'configs.txt', 'w') as fw:
            print(args, file=fw)

    if param.mode == 'train':
        print('enter the TRAINING phrase')
        train()

    elif param.mode == 'decode':
        """
        python main_supervise.py -m decode --name timit_supervised2_decode.txt --gpu 0 -c configs/timit_supervised2.yaml
        """
        print('enter the DECODING phrase')
        assert args.dirs.checkpoint
        assert param.name

        Decode(param.name)


        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
