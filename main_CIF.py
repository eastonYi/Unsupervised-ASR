#!/usr/bin/env python

from datetime import datetime
from time import time
import os
import tensorflow as tf
import numpy as np

from utils.arguments import args
from utils.dataset import ASR_align_DataSet, TextDataSet
from utils.tools import TFData, gradient_penalty, CE_loss, decode, get_predicts, batch_cer

# from models.GAN import PhoneClassifier, PhoneDiscriminator3
from models.CIF import attentionAssign, CIF, PhoneClassifier
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(args.seed)


def train():
    with tf.device("/cpu:0"):
        dataset_train = ASR_align_DataSet(
            trans_file=args.dirs.train.trans,
            align_file=None,
            uttid2wav=args.dirs.train.wav_scp,
            feat_len_file=args.dirs.train.feat_len,
            args=args,
            _shuffle=True,
            transform=True)
        dataset_dev = ASR_align_DataSet(
            trans_file=args.dirs.dev.trans,
            align_file=None,
            uttid2wav=args.dirs.dev.wav_scp,
            feat_len_file=args.dirs.dev.feat_len,
            args=args,
            _shuffle=False,
            transform=True)
        feature_train = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args).read()
        feature_dev = TFData(dataset=dataset_dev,
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read()
        iter_feature_train = iter(feature_train.cache().repeat().shuffle(500).padded_batch(args.batch_size,
                ((), [None, args.dim_input])).prefetch(buffer_size=5))
        feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))


    # create model paremeters
    assigner = attentionAssign(args)
    G = PhoneClassifier(args, dim_input=args.model.attention.num_hidden)
    assigner.summary()
    G.summary()

    optimizer_G = tf.keras.optimizers.Adam(args.opti.G.lr, beta_1=0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(G=G, optimizer_G=optimizer_G)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    step = 0

    # if a checkpoint exists, restore the latest checkpoint.
    if args.dirs.checkpoint:
        _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
        ckpt.restore(_ckpt_manager.latest_checkpoint)
        print('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))
        step = int(_ckpt_manager.latest_checkpoint.split('-')[-1])

    start_time = datetime.now()
    num_processed = 0
    progress = 0

    while step < 99999999:
        start = time()

        uttids, x = next(iter_feature_train)
        y = dataset_train.get_attrs('trans', uttids.numpy())
        ce_loss_supervise, quantity_loss_supervise = train_G_supervised(
            x, y, assigner, G, optimizer_G, args.dim_output, args.lambda_supervision)

        num_processed += len(x)
        progress = num_processed / args.data.train_size
        if step % 10 == 0:
            print('loss_supervise: {:.3f}|{:.3f}\tbatch: {}|{}\tused: {:.3f}\t {:.3f}% iter: {}'.format(
                   ce_loss_supervise, quantity_loss_supervise, x.shape, None, time()-start, progress*100.0, step))
            # with writer.as_default():
            #     tf.summary.scalar("costs/cost_G", cost_G, step=step)
            #     tf.summary.scalar("costs/cost_D", cost_D, step=step)
            #     tf.summary.scalar("costs/gp", gp, step=step)
            #     tf.summary.scalar("costs/loss_supervise", loss_supervise, step=step)
        if step % args.dev_step == 0:
            cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, assigner, G)
            with writer.as_default():
                tf.summary.scalar("performance/cer", cer, step=step)
        if step % args.decode_step == 0:
            monitor(dataset_dev[0], assigner, G)

        step += 1

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


# @tf.function
def train_G_supervised(x, y, assigner, G, optimizer, dim_output, lambda_supervision):
    vars = G.trainable_variables + assigner.trainable_variables
    with tf.GradientTape() as tape:
        hidden, alpha = assigner(x)
        musk = tf.cast(tf.reduce_sum(tf.abs(x), -1) > 0, tf.float32)
        alpha *= musk
        # sum
        _num = tf.reduce_sum(alpha, -1)
        # scaling
        num = tf.reduce_sum(tf.cast(y>0, tf.float32), -1)
        alpha *= tf.tile((num/_num)[:, None], [1, alpha.shape[1]])

        l = CIF(hidden, alpha, threshold=args.model.attention.threshold)
        logits= G(l, training=True)
        ce_loss = CE_loss(logits, y, dim_output, confidence=0.9)

        quantity_loss = tf.reduce_mean(tf.losses.mean_squared_error(_num, num))

        loss = ce_loss + quantity_loss * 1.0

    gradients = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(gradients, vars))

    return ce_loss, quantity_loss


def evaluate(feature, dataset, dev_size, assigner, model):
    num_processed = 0
    total_cer_dist = 0
    total_cer_len = 0
    for batch in feature:
        uttids, x = batch
        # trans = dataset.get_attrs('trans', uttids.numpy(), args.max_label_len)
        trans = dataset.get_attrs('trans', uttids.numpy())

        hidden, alpha = assigner(x)
        musk = tf.cast(tf.reduce_sum(tf.abs(x), -1) > 0, tf.float32)
        alpha *= musk
        l = CIF(hidden, alpha, threshold=args.model.attention.threshold)
        logits = model(l)
        preds = get_predicts(logits)

        batch_cer_dist, batch_cer_len = batch_cer(preds.numpy(), trans)
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len

        num_processed += len(x)

    cer = total_cer_dist/total_cer_len
    print('dev PER: {:.3f}\t {} / {}'.format(cer, num_processed, dev_size))

    return cer


def monitor(sample, assigner, model):
    x = np.array([sample['feature']], dtype=np.float32)
    hidden, alpha = assigner(x)
    # l = CIF(x, alpha, threshold=args.model.G.threshold, max_label_len=args.max_label_len)
    l = CIF(hidden, alpha, threshold=args.model.attention.threshold)
    logits = model(l)
    predicts = get_predicts(logits)
    print('predicts: \n', predicts.numpy()[0])
    print('trans: \n', sample['trans'])


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', param.gpu)
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

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
