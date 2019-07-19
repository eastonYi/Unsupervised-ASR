#!/usr/bin/env python

from datetime import datetime
from time import time
import os
import tensorflow as tf
import numpy as np

from eastonCode.tfTools.tfData import TFData

from utils.arguments import args
from utils.dataset import ASR_align_DataSet, TextDataSet
from utils.tools import batch_cer, gradient_penalty, frames_constrain_loss, aligns2indices, align_accuracy, get_predicts
from utils.model import PhoneClassifier, PhoneDiscriminator


ITERS = 200000 # How many iterations to train for
tf.random.set_seed(args.seed)

def train(Model):
    dataset_dev = ASR_align_DataSet(
        file=[args.dirs.dev.data],
        args=args,
        _shuffle=False,
        transform=True)
    with tf.device("/cpu:0"):
        # wav data
        tfdata_train = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=True)
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)

        iter_train = iter(tfdata_train.cache().repeat().shuffle(500).padded_batch(args.batch_size,
                ([None, args.dim_input], [None], [None])).prefetch(buffer_size=5))
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

        # text data
        dataset_text = TextDataSet(
            list_files=[args.dirs.lm.data],
            args=args,
            _shuffle=True)

        tfdata_train = tf.data.Dataset.from_generator(
            dataset_text, (tf.int32), (tf.TensorShape([None])))
        iter_text = iter(tfdata_train.cache().repeat().shuffle(100).padded_batch(args.batch_size,
            ([None])).prefetch(buffer_size=5))

    # create model paremeters
    G = PhoneClassifier(args)
    D = PhoneDiscriminator(args)
    G.summary()
    D.summary()
    optimizer_G = tf.keras.optimizers.Adam(args.opti.G.lr, beta_1=0.5, beta_2=0.9)
    optimizer_D = tf.keras.optimizers.Adam(args.opti.D.lr, beta_1=0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(G=G, optimizer_G = optimizer_G)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if args.dirs.checkpoint:
        _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
        ckpt.restore(_ckpt_manager.latest_checkpoint)
        print ('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))

    start_time = datetime.now()

    for iteration in range(ITERS):

        start = time()

        for _ in range(args.opti.D_G_rate):
            x, _, aligns = next(iter_train)
            text = next(iter_text)
            P_Real = tf.one_hot(text, args.dim_output)
            cost_D, gp = train_D(x, aligns, P_Real, text>0, G, D, optimizer_D, args.lambda_gp)

        x, _, aligns = next(iter_train)
        cost_G, fs = train_G(x, aligns, G, D, optimizer_G, args.lambda_fs)

        if iteration % 10 == 0:
            print('cost_G: {:.3f}|{:.3f}\tcost_D: {:.3f}|{:.3f}\tbatch: {}\tused: {:.3f}\titer: {}'.format(
                   cost_G, fs, cost_D, gp, x.shape, time()-start, iteration))
            with writer.as_default():
                tf.summary.scalar("costs/cost_G", cost_G, step=iteration)
                tf.summary.scalar("costs/cost_D", cost_D, step=iteration)
                tf.summary.scalar("costs/gp", gp, step=iteration)
                tf.summary.scalar("costs/fs", fs, step=iteration)
        if iteration % args.dev_step == 0:
            fer, cer = evaluation(tfdata_dev, G)
            with writer.as_default():
                tf.summary.scalar("performance/fer", fer, step=iteration)
                tf.summary.scalar("performance/cer", cer, step=iteration)
        if iteration % args.decode_step == 0:
            decode(dataset_dev, G)
        if iteration % args.save_step == 0:
            save_path = ckpt_manager.save()
            print('save model {}'.format(save_path))

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def evaluation(tfdata_dev, model):
    list_acc = []

    start_time = time()
    num_processed = 0
    progress = 0
    total_cer_dist = 0
    total_cer_len = 0
    for batch in tfdata_dev:
        x, y, aligns = batch
        logits = model(x)
        acc = align_accuracy(logits, y)
        list_acc.append(acc)
        preds = get_predicts(logits)
        batch_cer_dist, batch_cer_len = batch_cer(preds.numpy(), y)
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len
        num_processed += len(x)
        progress = num_processed / args.data.dev_size

    cer = total_cer_dist/total_cer_len
    fer = 1-np.mean(list_acc)
    print('dev FER: {:.3f}\t dev PER: {:.3f}\t {:.2f}min {} / {}'.format(
           1-np.mean(list_acc), cer, (time()-start_time)/60, num_processed, args.data.dev_size))

    return fer, cer


def decode(dataset, model):
    sample = dataset[0]
    x = np.array([sample['feature']], dtype=np.float32)
    logits = model(x)
    predits = get_predicts(logits)
    print('predits: \n', predits.numpy()[0])
    print('label: \n', sample['label'])
    print('align: ', sample['align'])


def train_G(x, aligns, G, D, optimizer_G, lambda_fs):
    indices = aligns2indices(aligns)
    params_G = G.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape_G:
        tape_G.watch(params_G)
        logits = G(x)
        P_G = tf.nn.softmax(logits)
        _P_G = tf.gather_nd(P_G, indices)
        disc_fake = D([_P_G, aligns>0])

        gen_cost = -tf.reduce_mean(disc_fake)
        fs = frames_constrain_loss(logits, aligns)
        gen_cost += lambda_fs * fs

    gradients_G = tape_G.gradient(gen_cost, params_G)
    optimizer_G.apply_gradients(zip(gradients_G, params_G))

    return gen_cost, fs


def train_D(x, aligns, P_Real, mask_real, G, D, optimizer_D, lambda_gp):
    indices = aligns2indices(aligns)
    params_D = D.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape_D:
        tape_D.watch(params_D)
        logits= G(x)
        P_G = tf.nn.softmax(logits)
        _P_G = tf.gather_nd(P_G, indices)
        disc_real = D([P_Real, mask_real]) # to be +inf
        disc_fake = D([_P_G, aligns>0]) # to be -inf

        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        gp = gradient_penalty(D, P_Real, _P_G, mask_real=mask_real, mask_fake=aligns>0)
        disc_cost += lambda_gp * gp

    gradients_D = tape_D.gradient(disc_cost, params_D)
    optimizer_D.apply_gradients(zip(gradients_D, params_D))

    return disc_cost, gp


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', param.gpu)

    if param.name:
        args.dir_model = args.dir_model /  param.name
        args.dir_log = args.dir_model / 'log'
        args.dir_checkpoint = args.dir_model / 'checkpoint'
        if args.dir_model.is_dir():
            os.system('rm -r '+ args.dir_model.name)
        args.dir_model.mkdir()
        args.dir_log.mkdir()
        args.dir_checkpoint.mkdir()
        with open(args.dir_model / 'configs.txt', 'w') as fw:
            print(args, file=fw)

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        assert len(gpus) > 0, "Not enough GPU hardware devices available"
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
        print('enter the TRAINING phrase')
        train(args.Model)


        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
