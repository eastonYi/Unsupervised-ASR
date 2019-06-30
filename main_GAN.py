#!/usr/bin/env python

from datetime import datetime
from time import time
import os
import tensorflow as tf
tf.config.gpu.set_per_process_memory_growth(True)
import numpy as np
from random import sample

from eastonCode.tfTools.tfData import TFData

from utils.arguments import args
from utils.dataset import ASR_align_DataSet, TextDataSet
from utils.tools import build_optimizer, warmup_exponential_decay, sampleFrames, batch_cer, gradient_penalty, frames_constrain_loss
from utils.model import PhoneClassifier, PhoneDiscriminator


ITERS = 200000 # How many iterations to train for
CRITIC_ITERS = 3

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

        # transformation_func = tf.data.experimental.bucket_by_sequence_length(
        #     element_length_func=lambda x,*y: tf.shape(x)[0],
        #     bucket_boundaries=args.list_bucket_boundaries,
        #     bucket_batch_sizes=args.list_batch_size,
        #     padded_shapes=([None, args.dim_input], [None], [None]))
        iter_train = iter(tfdata_train.cache().repeat().shuffle(500).padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=5))
        # iter_train = iter(tfdata_train.repeat().shuffle(500).apply(transformation_func).prefetch(buffer_size=5))
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

        # text data
        dataset_text = TextDataSet(
            list_files=[args.dirs.lm.data],
            args=args,
            _shuffle=True)

        tfdata_train = tf.data.Dataset.from_generator(
            dataset_text, (tf.int32), (tf.TensorShape([None])))
        iter_text = iter(tfdata_train.cache().\
            repeat().shuffle(10000).padded_batch(args.batch_size, ([None])).prefetch(buffer_size=5))

    # create model paremeters
    G = PhoneClassifier(args)
    D = PhoneDiscriminator(args)

    start_time = datetime.now()

    for iteration in range(ITERS):

        if iteration == 1:
            G.summary()
            D.summary()

        start = time()
        x, _, aligns = next(iter_train)
        cost_G, fs = train_G(x, aligns, G, D, G.trainable_variables, args.lambda_fs)

        for _ in range(CRITIC_ITERS):
            text = next(iter_text)
            P_Real = tf.one_hot(text, args.dim_output)
            x, _, aligns = next(iter_train)
            cost_D, gp = train_D(x, aligns, P_Real, G, D, D.trainable_variables, args.lambda_gp)

        if iteration % 1 == 0:
            print('cost_G: {:.3f}|{:.3f}\tcost_D: {:.3f}|{:.3f}\tused: {:.3f}'.format(cost_G, fs, cost_D, gp, time()-start))
        if iteration % args.dev_step == 0:
            evaluation(tfdata_dev, G)
        if iteration % args.decode_step == 0:
            decode(dataset_dev, G)


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
        P_output = model(x)
        acc = model.align_accuracy(P_output, y)
        list_acc.append(acc)
        preds = model.get_predicts(P_output)
        batch_cer_dist, batch_cer_len = batch_cer(preds.numpy(), y)
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len
        num_processed += len(x)
        progress = num_processed / args.data.dev_size

    cer = total_cer_dist/total_cer_len
    print('dev FER: {:.3f}\t dev PER: {:.3f}\t {:.2f}min {} / {}'.format(
           1-np.mean(list_acc), cer, (time()-start_time)/60, num_processed, args.data.dev_size))


def decode(dataset, model):
    sample = dataset[0]
    x = np.array([sample['feature']], dtype=np.float32)
    P_output = model(x)
    predits = model.get_predicts(P_output)
    print('predits: \n', predits.numpy()[0])
    print('label: \n', sample['label'])
    print('align: ', sample['align'])


def train_G(x, aligns, G, D, params_G, lambda_fs):
    with tf.GradientTape(watch_accessed_variables=False) as tape_G:
        tape_G.watch(params_G)
        _P_G, P_G = G(x, aligns)
        disc_fake = D(_P_G)

        gen_cost = -tf.reduce_mean(disc_fake)
        fs = frames_constrain_loss(P_G, aligns)
        gen_cost += lambda_fs * fs

    gradients_G = tape_G.gradient(gen_cost, params_G)
    G.optimizer.apply_gradients(zip(gradients_G, params_G))

    return gen_cost, fs


def train_D(x, aligns, P_Real, G, D, params_D, lambda_gp):
    with tf.GradientTape(watch_accessed_variables=False) as tape_D:
        tape_D.watch(params_D)
        _P_G, P_G = G(x, aligns)
        disc_real = D(P_Real) # to be +inf
        disc_fake = D(_P_G) # to be -inf

        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        gp = gradient_penalty(D, P_Real, _P_G)
        disc_cost += lambda_gp * gp

    gradients_D = tape_D.gradient(disc_cost, params_D)
    D.optimizer.apply_gradients(zip(gradients_D, params_D))

    return disc_cost, gp


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)

    from utils.model import Embed_LSTM_Model as Model_LM

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        print('enter the TRAINING phrase')
        train(args.Model)
        # lm_assistant(Model, Model_LM)

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
