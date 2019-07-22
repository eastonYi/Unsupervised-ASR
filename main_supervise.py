#!/usr/bin/env python

from datetime import datetime
from time import time
import os
import tensorflow as tf
import numpy as np

from eastonCode.tfTools.tfData import TFData

from utils.arguments import args
from utils.dataset import ASR_align_DataSet, TextDataSet
from utils.tools import batch_cer, gradient_penalty, frames_constrain_loss, aligns2indices, align_accuracy, get_predicts, CE_loss

from models.GAN import PhoneClassifier


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
    G.summary()
    optimizer_G = tf.keras.optimizers.Adam(args.opti.G.lr, beta_1=0.5, beta_2=0.9)

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

        x, y, aligns = next(iter_train)
        loss_supervise = train_G_supervised(x, y, G, optimizer_G, args.dim_output)

        if iteration % 10 == 0:
            print('loss_supervise: {:.3f}\tbatch: {}\tused: {:.3f}\titer: {}'.format(
                   loss_supervise, x.shape, time()-start, iteration))
            with writer.as_default():
                tf.summary.scalar("costs/loss_supervise", loss_supervise, step=iteration)
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
           fer, cer, (time()-start_time)/60, num_processed, args.data.dev_size))

    return fer, cer


def decode(dataset, model):
    sample = dataset[0]
    x = np.array([sample['feature']], dtype=np.float32)
    logits = model(x)
    predits = get_predicts(logits)
    print('predits: \n', predits.numpy()[0])
    print('label: \n', sample['label'])
    print('align: ', sample['align'])


@tf.function
def train_G_supervised(x, labels, G, optimizer_G, dim_output):
    with tf.GradientTape() as tape_G:
        logits = G(x, training=True)
        ce_loss = CE_loss(logits, labels, dim_output, confidence=0.9)
        gen_loss = ce_loss

    gradients_G = tape_G.gradient(gen_loss, G.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, G.trainable_variables))

    return gen_loss


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
        args.dir_exps = args.dir_exps /  param.name
        args.dir_log = args.dir_exps / 'log'
        args.dir_checkpoint = args.dir_exps / 'checkpoint'
        if args.dir_exps.is_dir():
            os.system('rm -r '+ str(args.dir_exps))
        args.dir_exps.mkdir()
        args.dir_log.mkdir()
        args.dir_checkpoint.mkdir()
        with open(args.dir_exps / 'configs.txt', 'w') as fw:
            print(args, file=fw)

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        assert len(gpus) > 0, "Not enough GPU hardware devices available"
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
        print('enter the TRAINING phrase')
        train(args.Model)


        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
