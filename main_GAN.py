#!/usr/bin/env python

from datetime import datetime
from time import time
import os
import tensorflow as tf

from utils.arguments import args
from utils.dataset import ASR_align_DataSet, TextDataSet
from utils.tools import TFData, gradient_penalty, frames_constrain_loss, aligns2indices,\
    CE_loss, evaluation, monitor, decode

from models.GAN import PhoneClassifier, PhoneDiscriminator2, PhoneDiscriminator3


tf.random.set_seed(args.seed)

def train():
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
                        args=args).read(_shuffle=False)
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)

        x_0, y_0, _ = next(iter(tfdata_train.take(args.num_supervised).map(lambda x, y, z: (x, y, z[:args.max_seq_len])).\
            padded_batch(args.num_supervised, ([None, args.dim_input], [None], [None]))))
        iter_train = iter(tfdata_train.cache().repeat().shuffle(3000).map(lambda x, y, z: (x, y, z[:args.max_seq_len])).\
            padded_batch(args.batch_size, ([None, args.dim_input], [None], [args.max_seq_len])).prefetch(buffer_size=3))
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

        # text data
        dataset_text = TextDataSet(
            list_files=[args.dirs.lm.data],
            args=args,
            _shuffle=True)

        tfdata_train_text = tf.data.Dataset.from_generator(
            dataset_text, (tf.int32), (tf.TensorShape([None])))
        iter_text = iter(tfdata_train_text.cache().repeat().shuffle(100).map(lambda x: x[:args.max_seq_len]).padded_batch(args.batch_size,
            ([args.max_seq_len])).prefetch(buffer_size=5))

    # create model paremeters
    G = PhoneClassifier(args)
    D = PhoneDiscriminator3(args)
    G.summary()
    D.summary()
    optimizer_G = tf.keras.optimizers.Adam(args.opti.G.lr, beta_1=0.5, beta_2=0.9)
    optimizer_D = tf.keras.optimizers.Adam(args.opti.D.lr, beta_1=0.5, beta_2=0.9)
    optimizer = tf.keras.optimizers.Adam(args.opti.G.lr, beta_1=0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(G=G, optimizer_G = optimizer_G)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=5)
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

        for _ in range(args.opti.D_G_rate):
            x, _, aligns = next(iter_train)
            text = next(iter_text)
            P_Real = tf.one_hot(text, args.dim_output)
            cost_D, gp = train_D(x, aligns, P_Real, text>0, G, D, optimizer_D, args.lambda_gp)

        x, _, aligns = next(iter_train)
        cost_G, fs = train_G(x, aligns, G, D, optimizer_G, args.lambda_fs)
        loss_supervise = train_G_supervised(x_0, y_0, G, optimizer_G, args.dim_output)

        num_processed += len(x)
        if step % 10 == 0:
            print('cost_G: {:.3f}|{:.3f}\tcost_D: {:.3f}|{:.3f}\tloss_supervise: {:.3f}\tbatch: {}|{}\tused: {:.3f}\t {:.3f}% iter: {}'.format(
                   cost_G, fs, cost_D, gp, loss_supervise, x.shape, text.shape, time()-start, progress*100.0, step))
            with writer.as_default():
                tf.summary.scalar("costs/cost_G", cost_G, step=step)
                tf.summary.scalar("costs/cost_D", cost_D, step=step)
                tf.summary.scalar("costs/gp", gp, step=step)
                tf.summary.scalar("costs/fs", fs, step=step)
                tf.summary.scalar("costs/loss_supervise", loss_supervise, step=step)
        if step % args.dev_step == 0:
            fer, cer = evaluation(tfdata_dev, args.data.dev_size, G)
            with writer.as_default():
                tf.summary.scalar("performance/fer", fer, step=step)
                tf.summary.scalar("performance/cer", cer, step=step)
        if step % args.decode_step == 0:
            monitor(dataset_dev[0], G)
        if step % args.save_step == 0:
            save_path = ckpt_manager.save(step)
            print('save model {}'.format(save_path))

        step += 1

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def train_G(x, aligns, G, D, optimizer_G, lambda_fs):
    indices = aligns2indices(aligns)
    params_G = G.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape_G:
        tape_G.watch(params_G)
        logits = G(x, training=True)
        P_G = tf.nn.softmax(logits)
        _P_G = tf.gather_nd(P_G, indices)
        # disc_fake = D([_P_G, aligns>0], training=True)
        disc_fake = D(_P_G, training=True)

        gen_cost = -tf.reduce_mean(disc_fake)
        # gen_cost = tf.reduce_mean(tf.math.squared_difference(disc_fake, 1))
        fs = frames_constrain_loss(logits, aligns)
        # fs = 0
        gen_cost += lambda_fs * fs

    gradients_G = tape_G.gradient(gen_cost, params_G)
    optimizer_G.apply_gradients(zip(gradients_G, params_G))

    return gen_cost, fs


def train_D(x, aligns, P_Real, mask_real, G, D, optimizer_D, lambda_gp):
    indices = aligns2indices(aligns)
    params_D = D.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape_D:
        tape_D.watch(params_D)
        logits= G(x, training=True)
        P_G = tf.nn.softmax(logits)
        _P_G = tf.gather_nd(P_G, indices)
        # disc_real = D([P_Real, mask_real], training=True) # to be +inf
        # disc_fake = D([_P_G, aligns>0], training=True) # to be -inf
        disc_real = D(P_Real, training=True) # to be +inf
        disc_fake = D(_P_G, training=True) # to be -inf

        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        # disc_cost = tf.reduce_mean(disc_fake**2) + tf.reduce_mean(tf.math.squared_difference(disc_real, 1))
        # gp = gradient_penalty(D, P_Real, _P_G, mask_real=mask_real, mask_fake=aligns>0)
        gp = gradient_penalty(D, P_Real, _P_G)
        disc_cost += lambda_gp * gp

    gradients_D = tape_D.gradient(disc_cost, params_D)
    optimizer_D.apply_gradients(zip(gradients_D, params_D))

    return disc_cost, gp


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
        train()


        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
