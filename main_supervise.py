#!/usr/bin/env python

from datetime import datetime
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np

from utils.tools import TFData
from utils.arguments import args
from utils.dataset import ASR_align_DataSet
from utils.tools import CE_loss, evaluate, monitor
from utils.wfst import WFST_Decoder

from models.GAN import PhoneClassifier
# from models.GAN import PhoneClassifier2 as PhoneClassifier

ITERS = 200000 # How many iterations to train for
tf.random.set_seed(args.seed)


def Train():
    dataset_train = ASR_align_DataSet(
        trans_file=args.dirs.train.trans,
        align_file=args.dirs.train.align,
        uttid2wav=args.dirs.train.wav_scp,
        feat_len_file=args.dirs.train.feat_len,
        args=args,
        _shuffle=True,
        transform=True)
    dataset_dev = ASR_align_DataSet(
        trans_file=args.dirs.dev.trans,
        align_file=args.dirs.dev.align,
        uttid2wav=args.dirs.dev.wav_scp,
        feat_len_file=args.dirs.dev.feat_len,
        args=args,
        _shuffle=False,
        transform=True)
    with tf.device("/cpu:0"):
        # wav data
        feature_train = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args).read()
        feature_dev = TFData(dataset=dataset_dev,
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read()
        if args.num_supervised:
            dataset_train_supervise = ASR_align_DataSet(
                trans_file=args.dirs.train_supervise.trans,
                align_file=args.dirs.train_supervise.align,
                uttid2wav=args.dirs.train_supervise.wav_scp,
                feat_len_file=args.dirs.train_supervise.feat_len,
                args=args,
                _shuffle=False,
                transform=True)
            feature_train_supervise = TFData(dataset=dataset_train_supervise,
                            dir_save=args.dirs.train_supervise.tfdata,
                            args=args).read()
            supervise_uttids, supervise_x = next(iter(feature_train_supervise.take(args.num_supervised).\
                padded_batch(args.num_supervised, ((), [None, args.dim_input]))))
            supervise_aligns = dataset_train_supervise.get_attrs('align', supervise_uttids.numpy())
            # supervise_bounds = dataset_train_supervise.get_attrs('bounds', supervise_uttids.numpy())

        iter_feature_train = iter(feature_train.repeat().shuffle(500).padded_batch(args.batch_size,
                ((), [None, args.dim_input])).prefetch(buffer_size=5))
        feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    # create model paremeters
    model = PhoneClassifier(args)
    model.summary()
    optimizer_G = tf.keras.optimizers.Adam(args.opti.lr, beta_1=0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(model=model, optimizer_G=optimizer_G)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    step = 0

    # if a checkpoint exists, restore the latest checkpoint.
    if args.dirs.checkpoint:
        _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
        ckpt.restore(_ckpt_manager.latest_checkpoint)
        print('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))
        step = int(_ckpt_manager.latest_checkpoint.split('-')[-1])

    start_time = datetime.now()

    while step < 99999999:
        start = time()

        if args.num_supervised:
            x = supervise_x
            loss_supervise = train_G_supervised(supervise_x, supervise_aligns, model, optimizer_G, args.dim_output)
            # loss_supervise, bounds_loss = train_G_bounds_supervised(
            #     x, supervise_bounds, supervise_aligns, model, optimizer_G, args.dim_output)
        else:
            uttids, x = next(iter_feature_train)
            aligns = dataset_train.get_attrs('align', uttids.numpy())
            # trans = dataset_train.get_attrs('trans', uttids.numpy())
            loss_supervise = train_G_supervised(x, aligns, model, optimizer_G, args.dim_output)
            # loss_supervise = train_G_TBTT_supervised(x, aligns, model, optimizer_G, args.dim_output)
            # bounds = dataset_train.get_attrs('bounds', uttids.numpy())
            # loss_supervise, bounds_loss = train_G_bounds_supervised(x, bounds, aligns, model, optimizer_G, args.dim_output)
            # loss_supervise = train_G_CTC_supervised(x, trans, model, optimizer_G)

        if step % 10 == 0:
            print('loss_supervise: {:.3f}\tbatch: {}\tused: {:.3f}\tstep: {}'.format(
                   loss_supervise, x.shape, time()-start, step))
            # print('loss_supervise: {:.3f}\tloss_bounds: {:.3f}\tbatch: {}\tused: {:.3f}\tstep: {}'.format(
            #        loss_supervise, bounds_loss, x.shape, time()-start, step))
            with writer.as_default():
                tf.summary.scalar("costs/loss_supervise", loss_supervise, step=step)
        if step % args.dev_step == 0:
            fer, cer_0 = evaluate(feature_dev, dataset_dev, args.data.dev_size, model, beam_size=0, with_stamp=True)
            fer, cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, model, beam_size=0, with_stamp=False)
            with writer.as_default():
                tf.summary.scalar("performance/fer", fer, step=step)
                tf.summary.scalar("performance/cer_0", cer_0, step=step)
                tf.summary.scalar("performance/cer", cer, step=step)
        if step % args.decode_step == 0:
            monitor(dataset_dev[0], model)
        if step % args.save_step == 0:
            save_path = ckpt_manager.save(step)
            print('save model {}'.format(save_path))

        step += 1

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


# @tf.function
def train_G_supervised(x, labels, model, optimizer_G, dim_output):
    with tf.GradientTape() as tape_G:
        logits = model(x, training=True)
        ce_loss = CE_loss(logits, labels, dim_output, confidence=0.9)
        gen_loss = ce_loss

    gradients_G = tape_G.gradient(gen_loss, model.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, model.trainable_variables))

    return gen_loss

@tf.function
def train_G_TBTT_supervised(x, labels, model, optimizer_G, dim_output):
    """
    random cut head & make it can be split evenly
    """
    len_seq = args.model.G.len_seq
    cut_idx = tf.random.uniform((), minval=0, maxval=len_seq, dtype=tf.dtypes.int32).numpy()
    num_split = int((x.shape[1]-cut_idx) // len_seq)
    max_idx = cut_idx + num_split * len_seq
    # reshape x
    list_tensors = tf.split(x[:, cut_idx:max_idx, :], num_split, axis=1)
    x = tf.concat(list_tensors, 0)
    # reshape label
    list_tensors = tf.split(labels[:, cut_idx:max_idx], num_split, axis=1)
    labels = tf.concat(list_tensors, 0)

    with tf.GradientTape() as tape_G:
        logits = model(x, training=True)
        ce_loss = CE_loss(logits, labels, dim_output, confidence=0.8)
        gen_loss = ce_loss

    gradients_G = tape_G.gradient(gen_loss, model.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, model.trainable_variables))

    return gen_loss


# @tf.function
def train_G_bounds_supervised(x, bounds, labels, model, optimizer_G, dim_output):
    """
    random cut head & make it can be split evenly
    """
    len_seq = args.model.G.len_seq
    cut_idx = tf.random.uniform((), minval=0, maxval=len_seq, dtype=tf.dtypes.int32).numpy()
    num_split = int((x.shape[1]-cut_idx) // len_seq)
    max_idx = cut_idx + num_split * len_seq
    # reshape x
    list_tensors = tf.split(x[:, cut_idx:max_idx, :], num_split, axis=1)
    x = tf.concat(list_tensors, 0)
    # reshape label
    list_tensors = tf.split(bounds[:, cut_idx:max_idx], num_split, axis=1)
    bounds = tf.concat(list_tensors, 0)
    list_tensors = tf.split(labels[:, cut_idx:max_idx], num_split, axis=1)
    labels = tf.concat(list_tensors, 0)

    with tf.GradientTape() as tape_G:
        logits, logits_bounds = model(x, training=True)
        ce_loss = CE_loss(logits, labels, dim_output, confidence=0.8)
        bounds_loss = CE_loss(logits_bounds, bounds, 2, confidence=0.8)
        # bounds_loss = 0
        gen_loss = ce_loss + bounds_loss

    gradients_G = tape_G.gradient(gen_loss, model.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, model.trainable_variables))

    return gen_loss, bounds_loss


def Decode(save_file):
    # dataset = ASR_align_DataSet(
    #     trans_file=args.dirs.train.trans,
    #     align_file=None,
    #     uttid2wav=args.dirs.train.wav_scp,
    #     feat_len_file=args.dirs.train.feat_len,
    #     args=args,
    #     _shuffle=False,
    #     transform=True)
    dataset_dev = ASR_align_DataSet(
        trans_file=args.dirs.dev.trans,
        align_file=args.dirs.dev.align,
        uttid2wav=args.dirs.dev.wav_scp,
        feat_len_file=args.dirs.dev.feat_len,
        args=args,
        _shuffle=False,
        transform=True)
    feature_dev = TFData(dataset=dataset_dev,
                    dir_save=args.dirs.dev.tfdata,
                    args=args).read()
    feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    model = PhoneClassifier(args)
    model.summary()

    optimizer_G = tf.keras.optimizers.Adam(1e-4)
    ckpt = tf.train.Checkpoint(model=model, optimizer_G=optimizer_G)

    _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
    ckpt.restore(_ckpt_manager.latest_checkpoint)
    print('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))
    # fer, cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, model, beam_size=0, with_stamp=True)
    # fer, cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, model, beam_size=0, with_stamp=False)
    decode_outs = np.zeros((300), dtype=np.int32)
    wfst = WFST_Decoder(
        decode_outs=decode_outs,
        fcdll="wfst/WFST_Decode/bin/libctc_wfst_lib.so",
        fcfg="wfst/cfg.json")
    cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, model, wfst=wfst)
    fer = 0
    fer, cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, model, beam_size=10, with_stamp=False)
    print('FER:{:.3f}\t PER:{:.3f}'.format(fer, cer))
    # decode(dataset, model, args.idx2token, 'output/'+save_file)
    # decode(dataset, model, args.idx2token, 'output/'+save_file, align=True)


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

    if param.mode == 'train':
        """
        python main.py -m decode --gpu 1 --name align_supervised_dev.text -c configs/timit_supervised.yaml
        """
        if param.name:
            args.dir_exps = args.dir_exps / param.name
            args.dir_log = args.dir_exps / 'log'
            args.dir_checkpoint = args.dir_exps / 'checkpoint'
            if not args.dir_exps.is_dir(): args.dir_exps.mkdir()
            if not args.dir_log.is_dir(): args.dir_log.mkdir()
            if not args.dir_checkpoint.is_dir(): args.dir_checkpoint.mkdir()
        print('enter the TRAINING phrase')
        Train()

    elif param.mode == 'decode':
        """
        python main_supervise.py -m decode --name timit_supervised2_decode.txt --gpu 0 -c configs/timit_supervised2.yaml
        """
        print('enter the DECODING phrase')
        assert args.dirs.checkpoint
        assert param.name
        Decode(param.name)
