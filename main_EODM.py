#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import tensorflow as tf

from utils.dataset import ASR_align_DataSet
from utils.arguments import args
from utils.tools import TFData, read_ngram, ngram2kernel, CE_loss, decode, monitor, stamps2indices, frames_constrain_loss, evaluate
from models.EODM import P_Ngram, EODM_loss
from models.GAN import PhoneClassifier


def train():
    # load external LM
    with tf.device("/cpu:0"):
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
        dataset_train_supervise = ASR_align_DataSet(
            trans_file=args.dirs.train_supervise.trans,
            align_file=args.dirs.train_supervise.align,
            uttid2wav=args.dirs.train.wav_scp,
            feat_len_file=args.dirs.train.feat_len,
            args=args,
            _shuffle=False,
            transform=True)
        feature_train = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args).read()
        feature_dev = TFData(dataset=dataset_dev,
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read()
        supervise_uttids, supervise_x = next(iter(feature_train.take(args.num_supervised).\
            padded_batch(args.num_supervised, ((), [None, args.dim_input]))))
        supervise_aligns = dataset_train_supervise.get_attrs('align', supervise_uttids.numpy())

        iter_feature_train = iter(feature_train.cache().repeat().shuffle(500).padded_batch(args.batch_size,
                ((), [None, args.dim_input])).prefetch(buffer_size=5))
        feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    # get dataset ngram
    ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')
    kernel, py = ngram2kernel(ngram_py, args)

    # create model paremeters
    G = PhoneClassifier(args)
    compute_p_ngram = P_Ngram(kernel, args)
    G.summary()
    compute_p_ngram.summary()

    # build optimizer
    if args.opti.type == 'adam':
        optimizer = tf.keras.optimizers.Adam(args.opti.lr, beta_1=0.5, beta_2=0.9)
    elif args.opti.type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=args.opti.lr, momentum=0.9, decay=0.98)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(G=G, optimizer=optimizer)
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

        uttids, x = next(iter_feature_train)
        stamps = dataset_train.get_attrs('stamps', uttids.numpy())

        loss_EODM, loss_fs = train_step(x, stamps, py, G, compute_p_ngram, optimizer, args.lambda_fs)
        # loss_EODM = loss_fs = 0
        loss_supervise = train_G_supervised(supervise_x, supervise_aligns, G, optimizer, args.dim_output, args.lambda_supervision)

        num_processed += len(x)
        progress = num_processed / args.data.train_size
        if step % 10 == 0:
            print('EODM loss: {:.2f}\tloss_fs: {:.3f} * {}\tloss_supervise: {:.3f} * {}\tbatch: {} time: {:.2f} s {:.3f}% step: {}'.format(
                   loss_EODM, loss_fs, args.lambda_fs, loss_supervise, args.lambda_supervision, x.shape, time()-start, progress*100.0, step))
            with writer.as_default():
                tf.summary.scalar("costs/loss_EODM", loss_EODM, step=step)
                tf.summary.scalar("costs/loss_fs", loss_fs, step=step)
                tf.summary.scalar("costs/loss_supervise", loss_supervise, step=step)

        if step % args.dev_step == 0:
            fer, cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, G)
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


def Decode(save_file):
    dataset = ASR_align_DataSet(
        trans_file=args.dirs.train.trans,
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

    feature_dev = TFData(dataset=dataset_dev,
                    dir_save=args.dirs.dev.tfdata,
                    args=args).read()
    feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    G = PhoneClassifier(args)
    G.summary()

    optimizer_G = tf.keras.optimizers.Adam(1e-4)
    ckpt = tf.train.Checkpoint(G=G, optimizer_G=optimizer_G)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('checkpoint {} restored!!'.format(ckpt_manager.latest_checkpoint))
    fer, cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, G)
    decode(dataset, G, args.idx2token, 'output/'+save_file)


def train_step(x, stamps, py, model, compute_p_ngram, optimizer, lambda_fs):
    indices = stamps2indices(stamps)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.variables)
        logits = model(x, training=True)
        _logits = tf.gather_nd(logits, indices)
        loss_EODM = EODM_loss(_logits, stamps>0, compute_p_ngram, args.data.top_k, py)
        # loss_EODM = 0
        loss_fs = frames_constrain_loss(logits, stamps) if lambda_fs > 0 else 0
        loss = loss_EODM + lambda_fs * loss_fs
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_EODM, loss_fs


@tf.function
def train_G_supervised(x, aligns, G, optimizer, dim_output, lambda_supervision):
    with tf.GradientTape() as tape_G:
        logits = G(x, training=True)
        ce_loss = CE_loss(logits, aligns, dim_output, confidence=0.9)

    gradients_G = tape_G.gradient(ce_loss, G.trainable_variables)
    optimizer.apply_gradients(zip(gradients_G, G.trainable_variables))

    return ce_loss


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
    gpus = tf.config.list_physical_devices('GPU')
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
        # lm_assistant(Model, Model_LM)

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
