#!/usr/bin/env python

from datetime import datetime
from time import time
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.tools import TFData
from utils.arguments import args
from utils.dataset import ASR_align_ArkDataSet, ASR_align_DataSet
from utils.tools import batch_cer, get_tensor_len, ctc_shrink

from models.GAN import PhoneClassifier as Model
# from models.encoders.encoders import conv_lstm as Model
# from models.encoders.conv_gru import Conv_GRU as Model
# from models.encoders.fullyConv import FullyConv as Model

ITERS = 200000 # How many iterations to train for
tf.random.set_seed(args.seed)


def Train():
    with tf.device("/cpu:0"):
        # dataset_train = ASR_align_ArkDataSet(
        #     scp_file=args.dirs.train.scp,
        #     trans_file=args.dirs.train.trans,
        #     align_file=None,
        #     feat_len_file=None,
        #     args=args,
        #     _shuffle=False,
        #     transform=False)
        # dataset_dev = ASR_align_ArkDataSet(
        #     scp_file=args.dirs.dev.scp,
        #     trans_file=args.dirs.dev.trans,
        #     align_file=None,
        #     feat_len_file=None,
        #     args=args,
        #     _shuffle=False,
        #     transform=False)
        dataset_train = ASR_align_DataSet(
            trans_file=args.dirs.train.trans,
            align_file=args.dirs.train.align,
            uttid2wav=args.dirs.train.wav_scp,
            feat_len_file=args.dirs.train.feat_len,
            args=args,
            _shuffle=False,
            transform=True)
        dataset_dev = ASR_align_DataSet(
            trans_file=args.dirs.dev.trans,
            uttid2wav=args.dirs.dev.wav_scp,
            align_file=args.dirs.dev.align,
            feat_len_file=args.dirs.dev.feat_len,
            args=args,
            _shuffle=False,
            transform=True)
        # wav data
        feature_train = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(transform=True)
        feature_dev = TFData(dataset=dataset_dev,
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(transform=True)
        bucket = tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda uttid, x: tf.shape(x)[0],
            bucket_boundaries=args.list_bucket_boundaries,
            bucket_batch_sizes=args.list_batch_size,
            padded_shapes=((), [None, args.dim_input]))
        iter_feature_train = iter(feature_train.repeat().shuffle(10).apply(bucket).prefetch(buffer_size=5))
        # iter_feature_train = iter(feature_train.repeat().shuffle(500).padded_batch(args.batch_size,
        #         ((), [None, args.dim_input])).prefetch(buffer_size=5))
        # feature_dev = feature_dev.apply(bucket).prefetch(buffer_size=5)
        feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    # create model paremeters
    model = Model(args)
    model.summary()
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     args.opti.lr,
    #     decay_steps=args.opti.decay_steps,
    #     decay_rate=0.5,
    #     staircase=True)
    optimizer = tf.keras.optimizers.Adam(args.opti.lr, beta_1=0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
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
    while step < 99999999:
        start = time()

        uttids, x = next(iter_feature_train)
        trans = dataset_train.get_attrs('trans', uttids.numpy())
        loss_supervise = train_CTC_supervised(x, trans, model, optimizer)

        num_processed += len(x)
        progress = num_processed / args.data.train_size

        if step % 10 == 0:
            print('loss: {:.3f}\tbatch: {}\tused: {:.3f}\t {:.3f}% step: {}'.format(
                   loss_supervise, x.shape, time()-start, progress*100, step))
            with writer.as_default():
                tf.summary.scalar("costs/loss_supervise", loss_supervise, step=step)
        if step % args.dev_step == 0:
            cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, model)
            with writer.as_default():
                tf.summary.scalar("performance/cer", cer, step=step)
        if step % args.decode_step == 0:
            monitor(dataset_dev[0], model)
        if step % args.save_step == 0:
            save_path = ckpt_manager.save(step)
            print('save model {}'.format(save_path))

        step += 1

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def Decode(save_file):
    dataset_dev = ASR_align_ArkDataSet(
        scp_file=args.dirs.dev.scp,
        trans_file=args.dirs.dev.trans,
        align_file=None,
        feat_len_file=None,
        args=args,
        _shuffle=False,
        transform=False)
    feature_dev = TFData(dataset=dataset_dev,
                    dir_save=args.dirs.dev.tfdata,
                    args=args).read()
    feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    model = Model(args)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(1e-4)
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)

    _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
    ckpt.restore(_ckpt_manager.latest_checkpoint)
    print ('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))
    cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, model)
    print('PER:{:.3f}'.format(cer))


def ctc_loss(logits, len_logits, labels, len_labels):
    """
    No valid path found: It is possible that no valid path is found if the
    activations for the targets are zero.
    """
    ctc_loss = tf.nn.ctc_loss(
        labels,
        logits,
        label_length=len_labels,
        logit_length=len_logits,
        logits_time_major=False,
        blank_index=-1)

    return ctc_loss


def ctc_decode(logits, len_logits, beam_size=1):
    logits_timeMajor = tf.transpose(logits, [1, 0, 2])

    if beam_size == 1:
        decoded_sparse = tf.cast(tf.nn.ctc_greedy_decoder(
            logits_timeMajor,
            len_logits,
            merge_repeated=True)[0][0], tf.int32)
    else:
        decoded_sparse = tf.cast(tf.nn.ctc_beam_search_decoder(
            logits_timeMajor,
            len_logits,
            beam_width=beam_size,
            merge_repeated=True)[0][0], tf.int32)

    preds = tf.sparse.to_dense(
        decoded_sparse,
        default_value=0,
        validate_indices=True)

    return preds


def get_logits_len(logits):

    return tf.reduce_sum(tf.cast((tf.reduce_max(tf.abs(logits), -1) > 0), tf.int32), -1)


def monitor(sample, model):
    x = np.array([sample['feature']], dtype=np.float32)
    logits = model(x)

    len_logits = get_logits_len(logits)
    preds = ctc_decode(logits, len_logits)
    # logits_shrunk = ctc_shrink(logits)
    # preds = tf.argmax(logits_shrunk, -1)
    print('predicts: \n', preds.numpy()[0])
    # print('align: \n', sample['align'])
    print('trans: \n', sample['trans'])


def evaluate(feature, dataset, dev_size, model):
    num_processed = 0
    total_cer_dist = 0
    total_cer_len = 0
    total_res_len = 0
    for batch in feature:
        uttids, x = batch
        # import pdb; pdb.set_trace()
        # preds = forward(x, model)
        logits = model(x)
        len_logits = get_logits_len(logits)
        preds = ctc_decode(logits, len_logits)
        # logits_shrunk = ctc_shrink(logits)
        # preds = tf.argmax(logits_shrunk, -1)
        trans = dataset.get_attrs('trans', uttids.numpy())
        batch_cer_dist, batch_cer_len, batch_res_len = batch_cer(preds.numpy(), trans)
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len
        total_res_len += batch_res_len
        num_processed += len(x)
        sys.stdout.write('\rinfering {} / {} ...'.format(num_processed, dev_size))
        sys.stdout.flush()

    cer = total_cer_dist/total_cer_len
    print('dev PER: {:.3f}\t{} / {}'.format(cer, num_processed, dev_size))

    return cer


# @tf.function(experimental_relax_shapes=True)
def train_CTC_supervised(x, labels, model, optimizer):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        len_logits = get_logits_len(logits)
        len_labels = tf.reduce_sum(tf.cast(labels > 0, tf.int32), -1)
        loss = ctc_loss(logits, len_logits, labels, len_labels)
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


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
