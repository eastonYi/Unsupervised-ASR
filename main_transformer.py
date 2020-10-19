#!/usr/bin/env python

from datetime import datetime
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import numpy as np

from utils.arguments import args
from utils.dataset import ASR_align_ArkDataSet
from utils.tools import TFData, batch_cer, CustomSchedule, get_tensor_len

from models.transformer import Transformer

ITERS = 200000 # How many iterations to train for
tf.random.set_seed(args.seed)
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def Train():
    with tf.device("/cpu:0"):
        dataset_train = ASR_align_ArkDataSet(
            scp_file=args.dirs.train.scp,
            trans_file=args.dirs.train.trans,
            align_file=None,
            feat_len_file=None,
            args=args,
            _shuffle=False,
            transform=True)
        dataset_dev = ASR_align_ArkDataSet(
            scp_file=args.dirs.dev.scp,
            trans_file=args.dirs.dev.trans,
            align_file=None,
            feat_len_file=None,
            args=args,
            _shuffle=False,
            transform=True)
        # wav data
        feature_train = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=True, transform=True)
        feature_dev = TFData(dataset=dataset_dev,
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False, transform=True)
        bucket = tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda uttid, x: tf.shape(x)[0],
            bucket_boundaries=args.list_bucket_boundaries,
            bucket_batch_sizes=args.list_batch_size,
            padded_shapes=((), [None, args.dim_input]))
        iter_feature_train = iter(feature_train.repeat().shuffle(10).apply(bucket).prefetch(buffer_size=5))
        feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    # create model paremeters
    model, model_infer = Transformer(args)
    model.summary()
    model_infer.summary()

    learning_rate = CustomSchedule(args.model.G.d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

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
        trans_sos = dataset_train.get_attrs('trans_sos', uttids.numpy())
        trans_eos = dataset_train.get_attrs('trans_eos', uttids.numpy())
        loss_supervise = train_step(x, trans_sos, trans_eos, model, optimizer, args.dim_output)

        num_processed += len(x)
        progress = num_processed / args.data.train_size

        if step % 10 == 0:
            print('loss: {:.3f}\tbatch: {}\tused: {:.3f}\t {:.3f}% step: {}'.format(
                   loss_supervise, x.shape, time()-start, progress*100, step))
            with writer.as_default():
                tf.summary.scalar("costs/loss_supervise", loss_supervise, step=step)
        if step % args.decode_step == 0:
            monitor(dataset_dev[0], model_infer)
        if step % args.dev_step == 0 and loss_supervise < 1.0:
            cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, model_infer)
            with writer.as_default():
                tf.summary.scalar("performance/cer", cer, step=step)
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
                    args=args).read(_shuffle=False, transform=True)
    feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    _, model_infer = Transformer(args)
    # model.summary()
    model_infer.summary()

    optimizer = tf.keras.optimizers.Adam(1e-4)
    ckpt = tf.train.Checkpoint(model=model_infer, optimizer=optimizer)

    _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
    ckpt.restore(_ckpt_manager.latest_checkpoint)
    print ('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))
    cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, model_infer)
    print('PER:{:.3f}'.format(cer))


def monitor(sample, model):
    x = np.array([sample['feature']], dtype=np.float32)
    preds = forward(x, model, args.token2idx['<sos>'], args.token2idx['<eos>'], args.max_label_len)

    print('predicts: \n', preds.numpy()[0])
    print('trans: \n', sample['trans'])


def evaluate(feature, dataset, dev_size, model):
    num_processed = 0
    total_cer_dist = 0
    total_cer_len = 0
    total_res_len = 0
    for batch in feature:
        uttids, x = batch
        preds = forward(x, model, args.token2idx['<sos>'], args.token2idx['<eos>'], args.max_label_len)
        trans = dataset.get_attrs('trans', uttids.numpy())
        batch_cer_dist, batch_cer_len, batch_res_len = batch_cer(preds.numpy(), trans)
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len
        total_res_len += batch_res_len

        num_processed += len(x)
        print('infering {} / {} ...'.format(num_processed, dev_size))

    cer = total_cer_dist/total_cer_len
    print('dev PER: {:.3f}\t{} / {}'.format(cer, num_processed, dev_size))

    return cer


@tf.function(experimental_relax_shapes=True)
def train_step(x, trans_sos, trans_eos, model, optimizer, vocab_size):
    with tf.GradientTape() as tape:
        logits = model([x, trans_sos], training=True)
        loss = CE_loss(logits, trans_eos, vocab_size)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def CE_loss(logits, labels, vocab_size, confidence=0.9):

    low_confidence = (1.0 - confidence) / tf.cast(vocab_size-1, tf.float32)
    normalizing = -(confidence*tf.math.log(confidence) +
        tf.cast(vocab_size-1, tf.float32) * low_confidence * tf.math.log(low_confidence + 1e-20))
    soft_targets = tf.one_hot(
        tf.cast(labels, tf.int32),
        depth=vocab_size,
        on_value=confidence,
        off_value=low_confidence)

    xentropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=soft_targets)
    loss = xentropy - normalizing

    len_logits = get_tensor_len(logits)
    gen_loss = tf.sequence_mask(len_logits, dtype=tf.float32) * loss
    loss = tf.reduce_mean(tf.reduce_sum(gen_loss, -1) / tf.cast(len_logits, tf.float32))

    return loss


# @tf.function(experimental_relax_shapes=True)
def forward(x, model, sos_token, end_token, max_len):
    batch_size = tf.shape(x)[0]
    prevs = tf.ones([batch_size, 1], tf.int32) * sos_token
    cache = tf.zeros([batch_size, 0, args.model.G.num_layers, args.model.G.d_model], tf.float32)
    finished = tf.zeros([batch_size], dtype=tf.bool)

    for i in tf.range(max_len):
        # preds.shape == (batch_size, seq_len, vocab_size)
        preds, cache = model([x, prevs, cache], training=False)

        # 从 seq_len 维度选择最后一个词
        pred = preds[: ,-1:, :]  # (batch_size, 1, vocab_size)
        pred = tf.cast(tf.argmax(pred, axis=-1), tf.int32)
        # import pdb; pdb.set_trace()
        # 如果 predicted_id 等于结束标记，就返回结果
        has_eos = tf.equal(pred, end_token)
        finished = tf.logical_or(finished, has_eos)

        # 连接 predicted_id 与输出，作为解码器的输入传递到解码器。
        prevs = tf.concat([prevs, pred], axis=-1)
        # print(pred.numpy(), prevs.numpy())

        if tf.equal(tf.reduce_mean(tf.cast(finished, dtype=tf.int32)), 1):
            return prevs

    return prevs


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
