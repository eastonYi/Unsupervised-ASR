#!/usr/bin/env python

from datetime import datetime
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

from utils.arguments import args
from utils.dataset import ASR_align_ArkDataSet
from utils.tools import TFData, CustomSchedule, get_tensor_len

from models.transformer import Transformer_s1 as Transformer

ITERS = 200000 # How many iterations to train for
tf.random.set_seed(args.seed)


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

        feature_train = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=True, transform=True)
        bucket = tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda uttid, x: tf.shape(x)[0],
            bucket_boundaries=args.list_bucket_boundaries,
            bucket_batch_sizes=args.list_batch_size,
            padded_shapes=((), [None, args.dim_input]))
        iter_feature_train = iter(feature_train.repeat().shuffle(10).apply(bucket).prefetch(buffer_size=5))

    # create model paremeters
    model = Transformer(args)
    model.summary()

    # learning_rate = CustomSchedule(args.model.G.d_model)
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     0.0001,
    #     decay_steps=10000,
    #     decay_rate=0.5,
    #     staircase=True)
    optimizer = tf.keras.optimizers.Adam(0.000005, beta_1=0.5, beta_2=0.9,
                                         epsilon=1e-9)
    # optimizer = tf.keras.optimizers.SGD(0.1)

    ckpt = tf.train.Checkpoint(model=model)
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
    # uttids, x = next(iter_feature_train)
    # trans_sos = dataset_train.get_attrs('trans_sos', uttids.numpy())
    # trans_eos = dataset_train.get_attrs('trans_eos', uttids.numpy())
    while step < 99999999:
        start = time()
        uttids, x = next(iter_feature_train)
        trans_sos = dataset_train.get_attrs('trans_sos', uttids.numpy())
        trans_eos = dataset_train.get_attrs('trans_eos', uttids.numpy())
        loss_supervise = train_step(x, trans_sos, trans_eos, model, optimizer)

        num_processed += len(x)
        progress = num_processed / args.data.train_size

        if step % 20 == 0:
            print('loss: {:.3f}\tbatch: {}\tused: {:.3f}\t {:.3f}% step: {}'.format(
                   loss_supervise, x.shape, time()-start, progress*100, step))
        # if step % args.save_step == 0:
        #     save_path = ckpt_manager.save(step)
        #     print('save model {}'.format(save_path))

        step += 1

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


@tf.function(experimental_relax_shapes=True)
def train_step(x, trans_sos, trans_eos, model, optimizer):
    with tf.GradientTape() as tape:
        logits = model([x, trans_sos], training=True)
        loss = CE_loss(logits, trans_eos)
        # print('tf.argmax logits: ', tf.argmax(logits[0], -1).numpy())
        # print('trans_sos: ', trans_sos[0])
        # print('trans_eos: ', trans_eos[0])
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # if step > 10:
    #     import pdb; pdb.set_trace()
    # print(logits[0][:20])
    # print('preds:', tf.argmax(logits[0], -1))
    # print('trans_sos:', trans_sos)
    # print('trans_eos:', trans_eos[0])

    return loss

def CE_loss(logits, labels):

    _loss = tf.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=logits, from_logits=True)
    len_logits = get_tensor_len(logits)
    gen_loss = tf.sequence_mask(len_logits, dtype=tf.float32) * _loss
    # loss = tf.reduce_mean(tf.reduce_sum(gen_loss, -1) / tf.cast(len_logits, tf.float32))
    loss = tf.reduce_sum(gen_loss) / tf.cast(tf.reduce_sum(len_logits), tf.float32)

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
