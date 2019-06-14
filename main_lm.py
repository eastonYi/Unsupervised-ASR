#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
tf.config.gpu.set_per_process_memory_growth(True)
import numpy as np

from utils.arguments import args
# from utils.dataset import PTB_LMDataSet as LMDataSet
from utils.dataset import LMDataSet
from utils.tools import build_optimizer, warmup_exponential_decay


def train(Model):
    # create dataset and dataloader
    dataset_train = LMDataSet(
        list_files=[args.dirs.train.data],
        args=args,
        _shuffle=True)
    dataset_dev = LMDataSet(
        list_files=[args.dirs.dev.data],
        args=args,
        _shuffle=False)

    tfdata_train = tf.data.Dataset.from_generator(
        dataset_train, (tf.int32, tf.int32), (tf.TensorShape([None]), tf.TensorShape([None])))
    tfdata_dev = tf.data.Dataset.from_generator(
        dataset_dev, (tf.int32, tf.int32), (tf.TensorShape([None]), tf.TensorShape([None])))

    tfdata_train = tfdata_train.cache().repeat().shuffle(500).padded_batch(args.batch_size, ([None], [None])).prefetch(buffer_size=5)
    tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None], [None]))

    # for i in tfdata_train:
    #     print(i)
    #     import pdb; pdb.set_trace()

    # build optimizer
    warmup = warmup_exponential_decay(
        warmup_steps=args.opti.warmup_steps,
        peak=args.opti.peak,
        decay_steps=args.opti.decay_steps)
    optimizer = build_optimizer(warmup, args.opti.type)

    # create model paremeters
    model = Model(args, optimizer=optimizer, name='lstm')
    model.summary()

    # save & reload
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    if args.dirs.restore:
        latest_checkpoint = tf.train.CheckpointManager(ckpt, args.dirs.restore, max_to_keep=1).latest_checkpoint
        ckpt.restore(latest_checkpoint)
        print('{} restored!!'.format(latest_checkpoint))

    start_time = datetime.now()
    get_data_time = 0
    num_processed = 0
    progress = 0

    for global_step, batch in enumerate(tfdata_train):
        x, y = batch

        run_model_time = time()
        # build compute model on-the-fly
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = model.compute_loss(logits, y)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        num_processed += len(x)
        get_data_time = run_model_time - get_data_time
        run_model_time = time() - run_model_time

        progress = num_processed / args.data.train_size
        if global_step % 10 == 0:
            print('loss: {:.5f}\t batch: {} lr:{:.6f} time: {:.2f}|{:.2f} s {:.3f}% step: {}'.format(
                   loss, x.shape, warmup(global_step*1.0).numpy(), get_data_time, run_model_time, progress*100.0, global_step))
        get_data_time = time()

        if global_step % args.dev_step == 0:
            evaluation(tfdata_dev, model)
        # if global_step % args.decode_step == 0:
        #     decode(model)
        if global_step % args.save_step == 0:
            ckpt_manager.save()

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def evaluation(tfdata_dev, model):
    start_time = time()
    num_processed = 0
    progress = 0
    loss_sum = 0
    num_tokens = 0
    for batch in tfdata_dev:
        x, y = batch
        logits = model(x, training=False)
        loss, num_batch_tokens = model.compute_ppl(logits, y)
        loss_sum += loss
        num_tokens += num_batch_tokens
        num_processed += len(x)
        progress = num_processed / args.data.dev_size

    ppl = tf.exp(loss_sum/num_tokens)
    print('dev ppl: {:.3f}\t {:.2f}min {} / {}'.format(
            ppl, (time()-start_time)/60, num_processed, args.data.dev_size))


def decode(model):
    dataset = LMDataSet(
        file=[args.dirs.dev.data],
        args=args,
        _shuffle=False,
        transform=True)
    sample = dataset[0]
    x = np.array([sample['feature']], dtype=np.float32)
    logits = model(x, training=False)
    predits = model.get_predicts(logits)
    print('predits: \n', predits.numpy()[0])
    print('label: \n', sample['label'])
    print('align: ', sample['align'])


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default='0')
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)

    if args.model.structure == 'lstm':
        from utils.model import LSTM_Model as Model

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        print('enter the TRAINING phrase')
        train(Model)

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
