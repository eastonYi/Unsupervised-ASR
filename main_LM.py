#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import tensorflow as tf
import numpy as np

from utils.arguments import args
# from utils.dataset import PTB_LMDataSet as LMDataSet
from utils.dataset import LMDataSet, TextDataSet
from utils.tools import build_optimizer, warmup_exponential_decay, compute_ppl
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train(Model):
    # create dataset and dataloader
    dataset_train = TextDataSet(
        list_files=[args.dirs.train.data],
        args=args,
        _shuffle=True)
    dataset_dev = TextDataSet(
        list_files=[args.dirs.dev.data],
        args=args,
        _shuffle=False)

    args.data.train_size = len(dataset_train)
    args.data.dev_size = len(dataset_dev)

    tfdata_train = tf.data.Dataset.from_generator(
        dataset_train, (tf.int32), (tf.TensorShape([None])))
    tfdata_dev = tf.data.Dataset.from_generator(
        dataset_dev, (tf.int32), (tf.TensorShape([None])))

    tfdata_train = tfdata_train.cache().repeat().shuffle(500).padded_batch(args.batch_size, ([None])).prefetch(buffer_size=5)
    tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None]))

    # build optimizer
    warmup = warmup_exponential_decay(
        warmup_steps=args.opti.warmup_steps,
        peak=args.opti.peak,
        decay_steps=args.opti.decay_steps)
    optimizer = tf.keras.optimizers.Adam(
        warmup,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9)

    # create model paremeters
    model = Model(args)
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
        loss = train_step(x, y, model, optimizer)

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


# @tf.function
def train_step(x, labels, model, optimizer):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        mask = tf.cast(labels > 0, dtype=tf.float32)
        loss *= mask
        loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def evaluation(tfdata_dev, model):
    start_time = time()
    num_processed = 0
    progress = 0
    loss_sum = 0
    num_tokens = 0
    for batch in tfdata_dev:
        x, y = batch
        logits = model(x, training=False)
        loss, num_batch_tokens = compute_ppl(logits, y)
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


    from models.LSTM import Embed_LSTM_Model as Model

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        assert len(gpus) > 0, "Not enough GPU hardware devices available"
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
        print('enter the TRAINING phrase')
        train(Model)

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
