#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import tensorflow as tf
# tf.debugging.set_log_device_placement(True)
tf.config.gpu.set_per_process_memory_growth(True)
import numpy as np
from random import sample
import threading
from queue import Queue

from eastonCode.tfTools.tfData import TFData

from utils.arguments import args
from utils.dataset import ASR_align_DataSet
from utils.tools import build_optimizer, sampleFrames, read_ngram, batch_cer, pertubated_model_weights, ngram2kernel


def train(Model):
    # create dataset and dataloader
    with tf.device("/cpu:0"):
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_monitor = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_monitor = tfdata_monitor.cache().repeat().shuffle(500).padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=5)
        tfdata_iter = iter(tfdata_monitor)
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

    # get dataset ngram
    ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')

    # create model paremeters
    opti = tf.keras.optimizers.SGD(0.5)
    model = Model(args, optimizer=opti, name='fc')
    model.summary()

    # save & reload
    ckpt = tf.train.Checkpoint(model=model, optimizer=opti)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    if args.dirs.restore:
        latest_checkpoint = tf.train.CheckpointManager(ckpt, args.dirs.restore, max_to_keep=1).latest_checkpoint
        ckpt.restore(latest_checkpoint)
        print('{} restored!!'.format(latest_checkpoint))

    best_rewards = -999
    start_time = datetime.now()
    fer = 1.0
    seed = 999
    step = 0

    while 1:
        if fer < 0.69:
            break
        elif fer > 0.76 or step > 69:
            print('{}-th reset, pre FER: {:.3f}'.format(seed, fer))
            seed += 1
            step = 0
            tf.random.set_seed(seed)
            model = Model(args, optimizer=opti, name='fc')
            head_tail_constrain(next(tfdata_iter), model, opti)
            fer = mini_eva(tfdata_dev, model)
            ngram_sampled = sample(ngram_py, args.data.top_k)
            kernel, py = ngram2kernel(ngram_sampled, args)
        else:
            step += 1
            loss = train_step(model, tfdata_iter, kernel, py)
            fer = mini_eva(tfdata_dev, model)
            print('\tloss: {:.3f}\tFER: {:.3f}'.format(loss, fer))

    for global_step in range(99999):
        run_model_time = time()
        loss = train_step(model, tfdata_iter, kernel, py)

        used_time = time()-run_model_time
        if global_step % 1 == 0:
            print('full training loss: {:.3f}, spend {:.2f}s step {}'.format(loss, used_time, global_step))

        if global_step % args.dev_step == 0:
            evaluation(tfdata_dev, model)
        if global_step % args.decode_step == 0:
            decode(model)
        if global_step % args.fs_step == 0:
            fs_constrain(next(tfdata_iter), model, opti)
        if global_step % args.save_step == 0:
            ckpt_manager.save()

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


aligns_sampled = None
kernel = None
queue_input = Queue(maxsize=999)
queue_output = Queue(maxsize=999)
def train_mul(Model):

    # create dataset and dataloader
    with tf.device("/cpu:0"):
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_monitor = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_monitor = tfdata_monitor.cache().repeat().shuffle(500).padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=5)
        tfdata_iter = iter(tfdata_monitor)
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

    # get dataset ngram
    ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')

    # create model paremeters
    opti = tf.keras.optimizers.SGD(0.5)
    model = Model(args, optimizer=opti, name='fc')
    model.summary()

    def thread_session(thread_id, queue_input, queue_output):
        global kernel
        gpu = args.list_gpus[thread_id]
        with tf.device(gpu):
            opti_adam = build_optimizer(args, type='adam')
            model = Model(args, optimizer=opti_adam, name='fc'+str(thread_id))
            print('thread_{} is waiting to run on {}....'.format(thread_id, gpu))
            while True:
                # s = time()
                id, weights, x, aligns_sampled = queue_input.get()
                model.set_weights(weights)
                # t = time()
                logits = model(x, training=False)
                pz, K = model.EODM(logits, aligns_sampled, kernel)
                queue_output.put((id, pz, K))
                # print('{} {:.3f}|{:.3f}s'.format(gpu, t-s, time()-s))

    for id in range(4):
        thread = threading.Thread(
            target=thread_session,
            args=(id, queue_input, queue_output))
        thread.daemon = True
        thread.start()

    # save & reload
    ckpt = tf.train.Checkpoint(model=model, optimizer=opti)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    if args.dirs.restore:
        latest_checkpoint = tf.train.CheckpointManager(ckpt, args.dirs.restore, max_to_keep=1).latest_checkpoint
        ckpt.restore(latest_checkpoint)
        print('{} restored!!'.format(latest_checkpoint))

    best_rewards = -999
    start_time = datetime.now()
    fer = 1.0
    seed = 999
    step = 0
    global aligns_sampled, kernel

    while 1:
        if fer < 0.72:
            break
        elif fer > 0.80 or step > 69:
            print('{}-th reset, pre FER: {:.3f}'.format(seed, fer))
            seed += 1
            step = 0
            tf.random.set_seed(seed)
            model = Model(args, optimizer=opti, name='fc')
            head_tail_constrain(next(tfdata_iter), model, opti)
            fer = mini_eva(tfdata_dev, model)
            ngram_sampled = sample(ngram_py, args.data.top_k)
            kernel, py = ngram2kernel(ngram_sampled, args)
        else:
            step += 1
            loss = train_step(model, tfdata_iter, py)
            fer = mini_eva(tfdata_dev, model)
            print('\tloss: {:.3f}\tFER: {:.3f}'.format(loss, fer))

    for global_step in range(99999):
        run_model_time = time()
        loss = train_step(model, tfdata_iter, py)

        used_time = time()-run_model_time
        if global_step % 1 == 0:
            print('full training loss: {:.3f}, spend {:.2f}s step {}'.format(loss, used_time, global_step))

        if global_step % args.dev_step == 0:
            evaluation(tfdata_dev, model)
        if global_step % args.decode_step == 0:
            decode(model)
        if global_step % args.fs_step == 0:
            fs_constrain(next(tfdata_iter), model, opti)
        if global_step % args.save_step == 0:
            ckpt_manager.save()

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def evaluation(tfdata_dev, model):
    list_loss = []
    list_acc = []

    start_time = time()
    num_processed = 0
    progress = 0
    total_cer_dist = 0
    total_cer_len = 0
    for batch in tfdata_dev:
        x, y, aligns = batch
        logits = model(x, training=False)
        loss = model.align_loss(logits, y, args.dim_output, confidence=0.9)
        acc = model.align_accuracy(logits, y)
        list_loss.append(loss)
        list_acc.append(acc)
        preds = model.get_predicts(logits)
        batch_cer_dist, batch_cer_len = batch_cer(preds.numpy(), y)
        total_cer_dist += batch_cer_dist
        total_cer_len += batch_cer_len
        num_processed += len(x)
        progress = num_processed / args.data.dev_size

    cer = total_cer_dist/total_cer_len
    print('dev loss: {:.3f}\t dev FER: {:.3f}\t dev PER: {:.3f}\t {:.2f}min {} / {}'.format(
            np.mean(list_loss), 1-np.mean(list_acc), cer, (time()-start_time)/60, num_processed, args.data.dev_size))

    return 1-np.mean(list_acc)

def mini_eva(tfdata_dev, model):
    list_acc = []

    for batch in tfdata_dev:
        x, y, aligns = batch
        logits = model(x, training=False)
        acc = model.align_accuracy(logits, y)
        list_acc.append(acc)

    return 1-np.mean(list_acc)


def decode(model):
    dataset = ASR_align_DataSet(
        file=[args.dirs.dev.data],
        args=args,
        _shuffle=False,
        transform=True)
    sample = dataset[0]
    x = np.array([sample['feature']], dtype=np.float32)
    logits = model(x, training=False)
    predits = model.get_predicts(logits)
    print('predict: \n', predits.numpy()[0])
    print('label: \n', sample['label'])
    print('align: ', sample['align'])


def fs_constrain(batch, model, optimizer):
    mini_size = 100
    for i in range(len(batch)//mini_size):
        x, y, aligns = [m[i*mini_size, (i+1)*mini_size] for m in batch]
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss = model.frames_constrain_loss(logits, aligns)
        print('fs loss: ', loss.numpy())
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def head_tail_constrain(batch, model, optimizer):
    # mini_size = 1000
    # for i in range(len(batch[0])//mini_size):
    #     x, y, _ = [m[i*mini_size: (i+1)*mini_size] for m in batch]
    x, y, _ = batch
    y = tf.ones_like(y) * y[0][0]
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = model.align_loss(logits, y, args.dim_output, confidence=0.9)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train_step(model, tfdata_iter, py):
    lr = args.opti.lr
    sigma = args.opti.sigma
    size_population = args.opti.population

    weights_model = model.get_weights()

    # explore
    population = []
    weights_population = []
    for i in range(size_population):
        pertubation = [np.random.randn(*w.shape) for w in weights_model]
        population.append(pertubation)
        weights_pert = pertubated_model_weights(weights_model, pertubation, sigma=sigma)
        weights_population.append(weights_pert)

    # analyse feedbacks
    rewards = get_rewards(weights_population, model, tfdata_iter, py)
    _rewards = (rewards - np.mean(rewards)) / (np.std(rewards)+1e-11)

    # merge weights
    weights_new = []
    for index, w in enumerate(weights_model):
        A = tf.cast(tf.stack([p[index] for p in population], 0), 'float32')
        shape_out = A.shape[1:]
        _rewards = tf.reshape(_rewards, [1, size_population])
        A = tf.reshape(A, [size_population, -1])
        weights_new.append(w + lr/(size_population*sigma) * tf.reshape(tf.matmul(_rewards, A), shape_out))
    model.set_weights(weights_new)

    return tf.reduce_mean(-rewards)


def get_rewards(list_models, model, tfdata_iter, py):
    global queue_input, queue_output

    list_pz = [[] for _ in range(len(list_models))]
    list_K = [[] for _ in range(len(list_models))]

    # quary
    for _ in range(args.batch_multi):
        batch = next(tfdata_iter)
        x, _, aligns = batch
        aligns_sampled = sampleFrames(aligns)
        [queue_input.put((id, weights, x, aligns_sampled)) for (id, weights) in enumerate(list_models)]

    # collection
    for _ in range(args.batch_multi * len(list_models)):
        id, pz, K = queue_output.get()
        list_pz[id].append(pz)
        list_K[id].append(K)

    pz = tf.reduce_sum(list_pz, 1) / tf.reduce_sum(list_K, 1)
    rewards = tf.reduce_sum(py * tf.math.log(1e-15+pz), -1) # ngram loss

    return rewards


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)

    if param.mode == 'train':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print('enter the TRAINING phrase')
        # train(Model)
        train_mul(args.Model)

        # python main_es.py --gpu 1 -c configs/timit_es.yaml
