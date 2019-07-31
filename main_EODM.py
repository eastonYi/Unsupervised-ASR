#!/usr/bin/env python
from datetime import datetime
from time import time
import os
import tensorflow as tf

from utils.dataset import ASR_align_DataSet
from utils.arguments import args
from utils.tools import TFData, read_ngram, ngram2kernel, CE_loss, evaluation, decode, monitor, aligns2indices, frames_constrain_loss
from models.EODM import P_Ngram, EODM_loss


def train(Model):
    # load external LM
    with tf.device("/cpu:0"):
        dataset_dev = ASR_align_DataSet(
            file=[args.dirs.dev.data],
            args=args,
            _shuffle=False,
            transform=True)
        tfdata_train = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)

        x_0, y_0, aligns_0 = next(iter(tfdata_train.take(args.num_supervised).\
            padded_batch(args.num_supervised, ([None, args.dim_input], [None], [None]))))
        iter_train = iter(tfdata_train.cache().repeat().shuffle(3000).\
            padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=3))
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

    # get dataset ngram
    ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')
    kernel, py = ngram2kernel(ngram_py, args)

    # create model paremeters
    model = Model(args)
    compute_p_ngram = P_Ngram(kernel, args)
    model.summary()
    compute_p_ngram.summary()

    # build optimizer
    if args.opti.type == 'adam':
        optimizer = tf.keras.optimizers.Adam(args.opti.lr, beta_1=0.5, beta_2=0.9)
        # optimizer = tf.keras.optimizers.Adam(args.opti.lr*0.1, beta_1=0.5, beta_2=0.9)
    elif args.opti.type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(lr=args.opti.lr, momentum=0.9, decay=0.98)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(model=model, optimizer = optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=5)
    step = 0

    # if a checkpoint exists, restore the latest checkpoint.
    if args.dirs.checkpoint:
        _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
        ckpt.restore(_ckpt_manager.latest_checkpoint)
        print ('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))
        step = int(_ckpt_manager.latest_checkpoint.split('-')[-1])

    start_time = datetime.now()
    num_processed = 0
    progress = 0

    # step = 1600
    while step < 99999999:
        start = time()

        x, _, aligns = next(iter_train)
        loss_EODM, loss_fs = train_step(x, aligns, py, model, compute_p_ngram, optimizer, args.lambda_fs)
        loss_supervise = train_G_supervised(x_0, y_0, model, optimizer, args.dim_output)

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
            fer, cer = evaluation(tfdata_dev, args.data.dev_size, model)
            with writer.as_default():
                tf.summary.scalar("performance/fer", fer, step=step)
                tf.summary.scalar("performance/cer", cer, step=step)
        if step % args.decode_step == 0:
            monitor(dataset_dev[0], model)
        if step % args.save_step == 0:
            save_path = ckpt_manager.save(step)
            print('save model {}'.format(save_path))

        step += 1

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def train_step(x, aligns, py, model, compute_p_ngram, optimizer, lambda_fs):
    indices = aligns2indices(aligns)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.variables)
        logits = model(x, training=True)
        _logits = tf.gather_nd(logits, indices)
        loss_EODM = EODM_loss(_logits, aligns>0, compute_p_ngram, args.data.top_k, py)
        # loss_EODM = 0
        loss_fs = frames_constrain_loss(logits, aligns) if lambda_fs > 0 else 0
        loss = loss_EODM + lambda_fs * loss_fs
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_EODM, loss_fs


@tf.function
def train_G_supervised(x, labels, G, optimizer, dim_output):
    with tf.GradientTape() as tape_G:
        logits = G(x, training=True)
        ce_loss = CE_loss(logits, labels, dim_output, confidence=0.9)
        gen_loss = ce_loss

    gradients_G = tape_G.gradient(gen_loss, G.trainable_variables)
    optimizer.apply_gradients(zip(gradients_G, G.trainable_variables))

    return gen_loss


def train_step_with_supervision(x, aligns, x_0, y_0, py, model, compute_p_ngram, optimizer, dim_output, lambda_fs, lambda_supervision):
    indices = aligns2indices(aligns)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(model.variables)
        logits = model(x, training=True)
        _logits = tf.gather_nd(logits, indices)
        loss_EODM = EODM_loss(_logits, aligns>0, compute_p_ngram, args.data.top_k, py)
        # loss_EODM = 0

        loss_fs = frames_constrain_loss(logits, aligns) if lambda_fs > 0 else 0

        logits_0 = model(x_0, training=True)
        loss_supervise = CE_loss(logits_0, y_0, dim_output, confidence=0.9)

        loss = loss_EODM + lambda_fs * loss_fs + lambda_supervision * loss_supervise
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss_EODM, loss_fs, loss_supervise


# def lm_assistant(Model, Model_LM):
#     # load external LM
#     with tf.device("/cpu:0"):
#         tfdata_train = TFData(dataset=None,
#                         dataAttr=['feature', 'label', 'align'],
#                         dir_save=args.dirs.train.tfdata,
#                         args=args).read(_shuffle=True)
#         tfdata_dev = TFData(dataset=None,
#                         dataAttr=['feature', 'label', 'align'],
#                         dir_save=args.dirs.dev.tfdata,
#                         args=args).read(_shuffle=False)
#         tfdata_train = tfdata_train.repeat().shuffle(100).\
#             padded_batch(args.batch_size, ([None, args.dim_input], [None], [None])).prefetch(buffer_size=1000)
#         tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))
#
#     # # get dataset ngram
#     # ngram_py, total_num = read_ngram(args.data.k, args.dirs.ngram, args.token2idx, type='list')
#
#     # build optimizer
#     warmup = warmup_exponential_decay(
#         warmup_steps=args.opti.warmup_steps,
#         peak=args.opti.peak,
#         decay_steps=args.opti.decay_steps)
#     optimizer = build_optimizer(warmup, args.opti.type)
#
#     # create model paremeters
#     model_lm = Model_LM(args.args_lm, optimizer=tf.keras.optimizers.Adam(), name='lstm')
#     model = Model(args, optimizer=optimizer, name='fc')
#     model.summary()
#     model_lm.summary()
#
#     # save & reload
#     ckpt_lm = tf.train.Checkpoint(model=model_lm, optimizer=model_lm.optimizer)
#     latest_checkpoint = tf.train.CheckpointManager(ckpt_lm, args.args_lm.dirs.restore, max_to_keep=1).latest_checkpoint
#     assert latest_checkpoint
#     ckpt_lm.restore(latest_checkpoint)
#     print('LM {} restored!!'.format(latest_checkpoint))
#     lm_dev(model_lm)
#
#     ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
#     ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
#     if args.dirs.restore:
#         latest_checkpoint = tf.train.CheckpointManager(ckpt, args.dirs.restore, max_to_keep=1).latest_checkpoint
#         ckpt.restore(latest_checkpoint)
#         print('{} restored!!'.format(latest_checkpoint))
#
#     start_time = datetime.now()
#     get_data_time = 0
#     num_processed = 0
#     progress = 0
#
#     for step, batch in enumerate(tfdata_train):
#         x, y, aligns = batch
#         aligns_sampled = sampleFrames(aligns)
#         # ngram_sampled = sample(ngram_py, args.data.top_k)
#         # kernel, py = ngram2kernel(ngram_sampled, args)
#         run_model_time = time()
#         # build compute model on-the-fly
#         with tf.GradientTape(watch_accessed_variables=False) as tape:
#             tape.watch(model.variables)
#             logits = model(x, training=True)
#             loss = model.align_loss(logits, y)
#             # loss_EODM = model.EODM_loss(logits, aligns_sampled, kernel, py)
#             loss_LM = model_lm.compute_fitting_loss(logits, aligns_sampled)
#             # loss_LM = 0
#             loss_EODM = loss
#             # loss = loss_EODM + loss_LM
#         gradients = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#         acc = model.align_accuracy(logits, y)
#
#         num_processed += len(x)
#         get_data_time = run_model_time - get_data_time
#         run_model_time = time() - run_model_time
#
#         progress = num_processed / args.data.train_size
#         if step % 10 == 0:
#             print('EODM loss: {:.2f}\tlm loss: {:.2f}\t FER: {:.3f}\t batch: {} lr:{:.6f} time: {:.2f}|{:.2f} s {:.3f}% step: {}'.format(
#                    loss_EODM, loss_LM, 1-acc, x.shape, warmup(step*1.0).numpy(), get_data_time, run_model_time, progress*100.0, step))
#         get_data_time = time()
#
#         if step % args.dev_step == 0:
#             evaluation(tfdata_dev, model)
#         if step % args.decode_step == 0:
#             decode(model)
#         if step % args.fs_step == 0:
#             fs_constrain(batch, model, optimizer)
#         # if 1-acc > 0.80:
#         #     head_tail_constrain(batch, model, optimizer)
#         if step % args.save_step == 0:
#             ckpt_manager.save()
#
#     print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))
#
#
# def lm_dev(model):
#     # evaluate
#     dataset_dev = LMDataSet(
#         list_files=[args.dirs.dev.data],
#         args=args,
#         _shuffle=False)
#     tfdata_lm = tf.data.Dataset.from_generator(
#         dataset_dev,
#         (tf.int32, tf.int32),
#         (tf.TensorShape([None]), tf.TensorShape([None]))).\
#         padded_batch(args.batch_size, ([None], [None]))
#     start_time = time()
#     num_processed = 0
#     loss_sum = 0
#     num_tokens = 0
#     for batch in tfdata_lm:
#         x, y = batch
#         logits = model(x, training=False)
#         loss, num_batch_tokens = model.compute_ppl(logits, y)
#         loss_sum += loss
#         num_tokens += num_batch_tokens
#         num_processed += len(x)
#     ppl = tf.exp(loss_sum/num_tokens)
#     print('lm dev ppl: {:.3f}\t {:.2f}min {} / {}'.format(
#             ppl, (time()-start_time)/60, num_processed, args.data.dev_size))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    print('CUDA_VISIBLE_DEVICES: ', args.gpus)

    from models.GAN import PhoneClassifier as Model

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
        os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        assert len(gpus) > 0, "Not enough GPU hardware devices available"
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
        print('enter the TRAINING phrase')
        train(Model)
        # lm_assistant(Model, Model_LM)

        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
