#!/usr/bin/env python

from datetime import datetime
from time import time
import os
import numpy as np
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL']='0'
import tensorflow as tf

from utils.arguments import args
from utils.dataset import ASR_align_DataSet, TextDataSet
from utils.tools import TFData, gradient_penalty, decode, ctc_shrink, ctc_loss, \
    get_tensor_len, batch_cer, pad_to, save_varibales, load_values

from models.GAN import PhoneClassifier, PhoneDiscriminator3
# from models.GAN import PhoneClassifier2 as PhoneClassifier

tf.random.set_seed(args.seed)

def train():
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
            uttid2wav=args.dirs.train_supervise.wav_scp,
            feat_len_file=args.dirs.train_supervise.feat_len,
            args=args,
            _shuffle=False,
            transform=True)
        feature_train_supervise = TFData(dataset=dataset_train_supervise,
                        dir_save=args.dirs.train_supervise.tfdata,
                        args=args).read()
        feature_train = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args).read()
        feature_dev = TFData(dataset=dataset_dev,
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read()

        iter_feature_supervise = iter(feature_train_supervise.cache().repeat().padded_batch(args.batch_size,
                ((), [None, args.dim_input])).prefetch(buffer_size=5))
        iter_feature_train = iter(feature_train.cache().repeat().shuffle(500).padded_batch(args.batch_size,
                ((), [None, args.dim_input])).prefetch(buffer_size=5))
        feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

        dataset_text = TextDataSet(list_files=[args.dirs.lm.data],
                                   args=args, _shuffle=True)
        tfdata_train = tf.data.Dataset.from_generator(
            dataset_text, (tf.int32), (tf.TensorShape([None])))
        iter_text = iter(tfdata_train.cache().repeat().shuffle(1000).map(
            lambda x: x[:args.model.D.max_label_len]).padded_batch(args.batch_size, ([args.model.D.max_label_len])).prefetch(buffer_size=5))


    # create model paremeters
    G = PhoneClassifier(args)
    D = PhoneDiscriminator3(args)
    G.summary()
    D.summary()

    optimizer_G = tf.keras.optimizers.Adam(args.opti.G.lr, beta_1=0.5, beta_2=0.9)
    optimizer_D = tf.keras.optimizers.Adam(args.opti.D.lr, beta_1=0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(G=G, optimizer_G=optimizer_G)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    step = 0
    best = 999
    # if a checkpoint exists, restore the latest checkpoint.
    if args.dirs.checkpoint:
        ckpt.restore(args.dirs.checkpoint)
        print('checkpoint {} restored!!'.format(args.dirs.checkpoint))
        step = int(args.dirs.checkpoint.split('-')[-1])

    start_time = datetime.now()
    num_processed = 0
    progress = 0

    while step < 99999999:
        start = time()

        for _ in range(args.opti.D_G_rate):
            uttids, x = next(iter_feature_train)
            text = next(iter_text)
            P_Real = tf.one_hot(text, args.dim_output)
            loss_D, loss_D_fake, loss_D_real, gp = train_D(
                x, P_Real, text>0, G, D, optimizer_D, args.lambda_gp, args.model.D.max_label_len)
        # loss_D = gp = 0

        uttids, x = next(iter_feature_train)
        supervise_uttids, x_supervise = next(iter_feature_supervise)
        trans_supervise = dataset_train_supervise.get_attrs('trans', supervise_uttids.numpy())
        loss_G, loss_G_supervise = train_G(
            x, x_supervise, trans_supervise, G, D, optimizer_G, args.lambda_supervise, args.model.D.max_label_len)

        num_processed += len(x)
        progress = num_processed / args.data.train_size
        if step % 10 == 0:
            print('loss_G: {:.3f}|{:.3f}\tloss_D: {:.3f}|{:.3f}|{:.3f}|{:.3f}\tbatch: {}|{}\tused: {:.3f}\t {:.3f}% iter: {}'.format(
                   loss_G, loss_G_supervise, loss_D, loss_D_fake, loss_D_real, gp, x.shape, text.shape, time()-start, progress*100.0, step))
            with writer.as_default():
                tf.summary.scalar("costs/loss_G", loss_G, step=step)
                tf.summary.scalar("costs/loss_D", loss_D, step=step)
                tf.summary.scalar("costs/loss_G_supervise", loss_G_supervise, step=step)
        if step % args.dev_step == 0:
            cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, G)
            # if cer < best:
            #     G_values = save_varibales(G)
            #     best = cer
            # else:
            #     load_values(G, G_values)
            #     print('try failed')
            with writer.as_default():
                tf.summary.scalar("performance/cer", cer, step=step)
        if step % args.decode_step == 0:
            monitor(dataset_dev[0], G)
        if step % args.save_step == 0:
            save_path = ckpt_manager.save(step)
            print('save model {}'.format(save_path))

        step += 1

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


# @tf.function(experimental_relax_shapes=True)
def train_G(x, _x, _y, G, D, optimizer_G, lambda_supervise, len_D):
    params_G = G.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape_G:
        tape_G.watch(params_G)

        # supervise
        _logits = G(_x, training=True)
        loss_G_supervise = ctc_loss(
            _logits,
            get_tensor_len(_logits),
            _y,
            tf.reduce_sum(tf.cast(_y>0, tf.int32), -1))
        loss_G_supervise = tf.reduce_mean(loss_G_supervise)
        # loss_G = loss_G_supervise
        # unsupervise
        logits = G(x, training=True)
        logits_shrunk = pad_to(ctc_shrink(logits), len_D)[:, :len_D, :]
        P_G = tf.nn.softmax(logits_shrunk)
        disc_fake = D(P_G, training=False)

        loss_G = lambda_supervise * loss_G_supervise - tf.reduce_mean(disc_fake)

    gradients_G = tape_G.gradient(loss_G, params_G)
    optimizer_G.apply_gradients(zip(gradients_G, params_G))

    return loss_G, loss_G_supervise


# @tf.function(experimental_relax_shapes=True)
def train_D(x, P_Real, mask_real, G, D, optimizer_D, lambda_gp, len_D):
    params_D = D.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape_D:
        tape_D.watch(params_D)

        logits = G(x, training=False)
        logits_shrunk = pad_to(ctc_shrink(logits), len_D)[:, :len_D, :]
        P_G = tf.nn.softmax(logits_shrunk)
        disc_real = D(P_Real, training=True) # to be +inf
        disc_fake = D(P_G, training=True) # to be -inf

        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        gp = gradient_penalty(D, P_Real, P_G)
        disc_cost += lambda_gp * gp

    gradients_D = tape_D.gradient(disc_cost, params_D)
    optimizer_D.apply_gradients(zip(gradients_D, params_D))

    return disc_cost, tf.reduce_mean(disc_fake), tf.reduce_mean(disc_real), gp


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
    cer = evaluate(feature_dev, dataset_dev, args.data.dev_size, G)
    decode(dataset, G, args.idx2token, 'output/'+save_file)


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
        # len_logits = get_tensor_len(logits)
        # preds = ctc_decode(logits, len_logits)
        logits_shrunk = ctc_shrink(logits)
        preds = tf.argmax(logits_shrunk, -1)
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

def monitor(sample, model):
    x = np.array([sample['feature']], dtype=np.float32)
    logits = model(x)
    logits_shrunk = ctc_shrink(logits)
    preds = tf.argmax(logits_shrunk, -1)
    print('predicts: \n', preds.numpy()[0])
    # print('align: \n', sample['align'])
    print('trans: \n', sample['trans'])


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

    elif param.mode == 'decode':
        """
        python main_supervise.py -m decode --name timit_supervised2_decode.txt --gpu 0 -c configs/timit_supervised2.yaml
        """
        print('enter the DECODING phrase')
        assert args.dirs.checkpoint
        assert param.name
        Decode(param.name)


        # python ../../main.py -m save --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
