#!/usr/bin/env python

from datetime import datetime
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf

from eastonCode.tfTools.tfData import TFData

from utils.arguments import args
from utils.dataset import ASR_align_DataSet
from utils.tools import frames_constrain_loss, align_accuracy, get_predicts, CE_loss, evaluation, decode, monitor

from models.GAN import PhoneClassifier


ITERS = 200000 # How many iterations to train for
tf.random.set_seed(args.seed)

def Train():
    dataset_dev = ASR_align_DataSet(
        file=[args.dirs.dev.data],
        args=args,
        _shuffle=False,
        transform=True)
    with tf.device("/cpu:0"):
        # wav data
        tfdata_train = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.train.tfdata,
                        args=args).read(_shuffle=False)
        tfdata_dev = TFData(dataset=None,
                        dataAttr=['feature', 'label', 'align'],
                        dir_save=args.dirs.dev.tfdata,
                        args=args).read(_shuffle=False)
        if args.num_supervised:
            x_0, y_0, aligns_0 = next(iter(tfdata_train.take(args.num_supervised).\
                padded_batch(args.num_supervised, ([None, args.dim_input], [None], [None]))))
        iter_train = iter(tfdata_train.cache().repeat().shuffle(500).padded_batch(args.batch_size,
                ([None, args.dim_input], [None], [None])).prefetch(buffer_size=5))
        tfdata_dev = tfdata_dev.padded_batch(args.batch_size, ([None, args.dim_input], [None], [None]))

    # create model paremeters
    model = PhoneClassifier(args)
    model.summary()
    optimizer_G = tf.keras.optimizers.Adam(args.opti.lr, beta_1=0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(model=model, optimizer_G=optimizer_G)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=5)
    step = 0

    # if a checkpoint exists, restore the latest checkpoint.
    if args.dirs.checkpoint:
        _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
        ckpt.restore(_ckpt_manager.latest_checkpoint)
        print ('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))
        step = int(_ckpt_manager.latest_checkpoint.split('-')[-1])

    start_time = datetime.now()

    while step < 99999999:
        start = time()

        if args.num_supervised:
            x = x_0
            loss_supervise = train_G_supervised(x_0, y_0, model, optimizer_G, args.dim_output)
        else:
            x, y, aligns = next(iter_train)
            loss_supervise = train_G_supervised(x, y, model, optimizer_G, args.dim_output)

        if step % 10 == 0:
            print('loss_supervise: {:.3f}\tbatch: {}\tused: {:.3f}\tstep: {}'.format(
                   loss_supervise, x.shape, time()-start, step))
            with writer.as_default():
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


def Decode(save_file):
    dataset = ASR_align_DataSet(
        file=[args.dirs.train.data],
        args=args,
        _shuffle=False,
        transform=True)
    # dataset = ASR_align_DataSet(
    #     file=[args.dirs.dev.data],
    #     args=args,
    #     _shuffle=False,
    #     transform=True)

    model = PhoneClassifier(args)
    model.summary()

    optimizer_G = tf.keras.optimizers.Adam(args.opti.lr, beta_1=0.5, beta_2=0.9)
    ckpt = tf.train.Checkpoint(model=model, optimizer_G=optimizer_G)

    _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
    ckpt.restore(_ckpt_manager.latest_checkpoint)
    print ('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))
    decode(dataset, model, args.idx2token, 'output/'+save_file)


# @tf.function
def train_G_supervised(x, labels, model, optimizer_G, dim_output):
    with tf.GradientTape() as tape_G:
        logits = model(x, training=True)
        ce_loss = CE_loss(logits, labels, dim_output, confidence=0.9)
        gen_loss = ce_loss

    gradients_G = tape_G.gradient(gen_loss, model.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, model.trainable_variables))

    return gen_loss


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
        python ../../main.py -m decode --gpu 1 --name kin_asr -c configs/rna_char_big3.yaml
        """
        if param.name:
            args.dir_exps = args.dir_exps / param.name
            args.dir_log = args.dir_exps / 'log'
            args.dir_checkpoint = args.dir_exps / 'checkpoint'
            if args.dir_exps.is_dir():
                os.system('rm -r '+ str(args.dir_exps))
            args.dir_exps.mkdir()
            args.dir_log.mkdir()
            args.dir_checkpoint.mkdir()
            with open(args.dir_exps / 'configs.txt', 'w') as fw:
                print(args, file=fw)
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
