#!/usr/bin/env python

from datetime import datetime
from time import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import logging
import sys
import yaml
from pathlib import Path

from utils.tools import TFData, mkdirs
from utils.dataset import ASR_classify_DataSet
from utils.tools import CE_loss


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        try:
            if type(self[item]) is dict:
                self[item] = AttrDict(self[item])
            res = self[item]
        except:

            print('not found {}'.format(item))
            res = None
        return res
ITERS = 200000 # How many iterations to train for
tf.random.set_seed(0)
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')
CONFIG_FILE = sys.argv[-1]
args = AttrDict(yaml.load(open(CONFIG_FILE), Loader=yaml.SafeLoader))

# dirs
dir_dataInfo = Path.cwd() / 'data'
dir_exps = Path.cwd() / 'exps' / args.dirs.exp
args.dir_exps = dir_exps / CONFIG_FILE.split('/')[-1].split('.')[0]
args.dir_log = args.dir_exps / 'log'
args.dir_checkpoint = args.dir_exps / 'checkpoint'

if not dir_dataInfo.is_dir(): dir_dataInfo.mkdir()
if not dir_exps.is_dir(): dir_exps.mkdir()
if not args.dir_exps.is_dir(): args.dir_exps.mkdir()
if not args.dir_log.is_dir(): args.dir_log.mkdir()
if not args.dir_checkpoint.is_dir(): args.dir_checkpoint.mkdir()

args.dirs.train.tfdata = Path(args.dirs.train.tfdata)
args.dirs.dev.tfdata = Path(args.dirs.dev.tfdata)
mkdirs(args.dirs.train.tfdata)
mkdirs(args.dirs.dev.tfdata)
args.dirs.train.feat_len = args.dirs.train.tfdata/'feature_length.txt'
args.dirs.dev.feat_len = args.dirs.dev.tfdata/'feature_length.txt'

try:
    args.dim_input = TFData.read_tfdata_info(args.dirs.train.tfdata)['dim_feature']
    args.data.train_size = TFData.read_tfdata_info(args.dirs.train.tfdata)['size_dataset']
    args.data.dev_size = TFData.read_tfdata_info(args.dirs.dev.tfdata)['size_dataset']
except:
    print("have not converted to tfdata yet: ")


def Classifier(args):
    x = input_x = tf.keras.layers.Input(shape=[None, args.dim_input],
                                        name='generator_input_x')

    for _ in range(2):
        x = tf.keras.layers.GRU(256,
                                return_sequences=True,
                                dropout=0.1)(x)

    logits = tf.keras.layers.Dense(args.dim_output, activation='linear')(x)
    len_x = tf.reduce_sum(tf.cast(tf.reduce_sum(tf.abs(input_x), -1) > 0, tf.int32), -1)
    ids = tf.stack([tf.range(tf.shape(x)[0]), len_x-1], 1)
    logits_c = tf.gather_nd(logits, ids)
    logits_c = tf.reshape(logits_c, [-1, args.dim_output])

    model = tf.keras.Model(inputs=input_x,
                           outputs=logits_c,
                           name='sequence_generator')

    return model


def Train():
    dataset_train = ASR_classify_DataSet(
        dir_wavs=args.dirs.wav,
        class_file=args.dirs.train.label,
        args=args,
        _shuffle=True,
        transform=True)
    # dataset_dev = ASR_classify_DataSet(
    #     dir_wavs=args.dirs.wav,
    #     class_file=args.dirs.dev.label,
    #     args=args,
    #     _shuffle=False,
    #     transform=True)
    dataset_dev = ASR_classify_DataSet(
        dir_wavs=args.dirs.wav,
        class_file=args.dirs.train.label,
        args=args,
        _shuffle=False,
        transform=True)

    args.vocab = dataset_train.dict_class
    args.dim_output = len(args.vocab)

    with tf.device("/cpu:0"):
        # wav data
        # TFData(dataset=dataset_train,
        #        dir_save=args.dirs.train.tfdata,
        #        args=args).save('0')
        # TFData(dataset=dataset_dev,
        #        dir_save=args.dirs.dev.tfdata,
        #        args=args).save('0')
        # import pdb; pdb.set_trace()

        feature_train = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args).read()
        # feature_dev = TFData(dataset=dataset_dev,
        #                 dir_save=args.dirs.dev.tfdata,
        #                 args=args).read()
        feature_dev = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args).read()

        iter_feature_train = iter(feature_train.repeat().shuffle(500).padded_batch(args.batch_size,
                ((), [None, args.dim_input])).prefetch(buffer_size=5))
        feature_dev = feature_dev.padded_batch(args.batch_size, ((), [None, args.dim_input]))

    # create model paremeters
    model = Classifier(args)
    model.summary()
    optimizer_G = tf.keras.optimizers.Adam(args.opti.lr, beta_1=0.5, beta_2=0.9)

    writer = tf.summary.create_file_writer(str(args.dir_log))
    ckpt = tf.train.Checkpoint(model=model, optimizer_G=optimizer_G)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args.dir_checkpoint, max_to_keep=20)
    step = 0

    # if a checkpoint exists, restore the latest checkpoint.
    if args.dirs.checkpoint:
        _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
        ckpt.restore(_ckpt_manager.latest_checkpoint)
        print('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))
        step = int(_ckpt_manager.latest_checkpoint.split('-')[-1])

    start_time = datetime.now()

    while step < 9999:
        start = time()
        uttids, x = next(iter_feature_train)
        y = dataset_train.get_y(uttids.numpy())
        loss = train_G_supervised(x, y, model, optimizer_G, args.dim_output)
        if step % 10 == 0:
            print('loss: {:.3f}\tbatch: {}\tused: {:.3f}\tstep: {}'.format(
                   loss, x.shape, time()-start, step))
        if step % args.dev_step == 0:
            # p = evaluate(feature_dev, dataset_dev, args.data.dev_size, model)
            p = evaluate(feature_dev, dataset_dev, args.data.dev_size, model)
            print('eval performance: {:.2f}%'.format(p*100))
        if step % args.save_step == 0:
            save_path = ckpt_manager.save(step)
            print('save model {}'.format(save_path))

        step += 1

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def evaluate(feature_dev, dataset_dev, dev_size, model):
    list_p = []
    for uttids, x in feature_dev:
        y = dataset_dev.get_y(uttids.numpy())
        logits = model(x)
        _y = tf.argmax(logits, -1)
        p = tf.reduce_mean(tf.cast(tf.equal(y, _y), tf.float32))
        list_p.append(p)

    return tf.reduce_mean(list_p)


# @tf.function
def train_G_supervised(x, y, model, optimizer_G, dim_output):
    with tf.GradientTape() as tape_G:
        logits = model(x, training=True)
        ce_loss = CE_loss(logits, y, dim_output, confidence=0.9)
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

    os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)

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
