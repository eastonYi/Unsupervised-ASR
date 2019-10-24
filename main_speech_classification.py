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
from tqdm import tqdm
from random import shuffle

from utils.tools import mkdirs, get_tensor_len
from utils.dataset import ASR_classify_ArkDataSet


class TFData:
    """
    test on TF2.0
    """
    def __init__(self, dataset, dir_save, args, size_file=5000000, max_feat_len=3000):
        self.dataset = dataset
        self.max_feat_len = max_feat_len
        self.dir_save = dir_save
        self.args = args
        self.size_file = size_file
        self.dim_feature = dataset[0]['feature'].shape[-1] \
            if dataset else self.read_tfdata_info(dir_save)['dim_feature']

    def split_save(self, length_file='feature_length.txt', capacity=50000):
        num_token = 0
        num_damaged_sample = 0
        fw = open(self.dir_save/length_file, 'w')

        def serialize_example(feature, y):
            atts = {
                'feature': self._bytes_feature(feature.tostring()),
                'class': self._int_feature(y),
            }
            example_proto = tf.train.Example(features=tf.train.Features(feature=atts))

            return example_proto.SerializeToString()

        def generator():
            nonlocal fw, i, capacity
            # for sample, _ in zip(self.dataset, tqdm(range(len(self.dataset)))):
            for j in tqdm(range(i*capacity, min((i+1)*capacity, len(self.dataset)))):
                sample = self.dataset[j]
                line = sample['uttid'] + ' ' + str(len(sample['feature']))
                fw.write(line + '\n')
                yield serialize_example(sample['feature'], sample['class'])

        for i in range(len(self.dataset)//capacity + 1):
        # for i in [2,3,4]:
            dataset_tf = tf.data.Dataset.from_generator(
                generator=generator,
                output_types=tf.string,
                output_shapes=())
            record_file = self.dir_save/'{}.recode'.format(i)
            mkdirs(record_file)
            writer = tf.data.experimental.TFRecordWriter(str(record_file))
            writer.write(dataset_tf)

        with open(str(self.dir_save/'tfdata.info'), 'w') as fw:
            fw.write('data_file {}\n'.format(self.dataset.file))
            fw.write('dim_feature {}\n'.format(self.dim_feature))
            fw.write('num_tokens {}\n'.format(num_token))
            fw.write('size_dataset {}\n'.format(len(self.dataset)-num_damaged_sample))
            fw.write('damaged samples: {}\n'.format(num_damaged_sample))

        return

    def read(self, _shuffle=False):
        """
        the tensor could run unlimitatly
        return a iter
        """
        list_filenames = self.fentch_filelist(self.dir_save)
        if _shuffle:
            shuffle(list_filenames)
        else:
            list_filenames.sort()

        raw_dataset = tf.data.TFRecordDataset(list_filenames)

        def _parse_function(example_proto):
            sample = tf.io.parse_single_example(
                example_proto,
                features={
                    'feature': tf.io.FixedLenFeature([], tf.string),
                    'class': tf.io.FixedLenFeature([], tf.int64)
                }
            )
            feature = tf.reshape(tf.io.decode_raw(sample['feature'], tf.float32),
                                 [-1, self.dim_feature])[:self.max_feat_len, :]
            label = tf.reshape(sample['class'], [1])

            return feature, label

        feature = raw_dataset.map(_parse_function)

        return feature

    def __len__(self):
        return self.read_tfdata_info(self.dir_save)['size_dataset']

    @staticmethod
    def fentch_filelist(dir_data):
        p = Path(dir_data)
        assert p.is_dir()

        return [str(i) for i in p.glob('*.recode')]

    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a list of string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _int_feature(value):
        """Returns a int_list."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def read_tfdata_info(dir_save):
        data_info = {}
        with open(dir_save/'tfdata.info') as f:
            for line in f:
                if 'dim_feature' in line or \
                    'num_tokens' in line or \
                    'size_dataset' in line:
                    line = line.strip().split(' ')
                    data_info[line[0]] = int(line[1])

        return data_info


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
    args.dim_input = TFData.read_tfdata_info(args.dirs.dev.tfdata)['dim_feature']
    args.data.train_size = TFData.read_tfdata_info(args.dirs.train.tfdata)['size_dataset']
    args.data.dev_size = TFData.read_tfdata_info(args.dirs.dev.tfdata)['size_dataset']
except:
    print("have not converted to tfdata yet: ")


def Classifier(args):
    num_layers = 2
    x = input_x = tf.keras.layers.Input(shape=[None, args.dim_input],
                                        name='generator_input_x')

    for _ in range(num_layers):
        x = tf.keras.layers.GRU(256,
                                return_sequences=True,
                                dropout=0.5)(x)
        x = tf.keras.layers.MaxPooling1D()(x)

    logits = tf.keras.layers.Dense(args.dim_output, activation='linear')(x)
    len_x = tf.reduce_sum(tf.cast(tf.reduce_sum(tf.abs(input_x), -1) > 0, tf.int32), -1)
    len_x = tf.cast(len_x/tf.pow(2, num_layers), tf.int32)

    ids = tf.stack([tf.range(tf.shape(x)[0]), len_x-1], 1)
    logits_c = tf.gather_nd(logits, ids)
    logits_c = tf.reshape(logits_c, [-1, args.dim_output])

    model = tf.keras.Model(inputs=input_x,
                           outputs=logits,
                           name='sequence_generator')

    return model


def Conv1D(dim_output, kernel_size, strides=1, padding='same'):
    conv_op = tf.keras.layers.Conv1D(
        filters=dim_output,
        kernel_size=(kernel_size,),
        strides=strides,
        padding='same',
        use_bias=True)

    return conv_op


def Train():
    dataset_train = ASR_classify_ArkDataSet(
        scp_file=args.dirs.train.scp,
        class_file=args.dirs.train.label,
        args=args,
        _shuffle=True)
    dataset_dev = ASR_classify_ArkDataSet(
        scp_file=args.dirs.dev.scp,
        class_file=args.dirs.dev.label,
        args=args,
        _shuffle=False)

    args.vocab = dataset_train.dict_class
    args.dim_output = len(args.vocab)

    with tf.device("/cpu:0"):
        # wav data
        # TFData(dataset=dataset_train,
        #        dir_save=args.dirs.train.tfdata,
        #        args=args, max_feat_len=args.max_seq_len).split_save(capacity=500)
        # TFData(dataset=dataset_dev,
        #        dir_save=args.dirs.dev.tfdata,
        #        args=args, max_feat_len=args.max_seq_len).split_save(capacity=500)
        # import pdb; pdb.set_trace()

        feature_train = TFData(dataset=dataset_train,
                        dir_save=args.dirs.train.tfdata,
                        args=args, max_feat_len=args.max_seq_len).read()
        feature_dev = TFData(dataset=dataset_dev,
                        dir_save=args.dirs.dev.tfdata,
                        args=args, max_feat_len=args.max_seq_len).read()
        iter_feature_train = iter(feature_train.cache().repeat().shuffle(5).padded_batch(args.batch_size,
                ([None, args.dim_input], [1])).prefetch(buffer_size=2))
        feature_train = feature_train.padded_batch(1, ([None, args.dim_input], [1]))
        feature_dev = feature_dev.padded_batch(1, ([None, args.dim_input], [1]))

    # create model paremeters
    model = Classifier(args)
    model.summary()
    optimizer_G = tf.keras.optimizers.Adam(args.opti.lr, beta_1=0.5, beta_2=0.9)

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
    processed = 0
    while step < 999999:
        start = time()
        x, y = next(iter_feature_train)
        processed += len(x)
        loss = train_G_supervised(x, y, model, optimizer_G, args.dim_output)
        batch_time = time()-start
        if step % 10 == 0:
            print('loss: {:.3f}\tbatch: {}\tused: {:.3f}\t {:.2f}% step: {}'.format(
                   loss, x.shape, batch_time, processed/args.data.train_size , step))
        if step % args.dev_step == 1:
            p_train = evaluate(feature_train, dataset_train, args.data.train_size, model)
            p_dev = evaluate(feature_dev, dataset_dev, args.data.dev_size, model)
            print('eval performance: train: {:.2f}% dev: {:.2f}%'.format(p_train*100, p_dev*100))
        if step % args.save_step == 0:
            save_path = ckpt_manager.save(step)
            print('save model {}'.format(save_path))

        step += 1

    print('training duration: {:.2f}h'.format((datetime.now()-start_time).total_seconds()/3600))


def evaluate(feature_dev, dataset_dev, dev_size, model):
    list_p = []
    list_res = []
    for x, y in tqdm(feature_dev):
        y = y[0]
        logits = model(x)
        _y = tf.argmax(tf.math.bincount(tf.argmax(logits, -1, output_type=tf.int32)),
                       output_type=tf.int64)
        p = tf.cast(tf.equal(y, _y), tf.float32)
        list_p.append(p)
        list_res.append((y.numpy()[0], _y.numpy()))
    print(list_res[:30])
    return tf.reduce_mean(list_p)


# @tf.function(experimental_relax_shapes=True)
def train_G_supervised(x, y, model, optimizer_G, dim_output):
    with tf.GradientTape() as tape_G:
        # gen_loss = step(model, x, y, dim_output)
        logits = model(x, training=True)
        _y = tf.tile(y, [1, logits.shape[1]])
        gen_loss = CE_loss(logits, _y, dim_output, confidence=0.9)

    gradients_G = tape_G.gradient(gen_loss, model.trainable_variables)
    optimizer_G.apply_gradients(zip(gradients_G, model.trainable_variables))

    return gen_loss


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


def Infer():
    dataset_dev = ASR_classify_ArkDataSet(
        scp_file=args.dirs.dev.scp,
        class_file=args.dirs.dev.label,
        args=args,
        _shuffle=False)

    args.vocab = dataset_dev.dict_class
    args.dim_output = len(args.vocab)

    with tf.device("/cpu:0"):
        feature_dev = TFData(dataset=dataset_dev,
                             dir_save=args.dirs.dev.tfdata,
                             args=args, max_feat_len=args.max_seq_len).read()
        feature_dev = feature_dev.padded_batch(1, ([None, args.dim_input], [1]))

    # create model paremeters
    model = Classifier(args)
    model.summary()
    optimizer_G = tf.keras.optimizers.Adam(args.opti.lr, beta_1=0.5, beta_2=0.9)

    ckpt = tf.train.Checkpoint(model=model, optimizer_G=optimizer_G)

    _ckpt_manager = tf.train.CheckpointManager(ckpt, args.dirs.checkpoint, max_to_keep=1)
    ckpt.restore(_ckpt_manager.latest_checkpoint)
    print('checkpoint {} restored!!'.format(_ckpt_manager.latest_checkpoint))

    # p_train = evaluate(feature_train, dataset_train, args.data.train_size, model)
    p_dev = evaluate(feature_dev, dataset_dev, args.data.dev_size, model)
    print('dev acc: {:.3f}%'.format(p_dev*100))


def wav2class(model, wav_file='ex.wav'):
    os.system('. extract_feat/path.sh')
    os.system('. extract_feat/cmd.sh')
    os.system('mkdir extract_feat/tmp/data -p')
    os.system('mkdir extract_feat/tmp/fbank80 -p')
    os.system('mkdir extract_feat/tmp/fbank_lef_sub -p')
    os.system('touch extract_feat/tmp/data/wav.scp')
    os.system('touch extract_feat/tmp/data/utt2spk')

    name = wav_file.split('/')[-1]
    with open('extract_feat/tmp/data/wav.scp', 'w') as wavscp, open('extract_feat/tmp/data/utt2spk', 'w') as utt2spk:
        wavscp.write(name + ' ' + wav_file)
        utt2spk.write(name + ' ' + name)

    os.system('cp extract_feat/tmp/data/utt2spk extract_feat/tmp/data/spk2utt')
    os.system('cp extract_feat/tmp/data/utt2spk extract_feat/tmp/data/text')
    os.system('bash extract_feat/run_fbank.sh')
    os.system('splice-feats --left-context=64 --right-context=0 scp:extract_feat/tmp/data/feats.scp ark:- | subsample-feats --n=64 ark:- ark,scp:`pwd`/extract_feat/tmp/fbank_lef_sub/cmvn_lef64_sub64.ark,`pwd`/extract_feat/tmp/fbank_lef_sub/cmvn_lef64_sub64.scp')
    scp_path = sys.path[0] + '/extract_feat/tmp/fbank_lef_sub/cmvn_lef64_sub64.scp'

    from tools import ArkReader
    scp_reader = ArkReader(scp_path)
    feats = scp_reader.read_utt_data(name)
    print('extract feats success!')

    x = tf.expand_dims(feats, 0)
    if len(x) < 2:
        x = tf.concat([x, x], 0)
    logits = model(x, training=False)
    _y = tf.argmax(tf.math.bincount(tf.argmax(logits, -1, output_type=tf.int32)),
                   output_type=tf.int64).numpy()

    return _y



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
    elif param.mode == 'infer':
        Infer()
    elif param.mode == 'online':
        assert args.dirs.checkpoint
        from flask import Flask
        from flask import request

        app = Flask(__name__)

        args.dim_output = 3
        model = Classifier(args)

        @app.route('/', methods=['POST'])
        def evaluator():
            f = request.files['file']
            filename = f.filename
            file_name = 'extract_feat/' + filename
            f.save(file_name)
            # result = 'save ' + file_name + 'success!'
            result = wav2class(model, os.pwd() + '/' + file_name)

            return result

        app.run(host='0.0.0.0', port=5000)
