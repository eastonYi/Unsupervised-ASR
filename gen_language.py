# coding=utf-8

import numpy as np
import tensorflow as tf
from time import time

from utils.dataset import load_dataset
from utils.model import Generator, Discriminator
# from tensorflow.keras.layers import Embedding
np.set_printoptions(precision=4)


BATCH_SIZE = 128 # Batch size
ITERS = 200000 # How many iterations to train for
SEQ_LEN = 32 # Sequence length in characters
size_embedding = 512
DIM = 512 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
CRITIC_ITERS = 10 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 10000000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).
DATA_DIR = '/data/sxu/easton/data/WMT/1-billion-word-language-modeling-benchmark-r13output'


def main():
    lines, charmap, inv_charmap = load_dataset(
        max_length=SEQ_LEN,
        max_n_examples=MAX_N_EXAMPLES,
        data_dir=DATA_DIR)
    # Dataset iterator
    def inf_train_gen():
        while True:
            np.random.shuffle(lines)
            for i in range(0, len(lines)-BATCH_SIZE+1, BATCH_SIZE):
                yield np.array(
                    [[charmap[c] for c in l] for l in lines[i:i+BATCH_SIZE]],
                    dtype='int32'
                )
    iter_data = inf_train_gen()

    G = Generator(dim_hidden=DIM, seq_len=SEQ_LEN, dim_output=len(charmap))
    D = Discriminator(dim_hidden=DIM, seq_len=SEQ_LEN)

    for iteration in range(ITERS):

        if iteration == 1:
            G.summary()
            D.summary()

        start = time()

        cost_G = train_G(G, D, G.trainable_variables)

        for _ in range(CRITIC_ITERS):
            text = next(iter_data)
            cost_D, gp = train_D(text, charmap, G, D, D.trainable_variables)

        if iteration % 1 == 0:
            fake_inputs = G(1, SEQ_LEN)
            fake_inputs_discrete = tf.argmax(fake_inputs[0], -1)
            print('cost_G: {:.3f}\t cost_D: {:.3f}|{:.3f}\t used: {:.3f}'.format(cost_G, cost_D, gp, time()-start))
            print(''.join(inv_charmap[i] for i in text[0]), ' || ', ''.join(inv_charmap[i] for i in fake_inputs_discrete))


def train_G(G, D, params_G):
    """
    real_inputs_discrete:  tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape_G:
        tape_G.watch(params_G)
        fake_inputs = G(BATCH_SIZE, SEQ_LEN)
        disc_fake = D(fake_inputs)

        gen_cost = -tf.reduce_mean(disc_fake)

    gradients_G = tape_G.gradient(gen_cost, params_G)
    G.optimizer.apply_gradients(zip(gradients_G, params_G))

    return gen_cost


def train_D(text, charmap, G, D, params_D):
    real_inputs_discrete = text
    with tf.GradientTape(watch_accessed_variables=False) as tape_D:
        tape_D.watch(params_D)
        real_inputs = tf.one_hot(real_inputs_discrete, len(charmap))
        fake_inputs = G(BATCH_SIZE, SEQ_LEN)
        disc_real = D(real_inputs) # to be +inf
        disc_fake = D(fake_inputs) # to be -inf

        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        gp = gradient_penalty(D, real_inputs, fake_inputs)
        disc_cost += LAMBDA*gp

    gradients_D = tape_D.gradient(disc_cost, params_D)
    D.optimizer.apply_gradients(zip(gradients_D, params_D))

    return disc_cost, gp


def gradient_penalty(f, real, fake):
    def _interpolate(a, b):
        shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
        alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
        inter = a + alpha * (b - a)
        inter.set_shape(a.shape)
        return inter

    x = _interpolate(real, fake)
    with tf.GradientTape() as t:
        t.watch(x)
        pred = f(x)
    grad = t.gradient(pred, x)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
    gp = tf.reduce_mean((norm - 1.)**2)

    return gp


if __name__ == '__main__':
    import os
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', type=str, dest='mode', default='train')
    parser.add_argument('--name', type=str, dest='name', default=None)
    parser.add_argument('--gpu', type=str, dest='gpu', default=0)
    parser.add_argument('-c', type=str, dest='config')

    param = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = param.gpu
    print('enter the TRAINING phrase')
    main()
