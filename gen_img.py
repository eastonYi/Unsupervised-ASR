# coding=utf-8

import functools
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from time import time

from utils.dataset import make_32x32_dataset
from utils.tools_img import immerge, imwrite
np.set_printoptions(precision=4)


batch_size = 64 # Batch size
ITERS = 200000 # How many iterations to train for
SEQ_LEN = 32 # Sequence length in characters
size_embedding = 512
z_dim = 128 # Model dimensionality. This is fairly slow and overfits, even on
          # Billion Word. Consider decreasing for smaller datasets.
lr = 0.0002
CRITIC_ITERS = 5 # How many critic iterations per generator iteration. We
                  # use 10 for the results in the paper, but 5 should work fine
                  # as well.
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 10000000 # Max number of data examples to load. If data loading
                          # is too slow or takes too much RAM, you can decrease
                          # this (at the expense of having less training data).
d_norm = 'batch_norm'

output_dir = './output'
summary_dir = output_dir + '/summaries'
sample_dir = output_dir + '/samples_batchNorm'
data = 'fashion_mnist'


def main():
    dataset, shape, len_dataset = make_32x32_dataset(data, batch_size, repeat=25)
    n_G_upsamplings = n_D_downsamplings = 3

    iter_data = iter(dataset)

    # networks
    G = ConvGenerator(input_shape=(1, 1, z_dim), output_channels=shape[-1], n_upsamplings=n_G_upsamplings, norm=d_norm, name='G_img')
    D = ConvDiscriminator(input_shape=shape, n_downsamplings=n_D_downsamplings, norm=d_norm, name='D_img')
    G.summary()
    D.summary()

    # adversarial_loss_functions
    # d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
    G_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
    D_optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)

    # summary
    train_summary_writer = tf.summary.create_file_writer(output_dir + '/summaries')

    z = tf.random.normal(shape=(100, 1, 1, z_dim))
    for iteration in range(ITERS):
        start = time()

        for _ in range(CRITIC_ITERS):
            img = next(iter_data)
            cost_D, gp = train_D(img, G, D, D_optimizer)

        cost_G = train_G(G, D, G_optimizer)

        if iteration % 10 == 0:
            print('cost_G: {:.3f}\t cost_D: {:.3f}|{:.3f}\t used: {:.3f}\t iter: {}'.format(cost_G, cost_D, gp, time()-start, iteration))
            # print(''.join(inv_charmap[i] for i in text[0]), ' || ', ''.join(inv_charmap[i] for i in fake_inputs_discrete))

        if iteration % 100 == 0:
            x_fake = G(z, training=False)
            img = immerge(x_fake, n_rows=10).squeeze()
            imwrite(img, sample_dir + '/iter-%09d.jpg' % iteration)


def ConvGenerator(input_shape=(1, 1, 128),
                  output_channels=3,
                  dim=64,
                  n_upsamplings=4,
                  norm='batch_norm',
                  name='ConvGenerator'):

    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1: 1x1 -> 4x4
    d = min(dim * 2 ** (n_upsamplings - 1), dim * 8)
    h = tf.keras.layers.Conv2DTranspose(d, 4, strides=1, padding='valid', use_bias=False)(h)
    h = Norm()(h)
    h = tf.keras.layers.ReLU()(h)

    # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
    for i in range(n_upsamplings - 1):
        d = min(dim * 2 ** (n_upsamplings - 2 - i), dim * 8)
        h = tf.keras.layers.Conv2DTranspose(d, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2DTranspose(output_channels, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.Activation('tanh')(h)

    return tf.keras.Model(inputs=inputs, outputs=h, name=name)


def ConvDiscriminator(input_shape=(64, 64, 3),
                      dim=64,
                      n_downsamplings=4,
                      norm='batch_norm',
                      name='ConvDiscriminator'):
    Norm = _get_norm_layer(norm)

    # 0
    h = inputs = tf.keras.Input(shape=input_shape)

    # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for i in range(n_downsamplings - 1):
        d = min(dim * 2 ** (i + 1), dim * 8)
        h = tf.keras.layers.Conv2D(d, 4, strides=2, padding='same', use_bias=False)(h)
        h = Norm()(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2: logit
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='valid')(h)

    return tf.keras.Model(inputs=inputs, outputs=h, name=name)


def train_G(G, D, opti_G):
    """
    real_inputs_discrete:  tf.int32, shape=[BATCH_SIZE, SEQ_LEN])
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape_G:
        tape_G.watch(G.trainable_variables)
        z = tf.random.normal(shape=(batch_size, 1, 1, z_dim))
        x_fake = G(z, training=True)
        disc_fake = D(x_fake, training=True)
        gen_cost = -tf.reduce_mean(disc_fake)

    gradients_G = tape_G.gradient(gen_cost, G.trainable_variables)
    opti_G.apply_gradients(zip(gradients_G, G.trainable_variables))

    return gen_cost


def train_D(real_inputs, G, D, opti_D):
    """
    """
    with tf.GradientTape(watch_accessed_variables=False) as tape_D:
        tape_D.watch(D.trainable_variables)
        z = tf.random.normal(shape=(batch_size, 1, 1, z_dim))
        fake_inputs = G(z, training=True)

        disc_real = D(real_inputs, training=True)
        disc_fake = D(fake_inputs, training=True)

        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        gp = gradient_penalty(D, real_inputs, fake_inputs)
        # gp = gradient_penalty(functools.partial(D, training=True), real_inputs, fake_inputs, mode='wgan-gp')
        disc_cost += LAMBDA*gp

    gradients_D = tape_D.gradient(disc_cost, D.trainable_variables)
    opti_D.apply_gradients(zip(gradients_D, D.trainable_variables))

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


def _get_norm_layer(norm):
    if norm == 'none':
        return lambda: lambda x: x
    elif norm == 'batch_norm':
        return tf.keras.layers.BatchNormalization
    # elif norm == 'instance_norm':
    #     return tfa.layers.InstanceNormalization
    # elif norm == 'layer_norm':
    #     return tfa.layers.GroupNormalization
        # return tf.keras.layers.LayerNormalization


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

    gpus = tf.config.experimental.list_physical_devices('GPU')
    assert len(gpus) > 0, "Not enough GPU hardware devices available"
    [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

    print('enter the TRAINING phrase')
    main()
