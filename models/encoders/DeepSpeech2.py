# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Network structure for DeepSpeech2 model."""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Bidirectional, Input
from utils.tools import get_tensor_len

# Supported rnn cells.
SUPPORTED_RNNS = {
    "LSTM": tf.keras.layers.LSTM,
    "GRU": tf.keras.layers.GRU,
}

# Parameters for batch normalization.
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997

# Filters of convolution layer
_CONV_FILTERS = 32


def DeepSpeech2(args):
    """Define DeepSpeech2 model.
    Args:
      num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.
      rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.
      is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
      rnn_hidden_size: an integer for the number of hidden states in each unit.
      dim_output: an integer, the number of output classes/labels.
      use_bias: a boolean specifying whether to use bias in the last fc layer.
    """
    num_rnn_layers = args.model.num_rnn_layers
    rnn_type = args.model.rnn_type
    is_bidirectional = args.model.is_bidirectional
    rnn_hidden_size = args.model.rnn_hidden_size
    dim_output = args.dim_output
    use_bias = args.model.use_bias
    size_feat = args.dim_input

    x = inputs = Input(shape=[None, args.dim_input], name='input_x')
    len_seq = get_tensor_len(x)
    size_length = tf.shape(x)[1]
    size_feat = int(size_feat/3)
    x = tf.reshape(x, [-1, size_length, size_feat, 3])
    # Two cnn layers.
    x = _conv_bn_layer(
        x, padding=(20, 5), filters=_CONV_FILTERS, kernel_size=(41, 11),
        strides=(2, 2))

    x = _conv_bn_layer(
        x, padding=(10, 5), filters=_CONV_FILTERS, kernel_size=(21, 11),
        strides=(2, 1))

    # output of conv_layer2 with the shape of
    # [batch_size (N), times (T), features (F), channels (C)].
    # Convert the conv output to rnn input.
    # batch_size = tf.shape(x)[0]
    len_seq = tf.cast(tf.math.ceil(tf.cast(len_seq, tf.float32)/4), tf.int32)
    size_length = tf.cast(tf.math.ceil(tf.cast(size_length, tf.float32)/4), tf.int32)
    size_feat = x.shape[2] * x.shape[3]
    x = tf.reshape(x, [-1, size_length, size_feat])

    # RNN layers.
    rnn_cell = SUPPORTED_RNNS[rnn_type]
    for layer_counter in range(num_rnn_layers):
        # No batch normalization on the first layer.
        # print(x)
        is_batch_norm = (layer_counter != 0)
        x = _rnn_layer(
            x, rnn_cell, rnn_hidden_size,
            is_batch_norm, is_bidirectional)

    # FC layer with batch norm.
    x = batch_norm()(x)
    logits = Dense(dim_output, use_bias=use_bias)(x)
    pad_mask = tf.tile(tf.expand_dims(tf.sequence_mask(len_seq, dtype=tf.float32), -1), [1, 1, dim_output])
    logits *= pad_mask

    model = tf.keras.Model(inputs=inputs,
                           outputs=logits,
                           name='sequence_generator')

    return model


def batch_norm():
    """Batch normalization layer.

    Note that the momentum to use will affect validation accuracy over time.
    Batch norm has different behaviors during training/evaluation. With a large
    momentum, the model takes longer to get a near-accurate estimation of the
    moving mean/variance over the entire training dataset, which means we need
    more iterations to see good evaluation results. If the training data is evenly
    distributed over the feature space, we can also try setting a smaller momentum
    (such as 0.1) to get good evaluation result sooner.

    Args:
        inputs: input data for batch norm layer.
        training: a boolean to indicate if it is in training stage.

    Returns:
        tensor output from batch norm layer.
    """
    return tf.keras.layers.BatchNormalization(
        momentum=_BATCH_NORM_DECAY,
        epsilon=_BATCH_NORM_EPSILON,
        fused=False)


def _conv_bn_layer(inputs, padding, filters, kernel_size, strides):
    """Defines 2D convolutional + batch normalization layer.

    Args:
        inputs: input data for convolution layer.
        padding: padding to be applied before convolution layer.
        filters: an integer, number of output filters in the convolution.
        kernel_size: a tuple specifying the height and width of the 2D convolution
          window.
        strides: a tuple specifying the stride length of the convolution.
        layer_id: an integer specifying the layer index.
        training: a boolean to indicate which stage we are in (training/eval).

    Returns:
        tensor output from the current layer.
    """
    # Perform symmetric padding on the feature dimension of time_step
    # This step is required to avoid issues when RNN output sequence is shorter
    # than the label length.
    # inputs = tf.pad(
    #     inputs,
    #     [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
    inputs = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides,
        padding="SAME", use_bias=False, activation=tf.nn.relu6)(inputs)

    return batch_norm()(inputs)


def _rnn_layer(inputs, rnn_cell, rnn_hidden_size, is_batch_norm, is_bidirectional):
    """Defines a batch normalization + rnn layer.

    Args:
        inputs: input tensors for the current layer.
        rnn_cell: RNN cell instance to use.
        rnn_hidden_size: an integer for the dimensionality of the rnn output space.
        layer_id: an integer for the index of current layer.
        is_batch_norm: a boolean specifying whether to perform batch normalization
          on input states.
        is_bidirectional: a boolean specifying whether the rnn layer is
          bi-directional.
        training: a boolean to indicate which stage we are in (training/eval).

    Returns:
        tensor output for the current layer.
    """
    if is_batch_norm:
        inputs = batch_norm()(inputs)

    if is_bidirectional:
        rnn_outputs = Bidirectional(rnn_cell(int(rnn_hidden_size/2),
                                             return_sequences=True))(inputs)
    else:
        rnn_outputs = rnn_cell(int(rnn_hidden_size),
                               return_sequences=True)(inputs)

    return rnn_outputs
