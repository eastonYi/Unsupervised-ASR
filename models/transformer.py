import tensorflow as tf
import math
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout

from utils.tools import get_tensor_len


def Transformer(args):
    num_layers = args.model.G.num_layers
    d_model = args.model.G.d_model
    num_heads = args.model.G.num_heads
    dff = 4 * d_model
    rate = args.model.G.dropout_rate
    dim_output = args.dim_output

    input_x = Input(shape=[None, args.dim_input], name='encoder_input')
    input_decoder = Input(shape=[None], dtype=tf.int32, name='decoder_input')
    cache = Input(shape=[None, num_layers, d_model], name='cache')

    # create encoder and connect
    encoded = Encoder(num_layers, d_model, num_heads, dff, rate)(input_x)

    # create two decoders: one for training and one for forward
    decoder = Decoder(num_layers, d_model, num_heads, dff, dim_output, rate)
    decoded = decoder(input_decoder, encoded, cache=None)
    _decoded, cache_decoder = decoder(input_decoder, encoded, cache)

    fc = Dense(dim_output)
    logits = fc(decoded)
    _logits = fc(_decoded)

    pad_mask = tf.cast(input_decoder > 0, tf.float32)
    pad_mask = tf.tile(tf.expand_dims(pad_mask, -1), [1, 1, dim_output])
    logits *= pad_mask
    _logits *= pad_mask

    model = tf.keras.Model([input_x, input_decoder], logits, name='transformer')
    model_infer = tf.keras.Model([input_x, input_decoder, cache], [_logits, cache], name='transformer_cache')

    return model, model_infer


def Transformer_s1(args):
    num_layers = 6
    d_model = 512
    num_heads = 8
    dff = 4 * d_model
    rate = 0.0
    dim_output = args.dim_output

    input_x = Input(shape=[None, args.dim_input], dtype=tf.float32, name='encoder_input')
    input_decoder = Input(shape=[None], dtype=tf.int32, name='decoder_input')

    encoded = Encoder(num_layers, d_model, num_heads, dff, rate)(input_x)
    decoded = Decoder_s1(num_layers, d_model, num_heads, dff, dim_output, rate)(input_decoder, encoded)
    logits = Dense(dim_output, activation='linear')(decoded)

    pad_mask = tf.cast(input_decoder > 0, tf.float32)
    pad_mask = tf.tile(tf.expand_dims(pad_mask, -1), [1, 1, dim_output])
    logits *= pad_mask

    # _decoder_input_eos = tf.concat([input_decoder[:, 1:],
    #                                 tf.zeros([tf.shape(input_decoder)[0], 1], tf.int32)], 1)
    # logits = tf.one_hot(_decoder_input_eos, dim_output) * 10.0

    model = tf.keras.Model([input_x, input_decoder], logits, name='transformer')

    return model


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.fc = Dense(d_model, use_bias=False, activation='linear')
        self.layernorm = LayerNormalization(epsilon=1e-6)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training):

        len_x = get_tensor_len(x)

        x = self.fc(x)
        x = self.layernorm(x)
        x = add_timing_signal_1d(x)

        x = self.dropout(x, training=training)

        encoder_padding = tf.equal(tf.sequence_mask(len_x), False) # bool tensor
        mask = attention_bias_ignore_padding(encoder_padding)

        for i in range(self.num_layers):
            x = self.enc_layers[i](inputs=x, training=training, mask=mask)

        x *= tf.expand_dims(1.0 - tf.cast(encoder_padding, tf.float32), axis=-1)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        # self.pos_encoding = positional_encoding(d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, dec_input, enc_output, cache, training):
        seq_len = tf.shape(dec_input)[1]
        len_encoded = get_tensor_len(enc_output)
        encoder_padding = tf.equal(tf.sequence_mask(len_encoded, maxlen=tf.shape(enc_output)[1]), False) # bool tensor
        padding_mask = attention_bias_ignore_padding(encoder_padding)
        look_ahead_mask = attention_bias_lower_triangle(tf.shape(dec_input)[1])

        new_cache = []

        x = self.embedding(dec_input)  # (batch_size, target_seq_len, d_model)
        # x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](
                x, enc_output, training, look_ahead_mask, padding_mask)

            if cache is not None:
                decoder_output = tf.concat([cache[:, :, i, :], x], axis=1)
                new_cache.append(decoder_output[:, :, None, :])

        # x.shape == (batch_size, target_seq_len, d_model)
        if cache is not None:
            new_cache = tf.concat(new_cache, axis=2)
            return x, new_cache
        else:
            return x


class Decoder_s1(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, rate=0.1):
        super().__init__()

        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, dec_input, enc_output, training):
        len_encoded = get_tensor_len(enc_output)
        encoder_padding = tf.equal(tf.sequence_mask(len_encoded), False) # bool tensor
        padding_mask = attention_bias_ignore_padding(encoder_padding)
        look_ahead_mask = attention_bias_lower_triangle(tf.shape(dec_input)[1])

        x = self.embedding(dec_input)
        x = add_timing_signal_1d(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](
                inputs=x, enc_output=enc_output, training=training,
                look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)

        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training, mask):

        attn_output = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        self.dropout3 = Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3


class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.fc_pre = tf.keras.layers.Dense(d_model)
        self.pos_encoding = positional_encoding(self.d_model)

        self.enc_layers = [AttentionBlock(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        # x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x = self.fc_pre(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        attn_output = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)

        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output, attention_weights
    """
    # print(q.shape, k.shape, (tf.matmul(q, tf.transpose(k, [0,1,3,2]))).shape)
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


def get_angles(pos, i, d_model):
    angle_rates = 1 / tf.pow(10000, (2.0 * tf.cast(i//2, tf.float32)) / tf.cast(d_model, tf.float32))

    return tf.cast(pos, tf.float32) * angle_rates


def positional_encoding(d_model, position=10000):
    angle_rads = get_angles(tf.range(position)[:, None], tf.range(d_model)[None, :], d_model)

    # apply sin to even indices in the array; 2i
    sines = tf.math.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[None, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):

    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    position = tf.cast(tf.range(length), tf.float32)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.cast(num_timescales, tf.float32) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.math.sin(scaled_time), tf.math.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.math.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])

    return x + signal


def attention_bias_ignore_padding(memory_padding):
    """Create an bias tensor to be added to attention logits.

    Args:
    memory_padding: a boolean `Tensor` with shape [batch, memory_length].

    Returns:
    a `Tensor` with shape [batch, 1, 1, memory_length].
    """
    ret = tf.cast(memory_padding, tf.float32) * -1e9

    return tf.expand_dims(tf.expand_dims(ret, 1), 1)


def attention_bias_lower_triangle(size):
    """Create an bias tensor to be added to attention logits.

    Args:
    length: a Scalar.

    Returns:
    a `Tensor` with shape [1, 1, length, length].
    """
    # lower_triangle = tf.matrix_band_part(tf.ones([length, length]), -1, 0)
    # ret = -1e9 * (1.0 - lower_triangle)
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    mask = tf.reshape(-1e9 * mask, [1, 1, size, size])

    return mask
