import numpy as np
import tensorflow as tf
from keras import layers
import keras


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)[0]


def compute_mask(inputs, mask_value=-10000):
    return tf.math.reduce_all(tf.not_equal(inputs, mask_value), axis=2)


def compute_mask_for_multihead_attention(query, value, mask_value=-10000):
    query_mask = tf.cast(compute_mask(inputs=query, mask_value=mask_value), dtype="int32")
    value_mask = tf.cast(compute_mask(inputs=value, mask_value=mask_value), dtype="int32")

    mask = tf.cast(
        tf.map_fn(fn=lambda array: tf.tensordot(array[0], array[1], axes=0), elems=(query_mask, value_mask), fn_output_signature=tf.int32), dtype=bool
    )

    return mask


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, rate=0.01, add_value=True, add_query=True, add_key=False, mask_value=-10000, **kwargs):

        super(EncoderLayer, self).__init__()

        self.add_value = add_value
        self.add_query = add_query
        self.add_key = add_key
        self.mask_value = mask_value

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim, **kwargs)
        self.dense_proj = keras.Sequential([layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.supports_masking = True

    def call(self, query, value, key, training, mask):

        attention_output = self.attention(query=query, value=value, key=key, attention_mask=mask)

        input_sum = 0
        if self.add_query:
            input_sum += query
        if self.add_key:
            input_sum += key
        if self.add_value:
            input_sum += value

        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.layernorm_1(input_sum + attention_output)
        ffn_output = self.dense_proj(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm_2(out1 + ffn_output)


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        seq_length,
        num_layers,
        embed_dim,
        dense_dim,
        num_heads,
        rate=0.01,
        encode_value=True,
        encode_query=True,
        encode_key=True,
        add_value=True,
        add_query=True,
        add_key=False,
        mask_value=-10000,
        **kwargs
    ):
        super(Encoder, self).__init__()

        self.mask_value = mask_value
        self.encode_value = encode_value
        self.encode_query = encode_query
        self.encode_key = encode_key

        self.pos_enc = tf.cast(positional_encoding(position=seq_length, d_model=embed_dim), dtype=tf.float64)
        self.enc_layers = [
            EncoderLayer(embed_dim, dense_dim, num_heads, rate, add_value, add_query, add_key, mask_value, **kwargs) for _ in range(num_layers)
        ]

    def call(self, query, value, key, training, mask=None):

        attention_mask = compute_mask_for_multihead_attention(query=query, value=value, mask_value=self.mask_value)

        # adding embedding and position encoding.
        if self.encode_query:
            query = tf.cast(query, dtype=tf.float64) + self.pos_enc
        if self.encode_key:
            key += tf.cast(key, dtype=tf.float64) + self.pos_enc
        if self.encode_value:
            value += tf.cast(value, dtype=tf.float64) + self.pos_enc

        x = self.enc_layers[0](query, value, key, training, attention_mask)

        if len(self.enc_layers) > 1:
            for i in range(1, len(self.enc_layers)):
                x = self.enc_layers[i](x, x, x, training, attention_mask)

        return x  # (batch_size, input_seq_len, embed_dim)
