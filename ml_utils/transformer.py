import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float64(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float64)[0]


def compute_mask(inputs, mask_value=-10000):
    return tf.math.reduce_all(tf.not_equal(inputs, mask_value), axis=2)


def compute_mask_for_multihead_attention(query, value, mask_value=-10000):
    query_mask = tf.cast(compute_mask(inputs=query, mask_value=mask_value), dtype="int32")
    value_mask = tf.cast(compute_mask(inputs=value, mask_value=mask_value), dtype="int32")

    mask = tf.cast(
        tf.map_fn(fn=lambda array: tf.tensordot(array[0], array[1], axes=0), elems=(query_mask, value_mask), fn_output_signature=tf.int32), dtype=bool
    )

    return mask


def compute_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return tf.cast(tf.equal(mask, 0), tf.int32)


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, key_dim, dense_dim, num_heads, output_shape=None, rate=0.01, add_value=True, add_query=True, add_key=False, **kwargs):

        super(EncoderLayer, self).__init__()

        self.key_dim = key_dim
        self.dense_dim = dense_dim
        self.num_head = num_heads
        self.ouptut_shape = output_shape
        self.rate = rate
        self.add_value = add_value
        self.add_query = add_query
        self.add_key = add_key

        if output_shape == None:
            output_shape = key_dim

        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, output_shape=output_shape, **kwargs)
        self.dense_proj = tf.keras.Sequential([tf.keras.layers.Dense(dense_dim, activation="relu"), tf.keras.layers.Dense(output_shape)])
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.supports_masking = True

    def call(self, query, value, key, training, mask, **kwargs):

        attention_output = self.attention(query=query, value=value, key=key, attention_mask=mask, **kwargs)
        inputs = list(zip([query, value, key], [self.add_query, self.add_value, self.add_key]))

        input_sum = tf.cast(tf.math.add_n([tf.cast(val[0], tf.float64) for val in inputs if val[1]]), tf.float64)

        attention_output = tf.cast(self.dropout1(attention_output, training=training), tf.float64)
        print(attention_output.dtype)
        print(input_sum.dtype)
        out1 = self.layernorm_1(input_sum + attention_output)
        ffn_output = self.dense_proj(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm_2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "dense_dim": self.dense_dim,
                "num_head": self.num_head,
                "ouptut_shape": self.ouptut_shape,
                "rate": self.rate,
                "add_value": self.add_value,
                "add_query": self.add_query,
                "add_key": self.add_key,
            }
        )
        return config


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        embed_dim: int,
        dense_dim: int,
        num_heads: int,
        rate: float = 0.01,
        key_dim: float = None,
        encode_value: bool = True,
        encode_query: bool = True,
        encode_key: bool = True,
        add_value: bool = True,
        add_query: bool = True,
        add_key: bool = False,
        value_vocab_size: int = None,
        query_vocab_size: int = None,
        key_vocab_size: int = None,
        mask_value: float = -10000,
        **kwargs,
    ):
        """Encoder found in transformers.

        Args:
            seq_length (int): max length of sequence. Used to generated 1st dimension of positional encoding.
            num_layers (int): # of encoder layers to use.
            embed_dim (int): the 2nd dimension of the positional encoding & output dim of multihead attention.
                Will be used to set key_dim if key_dim is not specified.
            dense_dim (int): dim of sandwiched dense layer in EncoderLayer.
            num_heads (int): num_heads in MultiAttentionHead.
            rate (float, optional): dropout rate for EncoderLayer. Defaults to 0.01.
            key_dim (float, optional): key_dim for keras's multihead attention layer. Basically, dimension of
                internal dense networks within the multihead attention layer.
            encode_value (bool, optional): whether to add positional encoding to value tensor. Defaults to True.
            encode_query (bool, optional): whether to add positional encoding to query tensor. Defaults to True.
            encode_key (bool, optional): whether to add positional encoding to key tensor. Defaults to True.
            add_value (bool, optional): whether to include value in the residual network in EncoderLayer. Defaults to True.
            add_query (bool, optional): wheter to include query in the residual network in EncoderLayer. Defaults to True.
            add_key (bool, optional): whether to include key in the residual network in EncoderLayer. Defaults to False.
            value_vocab_size (int, optional): vocabulary size of `value` tensor for embedding.
            query_vocab_size (int, optional): vocabulary size of `query` tensor for embedding.
            key_vocab_size (int, optional): vocabulary size of `key` tensor for embedding.
            mask_value (int, optional): mask value for padding masks. Defaults to -10000.
        """
        super(Encoder, self).__init__()

        self.seq_length = seq_length
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.rate: rate = 0.01
        self.key_dim = key_dim
        self.encode_value = encode_value
        self.encode_query = encode_query
        self.encode_key = encode_key
        self.add_value = add_value
        self.add_query = add_query
        self.add_key = add_key
        self.value_vocab_size = value_vocab_size
        self.query_vocab_size = query_vocab_size
        self.key_vocab_size = key_vocab_size
        self.mask_value = mask_value

        if key_dim == None:
            key_dim = embed_dim

        self.pos_enc = tf.cast(positional_encoding(position=seq_length, d_model=embed_dim), dtype=tf.float64)
        self.enc_layers = [
            EncoderLayer(
                key_dim=key_dim,
                dense_dim=dense_dim,
                num_heads=num_heads,
                rate=rate,
                add_value=add_value,
                add_query=add_query,
                add_key=add_key,
                output_shape=embed_dim,
                **kwargs,
            )
            for _ in range(num_layers)
        ]

        if value_vocab_size != None:
            self.query_embedding = tf.keras.layers.Embedding(value_vocab_size, embed_dim)
        if query_vocab_size != None:
            self.vocab_embedding = tf.keras.layers.Embedding(query_vocab_size, embed_dim)
        if key_vocab_size != None:
            self.key_embedding = tf.keras.layers.Embedding(key_vocab_size, embed_dim)

    def call(self, query, value, key, training, mask=None, **kwargs):

        if mask == None:
            mask = compute_mask_for_multihead_attention(query=query, value=value, mask_value=self.mask_value)

        # vocab embedding
        if hasattr(self, "query_embedding"):
            query = self.query_embedding(query)
        if hasattr(self, "vocab_embedding"):
            vocab = self.vocab_embedding(vocab)
        if hasattr(self, "key_embedding"):
            key = self.key_embedding(key)

        # position encoding.
        if self.encode_query:
            query = tf.cast(query, dtype=tf.float64) + self.pos_enc
        if self.encode_key:
            key = tf.cast(key, dtype=tf.float64) + self.pos_enc
        if self.encode_value:
            value = tf.cast(value, dtype=tf.float64) + self.pos_enc

        x = self.enc_layers[0](query, value, key, training, mask, **kwargs)

        if len(self.enc_layers) > 1:
            for i in range(1, len(self.enc_layers)):
                x = self.enc_layers[i](x, x, x, training, mask, **kwargs)

        return x  # (batch_size, input_seq_len, embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "seq_length": self.seq_length,
                "num_layers": self.num_layers,
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
                "rate": self.rate,
                "key_dim": self.key_dim,
                "encode_value": self.encode_value,
                "encode_query": self.encode_query,
                "encode_key": self.encode_key,
                "add_value": self.add_value,
                "add_query": self.add_query,
                "add_key": self.add_key,
                "value_vocab_size": self.value_vocab_size,
                "query_vocab_size": self.query_vocab_size,
                "key_vocab_size": self.key_vocab_size,
                "mask_value": self.mask_value,
            }
        )
        return config


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dense_dim, key_dim, rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__()

        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, **kwargs)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, **kwargs)

        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(dense_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)])

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, seq_length, num_layers, embed_dim, num_heads, dense_dim, target_vocab_size, rate=0.1):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, embed_dim)
        self.pos_encoding = positional_encoding(seq_length, embed_dim)

        self.dec_layers = [DecoderLayer(embed_dim=embed_dim, num_heads=num_heads, dense_dim=dense_dim, rate=rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f"decoder_layer{i+1}_block1"] = block1
            attention_weights[f"decoder_layer{i+1}_block2"] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights
