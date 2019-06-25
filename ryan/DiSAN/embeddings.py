"""
Embedding

"""
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class SentenceEmbedding(tf.keras.Model):

    def __init__(self, config, embedding_matrix=None):

        super(SentenceEmbedding, self).__init__()
        self.config = config
        if config.use_glove == True:
            self.word_embedding = layers.Embedding(config.vocab_size, config.embed_dim,
                                                   weights=[embedding_matrix], trainable=config.train_embedding)
        else:
            self.word_embedding = layers.Embedding(config.vocab_size, config.embed_dim)
        self.encoder = eval(config.encoder_type)(config)

    def call(self, x):
        sentence = self.word_embedding(x)
        embedding = self.encoder(sentence)

        return embedding

    def encode(self, x):
        embedding = self.encoder(x)
        return embedding

class LSTMEncoder(tf.keras.Model):
    """
    Basic LSTM Encoder
    """

    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.config = config
        self.rnn = layers.CuDNNLSTM(
                           units=config.hidden_dim,
                            )
        self.dropout = layers.Dropout(config.dropout)
        self.batch_norm = layers.BatchNormalization()

    def call(self, x):

        embedding = self.rnn(x)
        embedding = self.dropout(embedding)
        embedding = self.batch_norm(embedding)

        return embedding


class BiLSTMMaxPoolEncoder(tf.keras.Model):
    """
    Bidirectional LSTM with max pooling
    """
    def __init__(self, config):
        super(BiLSTMMaxPoolEncoder, self).__init__()
        self.config = config
        self.rnn = layers.CuDNNLSTM(
                           units=config.hidden_dim,
                            return_sequences=True
                            )
        self.bidirectional = layers.Bidirectional(self.rnn)
        self.dropout = layers.Dropout(config.dropout)

        self.max_pool = layers.GlobalMaxPool1D()

    def call(self, x):

        embedding = self.bidirectional(x)
        embedding = self.dropout(embedding)

        # Max pooling
        emb = self.max_pool(embedding)

        return emb


class HBMP(tf.keras.Model):
    """
    Hierarchical Bi-LSTM Max Pooling Encoder
    """
    def __init__(self, config):
        super(HBMP, self).__init__()
        self.config = config
        self.max_pool = layers.GlobalMaxPool1D()

        # self.cells = config.cells

        self.hidden_dim = config.hidden_dim
        self.rnn1 = layers.CuDNNLSTM(
	        units=config.hidden_dim,
            return_sequences=True
        )
        self.rnn2 = layers.CuDNNLSTM(
	        units=config.hidden_dim,
            return_sequences=True
        )
        self.rnn3 = layers.CuDNNLSTM(
	        units=config.hidden_dim,
            return_sequences=True
        )
        self.bidirectional_1 = layers.Bidirectional(self.rnn1)
        self.bidirectional_2 = layers.Bidirectional(self.rnn2)
        self.bidirectional_3 = layers.Bidirectional(self.rnn3)

    def call(self, x):

        emb1 = self.bidirectional_1(x)
        emb1 = self.max_pool(emb1)

        emb2 = self.bidirectional_2(x)
        emb2 = self.max_pool(emb2)

        emb3 = self.bidirectional_3(x)
        emb3 = self.max_pool(emb3)

        emb = tf.concat([emb1,emb2,emb3], axis=1)

        # emb = emb.squeeze(0)
        # emb = tf.squeeze(emb, axis=0)

        return emb