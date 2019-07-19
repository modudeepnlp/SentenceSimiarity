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

    def call(self, x):
        sentence = self.word_embedding(x)
        return sentence


def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask
    max_vec = tf.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = tf.exp(masked_vec, - max_vec)
    masked_exps = exps * mask
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros = (masked_sums == 0)
    masked_sums += zeros
    return masked_exps / (masked_sums + 1e-20)

def get_direct_mask_tile(direction, seq_len, device):
    """
    Todo
    :param direction:
    :param seq_len:
    :param device:
    :return:
    """

    if direction == 'fw':
        mask = layers.triu(mask, diagonal=1)
    elif direction == 'bw':
        mask = layers.tril(mask, diagonal=-1)
    else:
        raise NotImplementedError('only forward or backward mask is allowed!')
    mask.unsqueeze_(0)
    return mask

def get_rep_mask_tile(rep_mask):

    batch_size, seq_len = rep_mask.size()

    mask = rep_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)

    return mask



class Source2Token(tf.keras.Model):
    """
    Paper 3.2 f(x_i) = W^T * sigma(W(1)x_i + b(1)) + b

    """
    def __init__(self, d_h, dropout=0.2):
        super(Source2Token, self).__init__()

        self.d_h = d_h
        self.dropout_rate = dropout

        self.fc1 = layers.Dense(d_h)
        self.fc2 = layers.Dense(d_h)

        #TODO1: Xavier 등에대한 함수 확인

        self.elu = layers.elu()
        self.softmax = layers.Softmax(dim=2)
        self.dropout = layers.Dropout(dropout)

    def call(self, x, rep_mask):

        x = self.dropout(x)
        map1 = self.elu(self.fc1(x))
        map2 = self.fc2(self.dropout(map1))

        soft = masked_softmax(map2, rep_mask, dim=1)
        out = tf.sum(x * soft, dim=1)

        return out



class DiSA(tf.keras.Model):

    def __init__(self, config, direction):
        super(DiSA, self).__init__()

        self.d_e = config.d_e
        self.d_h = config.d_h
        self.direction = direction
        self.dropout_rate = config.dropout

        self.fc = layers.Dense(config.d_h)

        self.w_1 = layers.Dense(config.d_h)
        self.w_2 = layers.Dense(config.d_h)

        self.c = 5

        self.elu = tf.nn.elu
        self.softmax = layers.softmax(dim=-2)
        self.dropout = layers.Dropout(config.dropout)

    def call(self, x):
    # def call(self, x, rep_mask):

        # batch_size, seq_len, d_e = x.size()

        # Make diriectional mask
        # (batch, seq_len, seq_len)
        # rep_mask_tile = get_rep_mask_tile(rep_mask)
        # (1, seq_len, seq_len)
        # direct_mask_tile = get_direct_mask_tile(self.direction, seq_len)
        # (batch, seq_len, seq_len)
        # mask = rep_mask_tile * direct_mask_tile
        # (batch, seq_len, seq_len, 1)
        # mask.unsqueeze_(-1)

        # Transform the input seq to a seq of hidden (#14)
        x_dp = self.dropout(x)
        rep_map = self.elu(self.fc(x_dp))
        rep_map_tile = rep_map.unsqueeze(1).expand()
        rep_map_dp = self.dropout(rep_map)

        # Make logits
        # (batch, 1, seq_len, hid_dim)
        dependent_etd = self.w_1(rep_map_dp).unsqueeze(1)
        # (batch, seq_len, 1, hid_dim)
        head_etd = self.w_2(rep_map_dp).unsqueeze(2)

        # (batch, seq_len, seq_len, hid_dim)
        logits = self.c

















