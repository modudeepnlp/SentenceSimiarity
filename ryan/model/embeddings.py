"""
Embedding

"""
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

class SentenceEmbedding(tf.keras.Model):

    def __init__(self, config):

        super(SentenceEmbedding, self).__init__()
        self.config = config
        self.word_embedding = layers.Embedding(config.embed_size, config.embed_dim)
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
        self.rnn = layers.LSTM(
                           units=config.hidden_dim,
                           dropout=config.dropout
                            )
        self.batch_norm = layers.BatchNormalization()

    def call(self, x):

        embedding = self.rnn(x)
        embedding = self.batch_norm(embedding)

        return embedding


# class BiLSTMMaxPoolEncoder(tf.keras.Model):
#     """
#     Bidirectional LSTM with max pooling
#     """
#     def __init__(self, config):
#         super(BiLSTMMaxPoolEncoder, self).__init__()
#         self.config = config
#         self.rnn1 = nn.LSTM(input_size=config.embed_dim,
#                            hidden_size=config.hidden_dim,
#                            num_layers=config.layers,
#                            dropout=config.dropout,
#                            bidirectional=True)
#
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#
#     def forward(self, inputs):
#         batch_size = inputs.size()[1]
#         h_0 = c_0 = Variable(inputs.data.new(self.config.cells,
#                                              batch_size,
#                                              self.config.hidden_dim).zero_())
#         embedding = self.rnn1(inputs, (h_0, c_0))[0]
#         # Max pooling
#         emb = self.max_pool(embedding.permute(1,2,0))
#         emb = emb.squeeze(2)
#         return emb


# class HBMP(nn.Module):
#     """
#     Hierarchical Bi-LSTM Max Pooling Encoder
#     """
#     def __init__(self, config):
#         super(HBMP, self).__init__()
#         self.config = config
#         self.max_pool = nn.AdaptiveMaxPool1d(1)
#         self.cells = config.cells
#         self.hidden_dim = config.hidden_dim
#         self.rnn1 = nn.LSTM(input_size=config.embed_dim,
#                             hidden_size=config.hidden_dim,
#                             num_layers=config.layers,
#                             dropout=config.dropout,
#                             bidirectional=True)
#         self.rnn2 = nn.LSTM(input_size=config.embed_dim,
#                             hidden_size=config.hidden_dim,
#                             num_layers=config.layers,
#                             dropout=config.dropout,
#                             bidirectional=True)
#         self.rnn3 = nn.LSTM(input_size=config.embed_dim,
#                             hidden_size=config.hidden_dim,
#                             num_layers=config.layers,
#                             dropout=config.dropout,
#                             bidirectional=True)
#
#
#     def forward(self, inputs):
#         batch_size = inputs.size()[1]
#         h_0 = c_0 = Variable(inputs.data.new(self.config.cells,
#                                              batch_size,
#                                              self.config.hidden_dim).zero_())
#         out1, (ht1, ct1) = self.rnn1(inputs, (h_0, c_0))
#         emb1 = self.max_pool(out1.permute(1,2,0)).permute(2,0,1)
#
#         out2, (ht2, ct2) = self.rnn2(inputs, (ht1, ct1))
#         emb2 = self.max_pool(out2.permute(1,2,0)).permute(2,0,1)
#
#         out3, (ht3, ct3) = self.rnn3(inputs, (ht2, ct2))
#         emb3 = self.max_pool(out3.permute(1,2,0)).permute(2,0,1)
#
#         emb = torch.cat([emb1, emb2, emb3], 2)
#         emb = emb.squeeze(0)
#
#         return emb