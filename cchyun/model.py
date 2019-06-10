import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


# SNLI
class SNLI(nn.Module):
    def __init__(self, config):
        super(SNLI, self).__init__()

        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.encoder = HBMP(config)
        self.layout1 = nn.Linear(config.d_embed * 6 * 4, config.d_embed * 6 * 2)
        self.layout2 = nn.Linear(config.d_embed * 6 * 2, config.d_embed * 6 * 1)
        self.layout3 = nn.Linear(config.d_embed * 6 * 1, config.n_output)

    def forward(self, sentence1, sentence2):
        sentence1_embed = self.embed(sentence1)
        sentence1_ctx = self.encoder(sentence1_embed.permute(1, 0, 2))

        sentence2_embed = self.embed(sentence2)
        sentence2_ctx = self.encoder(sentence2_embed.permute(1, 0, 2))

        # output = sentence1_ctx - sentence2_ctx
        output = torch.cat([sentence1_ctx, sentence2_ctx, torch.abs(sentence1_ctx - sentence2_ctx), sentence1_ctx * sentence2_ctx], 1)
        output = self.layout1(output)
        output = self.layout2(output)
        output = self.layout3(output)
        return output


# Hierarchical BiLSTM Max Pooling
class HBMP(nn.Module):
    def __init__(self, config):
        super(HBMP, self).__init__()
        
        self.config = config
        
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.rnn1 = nn.LSTM(input_size=config.d_embed,
                            hidden_size=config.d_embed,
                            num_layers=config.n_layer,
                            dropout=config.dropout,
                            bidirectional=True)
        self.rnn2 = nn.LSTM(input_size=config.d_embed,
                            hidden_size=config.d_embed,
                            num_layers=config.n_layer,
                            dropout=config.dropout,
                            bidirectional=True)
        self.rnn3 = nn.LSTM(input_size=config.d_embed,
                            hidden_size=config.d_embed,
                            num_layers=config.n_layer,
                            dropout=config.dropout,
                            bidirectional=True)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        h_0 = c_0 = Variable(inputs.data.new(self.config.cells,
                                             batch_size,
                                             self.config.d_embed).zero_())
        out1, (ht1, ct1) = self.rnn1(inputs, (h_0, c_0))
        emb1 = self.max_pool(out1.permute(1,2,0)).permute(2,0,1)

        out2, (ht2, ct2) = self.rnn2(inputs, (ht1, ct1))
        emb2 = self.max_pool(out2.permute(1,2,0)).permute(2,0,1)

        out3, (ht3, ct3) = self.rnn3(inputs, (ht2, ct2))
        emb3 = self.max_pool(out3.permute(1,2,0)).permute(2,0,1)

        emb = torch.cat([emb1, emb2, emb3], 2)
        emb = emb.squeeze(0)

        return emb


"""
SNLI Configuration
"""
class SNLIConfig(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

