import torch
import torch.nn as nn
import config as config

class Sentence_Encoder(nn.Module):
    def __init__(self, vocab_size):
        super(Sentence_Encoder, self).__init__()
        stack = 1
        self.LSTM = nn.LSTM(config.embedding_dim, config.hidden_size, num_layers=stack, bidirectional=True)
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.init_h = torch.rand(stack*2, config.batch, config.hidden_size).cuda()
        self.init_c = torch.rand(stack*2, config.batch, config.hidden_size).cuda()


    def __max_pooling(self, out):
        temp_out = out.permute(1, 2, 0) # [seq, batch, dim] -> [batch, dim, seq = 5]
        seq_length = temp_out.shape[0]
        max_pool = nn.MaxPool1d(seq_length) 
        output = max_pool(temp_out) # [batch, dim, seq = 1]
        u = output.squeeze(2) # [batch, dim, seq = 1] -> [batch, dim]
        return u


    def forward(self, sentence):
        look_up = self.embedding(sentence)

        out1, (h1, c1) = self.LSTM(look_up, (self.init_h, self.init_c))
        u1 = self.__max_pooling(out1)

        out2, (h2, c2) = self.LSTM(look_up, (h1, c1))
        u2 = self.__max_pooling(out2)

        out3, (h3, c3) = self.LSTM(look_up, (h2, c2))
        u3 = self.__max_pooling(out3)

        u = torch.cat((u1, u2, u3), 1)
        return u

    
class Classifier(nn.Module):
    def __init__(self, vocab_size):
        super(Classifier, self).__init__()
        self.sent_enc_p = Sentence_Encoder(vocab_size)
        self.sent_enc_h = Sentence_Encoder(vocab_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True).cuda()
        self.dropout = nn.Dropout(p=config.dropout_keep_prob)

    def forward(self, premise, hypothesis):
        prem = self.sent_enc_p(premise)
        hypo = self.sent_enc_p(hypothesis)
        similarity = torch.cat([prem, hypo, torch.abs(prem-hypo), prem*hypo], 1).cuda()
        # print(similarity.shape) # [32, 768]
        vector = similarity.shape[1]
        
        dense = nn.Linear(vector, 128).cuda()
        fc = dense(similarity)
        act = self.leaky_relu(fc)
        drop = self.dropout(act)
        vector = drop.shape[1]

        dense = nn.Linear(vector, 128).cuda()
        fc = dense(drop)
        act = self.leaky_relu(fc)
        drop = self.dropout(act)
        vector = drop.shape[1]

        dense = nn.Linear(vector, 3).cuda()
        out = dense(drop)
        return out
        
        

