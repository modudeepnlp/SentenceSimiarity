import torch
import torch.nn as nn

class Ma_LSTM(nn.Module):
    def __init__(self, vocab_size):
        super(Ma_LSTM, self).__init__()

        self.embedding_dim = 100
        self.vocab_size = vocab_size
        self.hidden_size = 32
        self.stack = 1

        self.LSTM1 = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.stack, bidirectional=False)
        self.LSTM2 = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.stack, bidirectional=False)
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.batch = 32

        self.init_h_1 = torch.rand(self.stack, self.batch, self.hidden_size).cuda()
        self.init_c_1 = torch.rand(self.stack, self.batch, self.hidden_size).cuda()
        self.init_h_2 = torch.rand(self.stack, self.batch, self.hidden_size).cuda()
        self.init_c_2 = torch.rand(self.stack, self.batch, self.hidden_size).cuda()

        self.Dense1 = nn.Linear(self.hidden_size*self.hidden_size, 128)
        self.Dense2 = nn.Linear(128, 3)

        self.relu1 = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, sent1, sent2):
       
        look_up_1 = self.embedding(sent1)
        look_up_2 = self.embedding(sent2)

        out1, (h1, c1) = self.LSTM1(look_up_1, (self.init_h_1, self.init_c_1))
        out2, (h2, c2) = self.LSTM1(look_up_2, (self.init_h_2, self.init_c_2))
        
        a = out1[-1].view(self.batch, self.hidden_size, 1)
        b = out2[-1].view(self.batch, 1, self.hidden_size)

        similarity = a*b
        alignment = self.softmax(similarity)
        alignment = alignment * similarity

        fully_connect = alignment.view(self.batch, self.hidden_size*self.hidden_size)
        v = self.relu1(fully_connect)
        v = self.Dense1(v)
        v = self.relu1(v)
        logit = self.Dense2(v)

        return logit