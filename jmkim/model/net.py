import torch
import torch.nn as nn
import numpy as np


# class LSTM(nn.Module):
#     def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=2):
#         super(LSTM, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.batch_size = batch_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
#         self.linear = nn.Linear(self.hidden_dim, output_dim)
#
#     def init_hidden(self):
#         return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
#                 torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
#
#     def forward(self, input):
#         lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
#
#         y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
#         return y_pred.view(-1)


class Net(nn.Module):
    def __init__(self, vocab_len):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_len, 32)
        #self.lstm = nn.LSTM(input_dim, hidden_dim, batch_size)

        self.layer1 = nn.Linear(32, 32)
        self.layer2 = nn.Linear(32, 32)

        self.output = nn.Linear(32, 4)

    def forward(self, x, y):

        self._x = self.embedding(x)
        sentence1_ctx = self.layer1(self._x)
        sentence1_ctx = sentence1_ctx.mean(1)

        self._y = self.embedding(y)
        sentence2_ctx = self.layer2(self._y)
        sentence2_ctx = sentence2_ctx.mean(1)

        # self.left_lstm = self.lstm(self._x)
        # self.right_lstm = self.lstm(self._y)
        #
        #
        # print(torch.from_numpy(np.squeeze(np.exp(-1 * np.sum(np.absolute(np.subtract(self.left_lstm[0].data.numpy(), self.right_lstm[0].data.numpy())),
        #                   axis=0, keepdims=True)),axis=1)))
        # self.result = torch.from_numpy(np.squeeze(np.exp(-1 * np.sum(np.absolute(np.subtract(self.left_lstm[0].data.numpy(), self.right_lstm[0].data.numpy())),
        #                   axis=0, keepdims=True)),axis=1))
        
        self.result = self.output(sentence1_ctx - sentence2_ctx)
        return self.result
