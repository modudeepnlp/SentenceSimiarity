import torch
import torch.nn as nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

class Manhattan_LSTM(nn.Module):

	def __init__(self, batch_size, hidden_size, embedding, use_embedding=False, train_embedding=True):
		super(Manhattan_LSTM, self).__init__()
		self.batch_size = batch_size
		self.use_cuda = torch.cuda.is_available()
		self.hidden_size = hidden_size

		if use_embedding:
			self.embedding = nn.Embedding(embedding.shape[0], embedding.shape[1])
			self.embedding.weight = nn.Parameter(embedding)
			self.input_size = embedding.shape[1]  # V - Size of embedding vector

		else:
			self.embedding = nn.Embedding(embedding[0], embedding[1])
			self.input_size = embedding[1]

		self.embedding.weight.requires_grad = train_embedding

		self.lstm_1 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=False)
		self.lstm_2 = nn.LSTM(self.input_size, self.hidden_size, num_layers=1, bidirectional=False)

		self.Dense1 = nn.Linear(self.hidden_size * self.hidden_size, 128)
		self.Dense2 = nn.Linear(128, 3)

		self.relu1 = nn.ReLU()
		self.softmax = nn.Softmax()

	def exponent_neg_manhattan_distance(self, x1, x2):
		''' Helper function for the similarity estimate of the LSTMs outputs '''
		return torch.exp(-torch.sum(torch.abs(x1 - x2), dim=1))

	def forward(self, input, hidden):
		'''
		input           -> (2 x Max. Sequence Length (per batch) x Batch Size)
		hidden          -> (2 x Num. Layers * Num. Directions x Batch Size x Hidden Size)
		'''
		# print(input[0].shape) #21, 256
		# print(input[1].shape) #21, 256

		embedded_1 = self.embedding(input[0])  # L, B, V
		embedded_2 = self.embedding(input[1])  # L, B, V

		# print(embedded_1.shape)
		# print(embedded_2.shape)

		outputs_1, hidden_1 = self.lstm_1(embedded_1, (self.init_h, self.init_c))
		outputs_2, hidden_2 = self.lstm_2(embedded_2, (self.init_h, self.init_c))

		outputs_1 = outputs_1[-1].view(self.batch_size, self.hidden_size, 1)
		outputs_2 = outputs_2[-1].view(self.batch_size, 1, self.hidden_size)

		sim = outputs_1 * outputs_2
		alignment = self.softmax(sim)
		alignment = alignment * sim

		fully_connected = alignment.view(self.batch_size, self.hidden_size * self.hidden_size)
		v = self.relu1(fully_connected)
		v = self.Dense1(v)
		v = self.relu1(v)
		logits = self.Dense2(v)

		# similarity_scores = self.exponent_neg_manhattan_distance(outputs_1, outputs_2)

		# similarity_scores = self.exponent_neg_manhattan_distance(hidden_1[0].permute(1, 2, 0).view(batch_size, -1),
		#                                                          hidden_2[0].permute(1, 2, 0).view(batch_size, -1))

		return logits


	def init_weights(self):
		''' Initialize weights of lstm 1 '''
		for name_1, param_1 in self.lstm_1.named_parameters():
			if 'bias' in name_1:
				nn.init.constant_(param_1, 0.0)
			elif 'weight' in name_1:
				nn.init.xavier_normal_(param_1)

		''' Set weights of lstm 2 identical to lstm 1 '''
		lstm_1 = self.lstm_1.state_dict()
		lstm_2 = self.lstm_2.state_dict()

		for name_1, param_1 in lstm_1.items():
			# Backwards compatibility for serialized parameters.
			if isinstance(param_1, torch.nn.Parameter):
				param_1 = param_1.data

			lstm_2[name_1].copy_(param_1)

	def init_hidden(self, batch_size):
		# Hidden dimensionality : 2 (h_0, c_0) x Num. Layers * Num. Directions x Batch Size x Hidden Size
		# result = Variable(torch.zeros(2, 1, batch_size, self.hidden_size))
		self.init_h = Variable(torch.zeros(1, batch_size, self.hidden_size))
		self.init_c = Variable(torch.zeros(1, batch_size, self.hidden_size))

		if self.use_cuda:
			self.init_h = self.init_h.cuda()
			self.init_c = self.init_c.cuda()
			# result.cuda()

