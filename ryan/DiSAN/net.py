import tensorflow as tf
from tensorflow.keras import layers

from DiSAN.ops import *

def get_rep_mask(lengths, sentence_len, device):
	rep_mask = torch.FloatTensor(len(lengths), sentence_len).to(device)
	rep_mask.data.fill_(1)
	for i in range(len(lengths)):
		rep_mask[i, lengths[i]:] = 0

	return rep_mask

class NN4SNLI(tf.keras.Model):

	def __init__(self, config, train_embedding=None):
		super(NN4SNLI, self).__init__()

		self.config = config
		if config.use_glove == True:
			self.sentence_embedding = SentenceEmbedding(config, train_embedding)
		else:
			self.sentence_embedding = SentenceEmbedding(config)

		self.d_e = config.d_e
		self.d_h = config.d_h

		self.dropout = layers.Dropout(config.dropout)
		self.elu = tf.nn.elu

		# self.disan = DiSAN(config)
		self.disa = DiSA(config, direction='fw')

		self.fc = layers.Dense(config.d_h)
		self.fc_out = layers.Dense(config.out_dim)

		# init.xavier_uniform_(self.fc.weight.data)
		# init.constant_(self.fc.bias.data, 0)
		# init.xavier_uniform_(self.fc_out.weight.data)
		# init.constant_(self.fc_out.bias.data, 0)

	def call(self, x):

		prem = self.sentence_embedding(x[0])
		hypo = self.sentence_embedding(x[1])

		prem = self.disa(prem)



		# # Get representation masks for sentences of variable lengths
		# _, p_seq_len = prem.size()
		# _, h_seq_len = hypo.size()

		# print(p_seq_len)
		# print(h_seq_len)
		# print(prem.size())
		#
		# p_rep_mask = get_rep_mask(pre_length, p_seq_len)
		# # h_rep_mask = get_rep_mask(hypo_lengths, h_seq_len, self.device)
		# import sys
		# sys.exit(0)

		prem = self.disa(prem, direction='fw')




		print(prem)

		#
		# # Embedding
		# pre_x = self.word_emb(premise)
		# hypo_x = self.word_emb(hypothesis)
		#
		# # DiSAN
		# pre_s = self.disan(pre_x, p_rep_mask)
		# hypo_s = self.disan(hypo_x, h_rep_mask)
		#
		# # Concat
		s = tf.concat([prem, hypo, prem - hypo, prem * hypo], axis=-1)

		# Fully connected layer
		outs = self.elu(self.fc(self.dropout(s)))
		outs = self.fc_out(self.dropout(outs))

		return outs