import tensorflow as tf
from tensorflow.keras import layers
from HBMP.embeddings import SentenceEmbedding

class FCClassifier(tf.keras.Model):
	"""
	Natural Language Inference with Hierarchical BiLSTM Max Pooling Architecture (HBMP)
	"""
	def __init__(self, config):
		super(FCClassifier, self).__init__()
		self.config = config

		self.seq_in_size = 4 * config.hidden_dim
		self.fc_dim = config.fc_dim
		self.out_dim = config.out_dim
		self.activation = config.activation # 'relu'
		self.dropout = config.dropout

		if self.config.encoder_type == 'BiLSTMMaxPoolEncoder':
			self.seq_in_size *= 2
		elif self.config.encoder_type == 'HBMP':
			self.seq_in_size *= 6

	def mlp(self, x):

		# print(x)
		# print(self.dropout)
		# print("##########")

		x = layers.Dropout(self.dropout)(x)
		x = layers.Dense(self.fc_dim, activation=self.activation)(x)
		x = layers.Dropout(self.dropout)(x)
		x = layers.Dense(self.fc_dim, activation=self.activation)(x)
		x = layers.Dense(self.out_dim, activation='softmax')(x)

		return x

	def call(self, prem, hypo):
		features = tf.concat([prem, hypo, tf.abs(prem-hypo), prem*hypo], 1)

		output = self.mlp(features)
		return output

class NLIModel(tf.keras.Model):
	"""
	NLI 테스크의 Main 모델
	"""
	def __init__(self, config):
		super(NLIModel, self).__init__()
		self.config = config
		self.sentence_embedding = SentenceEmbedding(config)
		self.classifier = FCClassifier(config)

	def call(self, x):

		prem = self.sentence_embedding(x[0])
		hypo = self.sentence_embedding(x[1])
		# prem = self.sentence_embedding(x1)
		# hypo = self.sentence_embedding(x2)
		answer = self.classifier(prem, hypo)

		return answer











