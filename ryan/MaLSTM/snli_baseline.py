"""
https://github.com/Smerity/keras_snli/blob/master/snli_rnn.py
"""

import json
import numpy as np
import tensorflow as tf
import tarfile
import tempfile
import json
import os
import re
import sys

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding

# for test
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Input, Dropout, TimeDistributed
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"]="7" #For TEST

data_path = 'data/snli/'
BATCH_SIZE = 2048
USE_GLOVE = True
TRAIN_EMBED = False
EMBED_HIDDEN_SIZE = 300
SENT_HIDDEN_SIZE = 300
MAX_EPOCHS = 42
MAX_LEN = 42
DP = 0.2
L2 = 4e-6
PATIENCE = 5
LAYERS = 3
# Summation of word embeddings
RNN = None
# RNN = GRU
# RNN = LSTM

ACTIVATION = 'relu'
OPTIMIZER = 'rmsprop'
print('RNN / Embed / Sent = {}, {}, {}'.format(RNN, EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE))
print('GloVe / Trainable Word Embeddings = {}, {}'.format(USE_GLOVE, TRAIN_EMBED))

def extract_tokens_from_binary_parse(parse):
	return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
	for i, line in enumerate(open(fn)):
		if limit and i > limit:
			break
		data = json.loads(line)
		label = data['gold_label']
		s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
		s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
		if skip_no_majority and label == '-':
			continue
		yield (label, s1, s2)

def get_data(fn, limit=None):
	raw_data = list(yield_examples(fn=fn, limit=limit))
	left = [s1 for _, s1, s2 in raw_data]
	right = [s2 for _, s1, s2 in raw_data]
	print(max(len(x.split()) for x in left))
	print(max(len(x.split()) for x in right))

	LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
	Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
	Y = to_categorical(Y, len(LABELS))

	return left, right, Y

training = get_data(data_path + 'snli_1.0_train.jsonl')
validation = get_data(data_path + 'snli_1.0_dev.jsonl')
test = get_data(data_path + 'snli_1.0_test.jsonl')

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])

# Lowest index from the tokenizer is 1 - we need to include 0 in our vocab count
VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=MAX_LEN)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

print('Build model...')
print('Vocab size =', VOCAB)

USE_GLOVE = False

GLOVE_STORE = data_path + 'precomputed_glove.weights'
if USE_GLOVE:
	if not os.path.exists(GLOVE_STORE + '.npy'):
		print('Computing GloVe')
  
	embeddings_index = {}
	f = open(data_path + 'glove.840B.300d.txt')
	# f = open(data_path + 'glove.6B.300d.txt')
	for line in f:
		values = line.split(' ')
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()
	
	# prepare embedding matrix
	embedding_matrix = np.zeros((VOCAB, EMBED_HIDDEN_SIZE))
	for word, i in tokenizer.word_index.items():
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			# words not found in embedding index will be all-zeros.
			embedding_matrix[i] = embedding_vector
		else:
			print('Missing from GloVe: {}'.format(word))
		
	np.save(GLOVE_STORE, embedding_matrix)

	print('Loading GloVe')
	embedding_matrix = np.load(GLOVE_STORE + '.npy')

	print('Total number of null word embeddings:')
	print(np.sum(np.sum(embedding_matrix, axis=1) == 0))

	embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, weights=[embedding_matrix], input_length=MAX_LEN, trainable=TRAIN_EMBED)
else:
	embed = Embedding(VOCAB, EMBED_HIDDEN_SIZE, input_length=MAX_LEN)

premise = Input(shape=(MAX_LEN,), dtype='int32')
hypothesis = Input(shape=(MAX_LEN,), dtype='int32')

rnn_kwargs = dict(output_dim=SENT_HIDDEN_SIZE, dropout_W=DP, dropout_U=DP)
SumEmbeddings = tf.keras.layers.Lambda(lambda x: K.sum(x, axis=1), output_shape=(SENT_HIDDEN_SIZE, ))

translate = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

prem = embed(premise)
hypo = embed(hypothesis)

prem = translate(prem)
hypo = translate(hypo)


if RNN and LAYERS > 1:
	for l in range(LAYERS - 1):
		rnn = RNN(return_sequences=True, **rnn_kwargs)
		prem = BatchNormalization()(rnn(prem))
		hypo = BatchNormalization()(rnn(hypo))

rnn = SumEmbeddings if not RNN else RNN(return_sequences=False, **rnn_kwargs)
prem = rnn(prem)
hypo = rnn(hypo)
prem = BatchNormalization()(prem)
hypo = BatchNormalization()(hypo)

joint = tf.keras.layers.concatenate([prem, hypo])
joint = Dropout(DP)(joint)
for i in range(3):
	joint = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION)(joint)
	joint = Dropout(DP)(joint)
	joint = BatchNormalization()(joint)

pred = Dense(len(LABELS), activation='softmax')(joint)
model = Model(inputs=[premise, hypothesis], outputs=pred)
model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print('Training')
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
callbacks = [EarlyStopping(patience=PATIENCE), ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)]
model.fit([training[0], training[1]], training[2], batch_size=BATCH_SIZE, nb_epoch=MAX_EPOCHS, validation_data=([validation[0], validation[1]], validation[2]), callbacks=callbacks)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))