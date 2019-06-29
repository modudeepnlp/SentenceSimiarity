import os
import sys
import tensorflow as tf
from argparse import ArgumentParser

from HBMP.net import NLIModel
from util.data import get_data
import numpy as np

import time
from tqdm import tqdm
import tempfile

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # For TEST

parser = ArgumentParser(description='Helsinki NLI')
parser.add_argument("--corpus",
                    type=str,
                    choices=['snli', 'breaking_nli', 'multinli_matched', 'multinli_mismatched', 'scitail', 'all_nli'],
                    default='snli')
parser.add_argument('--epochs',
                    type=int,
                    default=50)
parser.add_argument('--batch_size',
                    type=int,
                    default=512)
parser.add_argument("--encoder_type",
                    type=str,
                    choices=['BiLSTMMaxPoolEncoder',
                             'LSTMEncoder',
                             'HBMP'],
                    default='BiLSTMMaxPoolEncoder')
parser.add_argument("--activation",
                    type=str,
                    choices=['tanh', 'relu', 'leakyrelu'],
                    default='relu')
parser.add_argument("--optimizer",
                    type=str,
                    choices=['rprop',
                             'adadelta',
                             'adagrad',
                             'rmsprop',
                             'adamax',
                             'asgd',
                             'adam',
                             'sgd'],
                    default='adam')
parser.add_argument('--embed_dim',
                    type=int,
                    default=300)
parser.add_argument('--fc_dim',
                    type=int,
                    default=600)
parser.add_argument('--hidden_dim',
                    type=int,
                    default=300)
parser.add_argument('--layers',
                    type=int,
                    default=1)
parser.add_argument('--dropout',
                    type=float,
                    default=0.2)
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.0005)
parser.add_argument('--lr_patience',
                    type=int,
                    default=1)
parser.add_argument('--lr_decay',
                    type=float,
                    default=0.99)
parser.add_argument('--lr_reduction_factor',
                    type=float,
                    default=0.2)
parser.add_argument('--weight_decay',
                    type=float,
                    default=0)
parser.add_argument('--preserve_case',
                    action='store_false',
                    dest='lower')
parser.add_argument('--word_embedding',
                    type=str,
                    default='glove.840B.300d')
parser.add_argument('--early_stopping_patience',
                    type=int,
                    default=3)
parser.add_argument('--save_path',
                    type=str,
                    default='results')
parser.add_argument('--seed',
                    type=int,
                    default=1234)
parser.add_argument('--max_len',
                    type=int,
                    default=42)
parser.add_argument('--use_glove',
                    type=str,
                    default=True)
parser.add_argument('--data_path',
                    type=str,
                    default='data/snli/')
parser.add_argument('--train_embedding',
                    type=str,
                    default=True)



config = parser.parse_args()

data_path = config.data_path

training = get_data(data_path + 'snli_1.0_train.jsonl')
validation = get_data(data_path + 'snli_1.0_dev.jsonl')
test = get_data(data_path + 'snli_1.0_test.jsonl')

tokenizer = Tokenizer(lower=False, filters='')
tokenizer.fit_on_texts(training[0] + training[1])
tokenizer.fit_on_texts(validation[0] + validation[1])

VOCAB = len(tokenizer.word_counts) + 1
LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}

to_seq = lambda X: pad_sequences(tokenizer.texts_to_sequences(X), maxlen=config.max_len)
prepare_data = lambda data: (to_seq(data[0]), to_seq(data[1]), data[2])

training = prepare_data(training)
validation = prepare_data(validation)
test = prepare_data(test)

print('Build model...')
print('Vocab size =', VOCAB)

config.vocab_size = VOCAB
config.out_dim = len(LABELS)

""" Load Glova Embedding """
GLOVE_STORE = data_path + 'precomputed_glove.weights'
if config.use_glove:
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
        embedding_matrix = np.zeros((VOCAB, config.embed_dim))
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
else:
    embedding_matrix=None

model = NLIModel(config, embedding_matrix)

model.compile(loss='categorical_crossentropy',
              optimizer=config.optimizer,
              metrics=['accuracy'])

print('Training')
_, tmpfn = tempfile.mkstemp()
# Save the best model during validation and bail out of training early if we're not improving
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=0, mode='auto')
model_ckpt = tf.keras.callbacks.ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)

callbacks = [early_stopping, model_ckpt]

# tr_dataset

history = model.fit(
    [training[0], training[1]],
    training[2],
    epochs=config.epochs,
    batch_size=config.batch_size,
    validation_data=([validation[0], validation[1]], validation[2]),
    callbacks=callbacks
	)

# Restore the best found model during validation
model.load_weights(tmpfn)

loss, acc = model.evaluate([test[0], test[1]], test[2], batch_size=config.batch_size)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


## TEST

# tr_dataset = tf.data.Dataset.from_tensor_slices((training[0], training[1], training[2])).shuffle(len(training[2]))
# # tr_dataset = tf.data.Dataset.from_tensor_slices((validation[0], validation[1], validation[2])).shuffle(len(validation))
# tr_dataset = tr_dataset.batch(config.batch_size, drop_remainder=True)

# tr_loss_metric = tf.keras.metrics.Mean(name='train_loss')
# # tr_acc_metric = tf.keras.metrics.CategoricalCrossentropy(name='train_accuracy')
# tr_acc_metric = tf.keras.metrics.SparseCategoricalCrossentropy(name='train_accuracy')

# opt = tf.optimizers.Adam(learning_rate = config.learning_rate)
# loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
# # loss_fn = tf.keras.backend.categorical_crossentropy
# # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model)
# manager = tf.train.CheckpointManager(ckpt, './data_out/tf_ckpts', max_to_keep=3)
# ckpt.restore(manager.latest_checkpoint)

# if manager.latest_checkpoint:
# 	print("Restored from {}".format(manager.latest_checkpoint))
# else:
# 	print("Initializing from scratch.")

# # from tensorflow.keras.utils import to_categorical

# for epoch in tqdm(range(config.epochs), desc='epochs'):

#     start = time.time()

#     tr_loss_metric.reset_states()
#     tr_acc_metric.reset_states()

#     tf.keras.backend.set_learning_phase(1)
#     tr_loss = 0

#     for step, tr in tqdm(enumerate(tr_dataset), desc='steps'):

#         x1_tr, x2_tr, y_tr = tr

#         # y_tr = tf.keras.utils.to_categorical(y_tr, len(LABELS))

#         with tf.GradientTape() as tape:
#             logits = model(x1_tr, x2_tr)
#             train_loss = loss_fn(y_tr, logits)
#         grads = tape.gradient(target=train_loss, sources=model.trainable_variables)
#         opt.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))

#         tr_loss_metric.update_state(train_loss)
#         tr_acc_metric.update_state(y_tr, logits)

#         if step % 10 == 0:

#             # a = tf.round(tf.nn.sigmoid(logits))
#             # b = to_categorical(y_tr, len(LABELS))

#             # correct_prediction = tf.equal(a, b)
#             # print(correct_prediction)
#             # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#             tr_loss /= (step+1)

#             tr_mean_loss = tr_loss_metric.result()
#             tr_mean_accuracy = tr_acc_metric.result()

#             template = 'Epoch {} Step {} Loss {:.4f} Acc {:.4f} Time {:.4f}'
#             print(template.format(epoch + 1, step, tr_mean_loss, tr_mean_accuracy, (time.time() - start)))
#             save_path = manager.save()

        # tr_loss = 0








# history = model.fit(
#     [train_data, train_data],
#     train_labels,
#     epochs=7,
#     batch_size=16,
#     validation_split=0.2,
# 	)

# callbacks=[early_stopping]
# test_loss, test_acc = classifier.evaluate(test_data, test_labels)




# import pickle
# import pandas as pd
#
# from tensorflow.keras import layers
# from sklearn.model_selection import train_test_split
# from model.data import Corpus
# from konlpy.tag import Mecab
# from pathlib import Path
# import json
# import pandas as pd
#
# proj_dir = Path.cwd()
# params = json.load((proj_dir / 'params' / 'config.json').open())
# train_path = params['filepath'].get('tr')
# val_path = params['filepath'].get('val')
# w2v_path = params['filepath'].get('vocab')
#
# with open(w2v_path, mode='rb') as io:
# 	w2v_vocab = pickle.load(io)
#
# tokenized = Mecab()
# processing = Corpus(vocab=w2v_vocab, tokenizer=tokenized)
#
#
# # Pad length를 통해 추가로 길이를 맞추어 주자
# # MAXLEN = 500
#
# M\