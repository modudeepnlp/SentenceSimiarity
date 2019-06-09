import os
import sys
import tensorflow as tf
from argparse import ArgumentParser
from model.net import NLIModel

parser = ArgumentParser(description='Helsinki NLI')
parser.add_argument("--corpus",
                    type=str,
                    choices=['snli', 'breaking_nli', 'multinli_matched', 'multinli_mismatched', 'scitail', 'all_nli'],
                    default='snli')
parser.add_argument('--epochs',
                    type=int,
                    default=20)
parser.add_argument('--batch_size',
                    type=int,
                    default=64)
parser.add_argument("--encoder_type",
                    type=str,
                    choices=['BiLSTMMaxPoolEncoder',
                             'LSTMEncoder',
                             'HBMP'],
                    default='LSTMEncoder')
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
                    default=600)
parser.add_argument('--layers',
                    type=int,
                    default=1)
parser.add_argument('--dropout',
                    type=float,
                    default=0.1)
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
                    default=10)


config = parser.parse_args()

imdb = tf.keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

train_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=config.max_len)

test_data = tf.keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=config.max_len)


config.embed_size = len(word_index)
config.out_dim = 2

model = NLIModel(config)

model.compile(loss='binary_crossentropy',
              optimizer=config.optimizer,
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')

train_data = train_data[:100]
train_labels = train_labels[:100]

history = model.fit(
    [train_data, train_data],
    train_labels,
    epochs=7,
    batch_size=16,
    validation_split=0.2,
	callbacks=[early_stopping])

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