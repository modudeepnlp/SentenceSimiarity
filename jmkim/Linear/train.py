import tensorflow as tf
import pickle

from pathlib import Path
from mecab import MeCab
from gluonnlp.data import PadSequence
from tqdm import tqdm
from absl import app
from model.net import Linear
from model.data import Corpus


def main():
    train_path = Path.cwd() / '..' / 'data_in' / 'train.txt'
    val_path = Path.cwd() / '..' / 'data_in' / 'val.txt'
    vocab_path = Path.cwd() / '..' / 'data_in' / 'vocab.pkl'

    length = 70
    batch_size = 1024
    learning_rate = 0.01
    epochs = 10

    with open(vocab_path, mode='rb') as io:
        vocab = pickle.load(io)

    train = tf.data.TextLineDataset(str(train_path)).shuffle(buffer_size=1000).batch(batch_size=batch_size,
                                                                                     drop_remainder=True)
    eval = tf.data.TextLineDataset(str(val_path)).batch(batch_size=batch_size, drop_remainder=True)

    tokenizer = MeCab()
    processing = Corpus(vocab, tokenizer)

    linear = Linear(vocab)

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    val_loss_metric = tf.keras.metrics.Mean(name='val_loss')
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')

    for epoch in range(epochs):
        train_loss_metric.reset_states()
        train_acc_metric.reset_states()
        val_loss_metric.reset_states()
        val_acc_metric.reset_states()
        tf.keras.backend.set_learning_phase(1)

        for step, val in tqdm(enumerate(train)):
            print(val)


if __name__ == '__main__':
    main()
