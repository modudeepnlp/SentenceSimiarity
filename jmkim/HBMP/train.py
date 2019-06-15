import pickle
import tensorflow as tf

from pathlib import Path
from mecab import MeCab
from tqdm import tqdm
from model.net import HBMP
from model.data import Corpus
from config import FLAGS
from absl import app


def main(argv):
    train_path = Path.cwd() / '..' / 'data_in' / 'train.txt'
    val_path = Path.cwd() / '..' / 'data_in' / 'val.txt'
    vocab_path = Path.cwd() / '..' / 'data_in' / 'vocab.pkl'

    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    epochs = FLAGS.epochs

    with open(vocab_path, mode='rb') as io:
        vocab = pickle.load(io)

    train = tf.data.TextLineDataset(str(train_path)).shuffle(buffer_size=batch_size).batch(batch_size=batch_size,
                                                                                     drop_remainder=True)
    eval = tf.data.TextLineDataset(str(val_path)).batch(batch_size=batch_size, drop_remainder=True)

    tokenizer = MeCab()
    corpus = Corpus(vocab, tokenizer)

    hbmp = HBMP()

    opt = tf.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    '''
    loss, accuracy
    '''
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
            sen1, sen2, label = corpus.token2idx(val)
            with tf.GradientTape() as tape:
                logits = hbmp(sen1, sen2)
                train_loss = loss_fn(label, logits)
                print(logits)

            grads = tape.gradient(target=train_loss, sources=hbmp.trainable_variables)
            opt.apply_gradients(grads_and_vars=zip(grads, hbmp.trainable_variables))

            train_loss_metric.update_state(train_loss)
            train_acc_metric.update_state(label, logits)

        tr_loss = train_loss_metric.result()

        tqdm.write(
            'epoch : {}, tr_acc : {:.3f}%, tr_loss : {:.3f}'.format(epoch + 1,
                                                                    train_acc_metric.result() * 100,
                                                                    tr_loss))


if __name__ == "__main__":
    app.run(main)