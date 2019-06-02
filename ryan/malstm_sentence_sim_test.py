import os
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import numpy as np

# from tensorflow.python.keras.models import Model, Sequential
# from tensorflow.python.keras.lslayers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout, Bidirectional


from util.utils import make_w2v_embeddings
from util.utils import split_and_zero_padding
from util.utils import ManDist

from glob import glob
from pathlib import Path
import json
import pandas as pd
from data_preprocessing import load_baloo_db

os.environ["CUDA_VISIBLE_DEVICES"]="6" #For TEST

proj_dir = Path.cwd()
params = json.load((proj_dir / 'params' / 'config.json').open())
train_path = params['filepath'].get('tr')
val_path = params['filepath'].get('val')
w2v_path = params['filepath'].get('vocab')

data_dir = "data/kor_1000_onehot/nlp=kmalemma/model=onehot_dssm"
q1, q2, label, q1_test, q2_test, label_test = load_baloo_db(data_dir)
# q1, q2, label = q1[:50000], q2[:50000], label[:50000]

# Make word2vec embeddings
MAXLEN = 15
EMB_DIM = 300
use_w2v = False

# Model variables
gpus = 1
batch_size = 16 * gpus
n_epoch = 50
n_hidden = 50

vocab_path = glob(os.path.join(data_dir, 'table=VOCABQIDCOUNT.text', 'part*'))[0]
with open(vocab_path, mode='rt', encoding='utf-8') as foo:
    VOC_SIZE = len(foo.readlines())

def pad_seq(query):
    return pad_sequences(query, padding='pre', truncating='post', maxlen=MAXLEN)

q1_set, q2_set, q1_test_set, q2_test_set = pad_seq(q1), pad_seq(q2), pad_seq(q1_test), pad_seq(q2_test)
label, label_test = np.asarray(label), np.asarray(label_test)



# Convert labels to their numpy representations

class MaLSTM(tf.keras.Model):

    def __init__(self, max_len, emb_dim, vocab_size):

        super(MaLSTM, self).__init__()

        self.MAX_LEN = max_len
        self.EMB_DIM = emb_dim
        self.VOC_SIZE = vocab_size

        self._embedding = layers.Embedding(self.VOC_SIZE, self.EMB_DIM, input_length=self.MAX_LEN)
        self._lstm = layers.LSTM(50)
        self._ma_dist = ManDist()
        self._cos_sim = layers.dot

    def call(self, x):

        print(x[0].shape)
        print(x[1].shape)

        q1_emb_layer = self._embedding(x[0])
        q2_emb_layer = self._embedding(x[1])

        print("######################")
        print(q1_emb_layer)
        print(q2_emb_layer)
        print("######################")

        q1_lstm = self._lstm(q1_emb_layer)
        q2_lstm = self._lstm(q2_emb_layer)

        malstm_dist = self._ma_dist([q1_lstm, q2_lstm])
        cos_sim = self._cos_sim([q1_lstm, q2_lstm], axes=-1, normalize=True)

        return cos_sim

sent_sim = MaLSTM(MAXLEN, EMB_DIM, VOC_SIZE)

lr = 1e-3
opt = tf.optimizers.Adam(learning_rate=lr)
loss_fn = tf.losses.mean_squared_error




















dataset = tf.data.Dataset.from_tensor_slices(())


tr_loss = 0
tf.keras.backend.set_learning_phase(1)
a = sent_sim([q1_set[:batch_size], q2_set[:batch_size]])
a
label[:batch_size]


with tf.GradientTape() as tape:
    mb_loss = loss_fn()

if gpus >= 2:
    # `multi_gpu_mode   l()` is a so quite buggy. it breaks the saved model.
    sent_sim = tf.keras.utils.multi_gpu_model(sent_sim, gpus=gpus)

sent_sim.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=0, mode='auto')

# Start trainings
training_start_time = time()

history = sent_sim.fit(
    [q1_set, q2_set], label,
    batch_size=batch_size, epochs=n_epoch,
    callbacks=[early_stopping],
    validation_data=([q1_test_set, q2_test_set], label_test)
)

training_end_time = time()
print("Training time finished.\n%d epochs in %12.2f" % (n_epoch,
                                                        training_end_time - training_start_time))

def plot_graphs(history, string):
	plt.plot(history.history[string])
	plt.plot(history.history['val_'+string])
	plt.xlabel("Epochs")
	plt.ylabel(string)
	plt.legend([string, 'val_'+string])
	plt.savefig(string + '_ma_lstm.png')

plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')

sent_sim.save_weights('./data/malstm', save_format='tf')


