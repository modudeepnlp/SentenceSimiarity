import os
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# from tensorflow.python.keras.models import Model, Sequential
# from tensorflow.python.keras.lslayers import Input, Embedding, LSTM, GRU, Conv1D, Conv2D, GlobalMaxPool1D, Dense, Dropout, Bidirectional

from util.utils import make_w2v_embeddings
from util.utils import split_and_zero_padding
from util.utils import ManDist

from pathlib import Path
import json
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"]="6" #For TEST

proj_dir = Path.cwd()
params = json.load((proj_dir / 'params' / 'config.json').open())
train_path = params['filepath'].get('tr')
val_path = params['filepath'].get('val')
w2v_path = params['filepath'].get('vocab')

# File paths
TRAIN_CSV = './data/quora-question-pairs/train.csv'
TEST_CSV = './data/quora-question-pairs/test.csv'

# Load training and test set
train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df[:10000]

for q in ['question1', 'question2']:
    train_df[q + '_n'] = train_df[q]

# Make word2vec embeddings
MAXLEN = 50
EMB_DIM = 300
use_w2v = False

train_df, embeddings = make_w2v_embeddings(train_df, embedding_dim=EMB_DIM, empty_w2v=not use_w2v)

# Model variables
gpus = 1
batch_size = 2048 * gpus
n_epoch = 50
n_hidden = 50

VOC_SIZE = len(embeddings)

# Split to train validation
validation_size = int(len(train_df) * 0.1)
training_size = len(train_df) - validation_size

X = train_df[['question1_n', 'question2_n']]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

X_train = split_and_zero_padding(X_train, MAXLEN)
X_validation = split_and_zero_padding(X_validation, MAXLEN)

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

X_train['left'].shape

# Make sure everything is ok
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)

class MaLSTM(tf.keras.Model):

    def __init__(self, max_len, emb_dim, vocab_size):

        super(MaLSTM, self).__init__()

        self.MAX_LEN = max_len
        self.EMB_DIM = emb_dim
        self.VOC_SIZE = vocab_size

        self._embedding = layers.Embedding(self.VOC_SIZE, self.EMB_DIM, input_length=self.MAX_LEN)
        self._lstm = layers.LSTM(50)
        self._ma_dist = ManDist()

    def call(self, x):

        print(x[0].shape)
        print(x[1].shape)

        q1_emb_layer = self._embedding(x[0])
        q2_emb_layer = self._embedding(x[1])

        print(q1_emb_layer.shape)
        print(q2_emb_layer.shape)

        q1_lstm = self._lstm(q1_emb_layer)
        q2_lstm = self._lstm(q2_emb_layer)

        malstm_dist = self._ma_dist([q1_lstm, q2_lstm])

        return malstm_dist

sent_sim = MaLSTM(MAXLEN, EMB_DIM, VOC_SIZE)

if gpus >= 2:
    # `multi_gpu_model()` is a so quite buggy. it breaks the saved model.
    sent_sim = tf.keras.utils.multi_gpu_model(sent_sim, gpus=gpus)

sent_sim.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=3,
                              verbose=0, mode='auto')

# Start trainings
training_start_time = time()

history = sent_sim.fit(
    [X_train['left'], X_train['right']], Y_train,
    batch_size=batch_size, epochs=n_epoch,
    callbacks=[early_stopping],
    validation_data=([X_validation['left'], X_validation['right']], Y_validation)
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


