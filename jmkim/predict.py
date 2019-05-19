import pandas as pd

import tensorflow as tf

from util import make_w2v_embeddings
from util import split_and_zero_padding
from util import ManDist

# File paths
TEST_CSV = './data_in/snli_1.0_test.txt'

gold_label = {}
gold_label['neutral'] = 0
gold_label['contradiction'] = 1
gold_label['entailment'] = 2

# Load training set
test_df = pd.read_csv(TEST_CSV,  sep='\t')
for q in ['sentence1', 'sentence2']:
    test_df[q + '_n'] = test_df[q]

# Make word2vec embeddings
embedding_dim = 300
max_seq_length = 20
test_df, embeddings = make_w2v_embeddings(test_df, embedding_dim=embedding_dim, empty_w2v=False)

# Split to dicts and append zero padding.
X_test = split_and_zero_padding(test_df, max_seq_length)

# Make sure everything is ok
assert X_test['left'].shape == X_test['right'].shape

# --

model = tf.keras.models.load_model('./data_out/SiameseLSTM.h5', custom_objects={'ManDist': ManDist})
model.summary()

prediction = model.predict([X_test['left'], X_test['right']])
print(prediction)