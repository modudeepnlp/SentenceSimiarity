import numpy as np

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from util import dssm_utils as utils

train, test = utils.build_dataset('./squad-v2.0.json')

pid, question, options, correct_indices, wrong_indices = train.iloc[40]
print('QUESTION', question, '\n')
for i, cand in enumerate(options):
    print(['[ ]', '[v]'][i in correct_indices], cand)

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow_hub as hub
tf.reset_default_graph()
sess = tf.InteractiveSession()

universal_sentence_encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder/2",
                                        trainable=False)
# consider as well:
# * lite: https://tfhub.dev/google/universal-sentence-encoder-lite/2
# * normal: https://tfhub.dev/google/universal-sentence-encoder/2
# * large: https://tfhub.dev/google/universal-sentence-encoder-large/2

sess.run([tf.global_variables_initializer(), tf.tables_initializer()]);

# tfhub implementation does tokenization for you
dummy_ph = tf.placeholder(tf.string, shape=[None])
dummy_vectors = universal_sentence_encoder(dummy_ph)

dummy_lines = [
    "How old are you?",                                                 # 0
    "In what mythology do two canines watch over the Chinvat Bridge?",  # 1
    "I'm sorry, okay, I'm not perfect, but I'm trying.",                # 2
    "What is your age?",                                                # 3
    "Beware, for I am fearless, and therefore powerful.",               # 4
]

dummy_vectors_np = sess.run(dummy_vectors, {
    dummy_ph: dummy_lines
})


plt.title('phrase similarity')
plt.imshow(dummy_vectors_np.dot(dummy_vectors_np.T), interpolation='none', cmap='gray')
plt.savefig("hi.png")


class Vectorizer:
	def __init__(self, output_size=256, hid_size=256, universal_sentence_encoder=universal_sentence_encoder):
		""" A small feedforward network on top of universal sentence encoder. 2-3 layers should be enough """
		self.universal_sentence_encoder = universal_sentence_encoder
		# define a few layers to be applied on top of u.s.e.
		# note: please make sure your final layer comes with _linear_ activation
		self.dense_1 = L.Dense(hid_size)
		self.dense_2 = L.Dense(output_size)
		self.dropout = L.Dropout(0.5)

	def __call__(self, input_phrases, is_train=True):
		"""
		Apply vectorizer. Use dropout and any other hacks at will.
		:param input_phrases: [batch_size] of tf.string
		:param is_train: if True, apply dropouts and other ops in train mode,
						 if False - evaluation mode
		:returns: predicted phrase vectors, [batch_size, output_size]
		"""
		x = self.universal_sentence_encoder(input_phrases)
		x = self.dropout(x)
		x = self.dense_1(x)
		x = self.dropout(x)
		x = self.dense_2(x)

		return x

question_vectorizer = Vectorizer()
answer_vectorizer = Vectorizer()

dummy_v_q = question_vectorizer(dummy_ph, is_train=True)
dummy_v_q_det = question_vectorizer(dummy_ph, is_train=False)
utils.initialize_uninitialized()

assert sess.run(dummy_v_q, {dummy_ph: dummy_lines}).shape == (5, 256)

assert np.allclose(
    sess.run(dummy_v_q_det, {dummy_ph: dummy_lines}),
    sess.run(dummy_v_q_det, {dummy_ph: dummy_lines})
), "make sure your model doesn't use dropout/noise or non-determinism if is_train=False"

print("Well done!")


import random

def iterate_minibatches(data, batch_size, shuffle=True, cycle=False):
    """
    Generates minibatches of triples: {questions, correct answers, wrong answers}
    If there are several wrong (or correct) answers, picks one at random.
    """
    indices = np.arange(len(data))
    while True:
        if shuffle:
            indices = np.random.permutation(indices)
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            batch = data.iloc[batch_indices]
            questions = batch['question'].values
            correct_answers = np.array([
                row['options'][random.choice(row['correct_indices'])]
                for i, row in batch.iterrows()
            ])
            wrong_answers = np.array([
                row['options'][random.choice(row['wrong_indices'])]
                for i, row in batch.iterrows()
            ])

            yield {
                'questions' : questions,
                'correct_answers': correct_answers,
                'wrong_answers': wrong_answers,
            }
        if not cycle:
            break

dummy_batch = next(iterate_minibatches(train.sample(3), 3))
print(dummy_batch)

def similarity(a, b):
    """ Dot product as a similarity function """
    cos_sim = L.dot([a,b], axes=1, normalize=False)
    # cos_sim = np.tensordot(a, b, axes=1)

    return cos_sim

def compute_loss(question_vectors, correct_answer_vectors, wrong_answer_vectors, delta=1.0):
    """
    Compute the triplet loss as per formula above.
    Use similarity function above for  sim[a, b]
    :param question_vectors: float32[batch_size, vector_size]
    :param correct_answer_vectors: float32[batch_size, vector_size]
    :param wrong_answer_vectors: float32[batch_size, vector_size]
    :returns: loss for every row in batch, float32[batch_size]
    Hint: DO NOT use tf.reduce_max, it's a wrong kind of maximum :)
    """
    correct_sim = similarity(question_vectors, correct_answer_vectors)
    wrong_sim = similarity(question_vectors, wrong_answer_vectors)

    print(correct_sim)
    print(wrong_sim)

    loss = tf.math.maximum(0.0, delta - correct_sim + wrong_sim)
    loss = tf.reshape(loss, [-1])

    return loss

dummy_v1 = tf.constant([[0.1, 0.2, -1], [-1.2, 0.6, 1.0]], dtype=tf.float32)
dummy_v2 = tf.constant([[0.9, 2.1, -6.6], [0.1, 0.8, -2.2]], dtype=tf.float32)
dummy_v3 = tf.constant([[-4.1, 0.1, 1.2], [0.3, -1, -2]], dtype=tf.float32)

# assert np.allclose(similarity(dummy_v1, dummy_v2).eval(), [7.11, -1.84])
# assert np.allclose(compute_loss(dummy_v1, dummy_v2, dummy_v3, delta=5.0).eval(), [0.0, 3.88])

placeholders = {
    key: tf.placeholder(tf.string, [None]) for key in dummy_batch.keys()
}

v_q = question_vectorizer(placeholders['questions'], is_train=True)
v_a_correct = answer_vectorizer(placeholders['correct_answers'], is_train=True)
v_a_wrong = answer_vectorizer(placeholders['wrong_answers'], is_train=True)

compute_loss(v_q, v_a_correct, v_a_wrong).eval()

loss = tf.reduce_mean(compute_loss(v_q, v_a_correct, v_a_wrong))
step = tf.train.AdamOptimizer().minimize(loss)

# we also compute recall: probability that a^+ is closer to q than a^-
test_v_q = question_vectorizer(placeholders['questions'], is_train=False)
test_v_a_correct = answer_vectorizer(placeholders['correct_answers'], is_train=False)
test_v_a_wrong = answer_vectorizer(placeholders['wrong_answers'], is_train=False)

correct_is_closer = tf.greater(similarity(test_v_q, test_v_a_correct),
                               similarity(test_v_q, test_v_a_wrong))
recall = tf.reduce_mean(tf.to_float(correct_is_closer))


import pandas as pd
from IPython.display import clear_output
from tqdm import tqdm

ewma = lambda x, span: pd.DataFrame({'x': x})['x'].ewm(span=span).mean().values
dev_batches = iterate_minibatches(test, batch_size=256, cycle=True)
loss_history = []
dev_recall_history = []
utils.initialize_uninitialized()

# infinite training loop. Stop it manually or implement early stopping

for batch in iterate_minibatches(train, batch_size=256, cycle=True):
	feed = {placeholders[key]: batch[key] for key in batch}
	loss_t, _ = sess.run([loss, step], feed)
	loss_history.append(loss_t)
	if len(loss_history) % 50 == 0:
		# measure dev recall = P(correct_is_closer_than_wrong | q, a+, a-)
		dev_batch = next(dev_batches)
		recall_t = sess.run(recall, {placeholders[key]: dev_batch[key] for key in dev_batch})
		dev_recall_history.append(recall_t)

	if len(loss_history) % 50 == 0:
		clear_output(True)
		plt.figure(figsize=[12, 6])
		plt.subplot(1, 2, 1), plt.title('train loss (hinge)'), plt.grid()
		plt.scatter(np.arange(len(loss_history)), loss_history, alpha=0.1)
		plt.plot(ewma(loss_history, span=100))
		plt.subplot(1, 2, 2), plt.title('dev recall (1 correct vs 1 wrong)'), plt.grid()
		dev_time = np.arange(1, len(dev_recall_history) + 1) * 100
		plt.scatter(dev_time, dev_recall_history, alpha=0.1)
		plt.plot(dev_time, ewma(dev_recall_history, span=10))
		plt.savefig("hi2.png")

print("Mean recall:", np.mean(dev_recall_history[-10:]))
assert np.mean(dev_recall_history[-10:]) > 0.85, "Please train for at least 85% recall on test set. "\
                                                  "You may need to change vectorizer model for that."
print("Well done!")






