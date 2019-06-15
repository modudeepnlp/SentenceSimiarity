import numpy as np

import matplotlib
matplotlib.use('TkAgg')
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
import sentencepiece as spm


tf.reset_default_graph()
sess = tf.InteractiveSession()

universal_sentence_encoder = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-lite/2",
                                        trainable=False)

x1 = tf.placeholder(tf.string, shape=[None])
x2 = tf.placeholder(tf.string, shape=[None])

universal_sentence_encoder(dict(x1), signature='spm_path', as_dict=True)


# consider as well:
# * lite: https://tfhub.dev/google/universal-sentence-encoder-lite/2
# * normal: https://tfhub.dev/google/universal-sentence-encoder/2
# * large: https://tfhub.dev/google/universal-sentence-encoder-large/2

sess.run([tf.global_variables_initializer(), tf.tables_initializer()]);

def process_to_IDs_in_sparse_format(sp, sentences):
	# An utility method that processes sentences with the sentence piece processor
	# 'sp' and returns the results in tf.SparseTensor-similar format:
	# (values, indices, dense_shape)
	ids = [sp.EncodeAsIds(x) for x in sentences]
	max_len = max(len(x) for x in ids)
	dense_shape=(len(ids), max_len)
	values=[item for sublist in ids for item in sublist]
	indices=[[row,col] for row in range(len(ids)) for col in range(len(ids[row]))]
	return (values, indices, dense_shape)

spm_path = sess.run(universal_sentence_encoder(signature="spm_path"))
sp = spm.SentencePieceProcessor()
sp.Load(spm_path)

input_placeholder = tf.sparse_placeholder(tf.int64, shape=[None, None])

embeddings = universal_sentence_encoder(
    inputs=dict(
        values=input_placeholder.values,
        indices=input_placeholder.indices,
        dense_shape=input_placeholder.dense_shape))


dummy_lines = [
    "How old are you?",                                                 # 0
    "In what mythology do two canines watch over the Chinvat Bridge?",  # 1
    "I'm sorry, okay, I'm not perfect, but I'm trying.",                # 2
    "What is your age?",                                                # 3
    "Beware, for I am fearless, and therefore powerful.",               # 4
]


values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, dummy_lines)

message_embeddings = sess.run(
      embeddings,
      feed_dict={input_placeholder.values: values,
                input_placeholder.indices: indices,
                input_placeholder.dense_shape: dense_shape})

print(message_embeddings)

plt.title('phrase similarity')
plt.imshow(message_embeddings.dot(message_embeddings.T), interpolation='none', cmap='gray')
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
		values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, input_phrases)
		message_embeddings = sess.run(
			embeddings,
			feed_dict={input_placeholder.values: values,
			           input_placeholder.indices: indices,
			           input_placeholder.dense_shape: dense_shape})

		# x = self.universal_sentence_encoder(input_phrases)
		x = self.dropout(message_embeddings)
		x = self.dense_1(x)
		x = self.dropout(x)
		x = self.dense_2(x)

		return x

question_vectorizer = Vectorizer()
answer_vectorizer = Vectorizer()

dummy_v_q = question_vectorizer(dummy_lines, is_train=True)
dummy_v_q_det = question_vectorizer(dummy_lines, is_train=False)
utils.initialize_uninitialized()

# def dummy_ph(dummy_lines):
# 	values, indices, dense_shape = process_to_IDs_in_sparse_format(sp, dummy_lines)
# 	message_embeddings = sess.run(
# 	      embeddings,
# 	      feed_dict={input_placeholder.values: values,
# 	                input_placeholder.indices: indices,
# 	                input_placeholder.dense_shape: dense_shape})
#
# 	return message_embeddings

assert dummy_v_q.eval().shape == (5, 256)

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

assert np.allclose(similarity(dummy_v1, dummy_v2).eval(), [7.11, -1.84])
assert np.allclose(compute_loss(dummy_v1, dummy_v2, dummy_v3, delta=5.0).eval(), [0.0, 3.88])

placeholders = {
    key: tf.placeholder(tf.string, [None]) for key in dummy_batch.keys()
}

placeholders['questions'].eval()

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
		plt.show()


print("Mean recall:", np.mean(dev_recall_history[-10:]))
assert np.mean(dev_recall_history[-10:]) > 0.85, "Please train for at least 85% recall on test set. "\
                                                  "You may need to change vectorizer model for that."
print("Well done!")



# optional: build tf graph required for select_best_answer
# <...>

# def select_best_answer(question, possible_answers):
#     """
#     Predicts which answer best fits the question
#     :param question: a single string containing a question
#     :param possible_answers: a list of strings containing possible answers
#     :returns: integer - the index of best answer in possible_answer
#     """
#     <YOUR CODE>
#     return <...>
#
#
# predicted_answers = [
#     select_best_answer(question, possible_answers)
#     for i, (question, possible_answers) in tqdm(test[['question', 'options']].iterrows(), total=len(test))
# ]
#
# accuracy = np.mean([
#     answer in correct_ix
#     for answer, correct_ix in zip(predicted_answers, test['correct_indices'].values)
# ])
# print("Accuracy: %0.5f" % accuracy)
# assert accuracy > 0.65, "we need more accuracy!"
# print("Great job!")
#
#
# def draw_results(question, possible_answers, predicted_index, correct_indices):
# 	print("Q:", question, end='\n\n')
# 	for i, answer in enumerate(possible_answers):
# 		print("#%i: %s %s" % (i, '[*]' if i == predicted_index else '[ ]', answer))
#
# 	print("\nVerdict:", "CORRECT" if predicted_index in correct_indices else "INCORRECT",
# 	      "(ref: %s)" % correct_indices, end='\n' * 3)
#
#
# for i in [1, 100, 1000, 2000, 3000, 4000, 5000]:
#     draw_results(test.iloc[i].question, test.iloc[i].options,
#                  predicted_answers[i], test.iloc[i].correct_indices)
#
#
# question = "What is my name?" # your question here!
# possible_answers = [
#     <...>
#     # ^- your options.
# ]
# predicted answer = select_best_answer(question, possible_answers)
#
# draw_results(question, possible_answers,
#              predicted_answer, [0])
