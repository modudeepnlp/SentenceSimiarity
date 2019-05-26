import numpy as np
import sys
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
# from config import Config
# config = Config()
import random
from itertools import combinations, compress, chain, permutations
# import config_test


def embed_collate_function_factory(max_len, emb_vector):
	def embed_collate_function(batch):
		PAD_TOKEN, PAD_INDEX = '[PAD]', 0

		sorted_batch = sorted(batch, key=lambda text_label: len(text_label[0]), reverse=True)
		count= len(sorted_batch)

		q1_len = [len(q1) for q1, q2, label in sorted_batch]
		q2_len = [len(q2) for q1, q2, label in sorted_batch]

		q1_longest, q2_longest = q1_len[0], q2_len[0]

		q1_batch = []
		q2_batch = []
		label_batch = []

		for q1, q2, label in batch:
			q1_length, q2_length = len(q1), len(q2)

			q1_padding = [PAD_INDEX] * (max_len - q1_length)
			q2_padding = [PAD_INDEX] * (max_len - q2_length)

			q1 = list(map(int, q1))
			q2 = list(map(int, q2))

			q1_pad_seq, q2_pad_seq = (q1 + q1_padding), (q2 + q2_padding)
			q1_pad_seq, q2_pad_seq = q1_pad_seq[:max_len], q2_pad_seq[:max_len]

			# input embed stuff
			q1_pad_seq = emb_vector[q1_pad_seq]
			q2_pad_seq = emb_vector[q2_pad_seq]

			# q1_pad_seq = pad_sequences(q1, maxlen=max_len, padding='post')
			# q2_pad_seq = pad_sequences(q2, maxlen=max_len, padding='post')

			q1_batch.append(q1_pad_seq)
			q2_batch.append(q2_pad_seq)
			label_batch.append(label)

		label_batch = np.reshape(label_batch, (-1, 1))

		return (q1_batch, q2_batch, q1_len, q2_len), label_batch

	return embed_collate_function

def onehot_collate_function_factory(max_len):
	def onehot_collate_function(batch):
		PAD_TOKEN, PAD_INDEX = '[PAD]', 0

		sorted_batch = sorted(batch, key=lambda text_label: len(text_label[0]), reverse=True)
		count= len(sorted_batch)

		q1_len = [len(q1) for q1, q2, label in sorted_batch]
		q2_len = [len(q2) for q1, q2, label in sorted_batch]

		q1_longest, q2_longest = q1_len[0], q2_len[0]

		q1_batch = []
		q2_batch = []
		label_batch = []

		for q1, q2, label in batch:
			q1_length, q2_length = len(q1), len(q2)

			q1_padding = [PAD_INDEX] * (max_len - q1_length)
			q2_padding = [PAD_INDEX] * (max_len - q2_length)

			q1 = list(map(int, q1))
			q2 = list(map(int, q2))

			q1_pad_seq, q2_pad_seq = (q1 + q1_padding), (q2 + q2_padding)
			q1_pad_seq, q2_pad_seq = q1_pad_seq[:max_len], q2_pad_seq[:max_len]

			# q1_pad_seq = pad_sequences(q1, maxlen=max_len, padding='post')
			# q2_pad_seq = pad_sequences(q2, maxlen=max_len, padding='post')

			q1_batch.append(q1_pad_seq)
			q2_batch.append(q2_pad_seq)
			label_batch.append(label)

		label_batch = np.reshape(label_batch, (-1, 1))

		return (q1_batch, q2_batch, q1_len, q2_len), label_batch

	return onehot_collate_function

def elmo_collate_function_factory(max_len):
	def elmo_collate_function(batch):
		PAD_TOKEN, PAD_INDEX = '[PAD]', 0

		sorted_batch = sorted(batch, key=lambda text_label: len(text_label[0]), reverse=True)
		count= len(sorted_batch)

		q1_len = [len(q1) for q1, q2, label in sorted_batch]
		q2_len = [len(q2) for q1, q2, label in sorted_batch]

		q1_longest, q2_longest = q1_len[0], q2_len[0]

		q1_batch = []
		q2_batch = []
		label_batch = []

		for q1, q2, label in batch:

			q1_batch.append(q1)
			q2_batch.append(q2)
			label_batch.append(label)

		label_batch = np.reshape(label_batch, (-1, 1))

		return (q1_batch, q2_batch, q1_len, q2_len), label_batch

	return elmo_collate_function

class QQDataset:

	def __init__(self, pos_len, neg_len, q1_data, q2_data, labels):
		self.data = []
		for q1, q2, label in zip(q1_data, q2_data, labels):
			self.data.append((q1, q2, label))

		self.label_counter = {0:neg_len, 1:pos_len}

	def __getitem__(self, item):
#         print("get_item {}".format(item))
		return self.data[item]

	def __len__(self):
		return len(self.data)

	@property
	def sample_weights(self):
		total_size = len(self)
		print(total_size)

		return [total_size / self.label_counter[label] for q1, q2, label in self.data]

def collate_function(batch):
#     print('batch', batch)
	q1_matrix = []
	q2_matrix = []
	label_matrix = []

	for q1, q2, label in batch:
		q1_matrix.append(q1)
		q2_matrix.append(q2)
		label_matrix.append(label)

	return np.stack(q1_matrix), np.stack(q2_matrix), np.stack(label_matrix)


def read_dataset(train_data_path):

	print("#### read_train_set start ####")

	with open(train_data_path, mode='rt', encoding='utf-8') as fhu:

		pos_q_set, pos_sim_set, pos_label_set = [], [], []
		neg_q_set, neg_sim_set, neg_label_set = [], [], []

		user_utt = fhu.readline()
		counter = 0

		while user_utt:
			counter += 1
			if counter % 10000 == 0:
				print("  reading %s, line %d" % (train_data_path, counter))
				sys.stdout.flush()

			qids, sim_qids, labels = user_utt.split('\u241D')

			if int(labels.replace('\n', '')) == 1:
				pos_q_set.append(qids.split("\u241E"))
				pos_sim_set.append(sim_qids.split("\u241E"))
				pos_label_set.append(labels.replace('\n', ''))
			else:
				neg_q_set.append(qids.split("\u241E"))
				neg_sim_set.append(sim_qids.split("\u241E"))
				neg_label_set.append(labels.replace('\n', ''))

			user_utt = fhu.readline()

		print("#### read_train_set done ####")
		rand_pick_num = 1
		pos_q_set, pos_sim_set, pos_label_set = pos_q_set * rand_pick_num, pos_sim_set * rand_pick_num, pos_label_set * rand_pick_num

		q_set = pos_q_set + neg_q_set
		sim_set = pos_sim_set + neg_sim_set
		label_set = pos_label_set + neg_label_set
		label_set = list(map(int, label_set))

		pos_len = len(pos_label_set)
		neg_len = len(neg_label_set)

		print("========= positive num data: {}, negative num data: {} ==========".format(pos_len, neg_len))
		print("========= total num data: {} ==========".format(len(label_set)))

		return q_set, sim_set, label_set, pos_len, neg_len


def read_train_set(train_data_path):

	print("#### read_train_set start ####")

	with open(train_data_path, mode='rt', encoding='utf-8') as fhu:

		pos_q_set, pos_sim_set, pos_label_set = [], [], []
		neg_q_set, neg_sim_set, neg_label_set = [], [], []

		user_utt = fhu.readline()
		counter = 0

		while user_utt:
			counter += 1
			if counter % 10000 == 0:
				print("  reading %s, line %d" % (train_data_path, counter))
				sys.stdout.flush()

			qids, sim_qids, labels = user_utt.split('\u241D')

			if int(labels.replace('\n', '')) == 1:
				pos_q_set.append(qids.split("\u241E"))
				pos_sim_set.append(sim_qids.split("\u241E"))
				pos_label_set.append(labels.replace('\n', ''))
			else:
				neg_q_set.append(qids.split("\u241E"))
				neg_sim_set.append(sim_qids.split("\u241E"))
				neg_label_set.append(labels.replace('\n', ''))

			user_utt = fhu.readline()

		print("#### read_train_set done ####")
		rand_pick_num = 4
		pos_q_set, pos_sim_set, pos_label_set = pos_q_set * rand_pick_num, pos_sim_set * rand_pick_num, pos_label_set * rand_pick_num

		q_set = pos_q_set + neg_q_set
		sim_set = pos_sim_set + neg_sim_set
		label_set = pos_label_set + neg_label_set

		print("========= positive num data: {}, negative num data: {} ==========".format(len(pos_label_set), len(neg_label_set)))
		print("========= total num data: {} ==========".format(len(label_set)))

		return q_set, sim_set, label_set


def get_batch(q_features, sim_q_features, labels, batch_size, num_epochs, max_document_length,
		   token_to_embedding=None, onehot=False):

	print("#### get_batch start ####")

	data_size = len(labels)
	num_batches_per_epoch = int((len(labels) - 1) / batch_size) + 1

	len_q_input = []
	len_sim_q_input = []

	for epoch in range(num_epochs):
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)

			batch_q_features_raw = q_features[start_index:end_index]
			batch_q_features = np.zeros((end_index - start_index, max_document_length), dtype=int)

			batch_sim_q_features_raw = sim_q_features[start_index:end_index]
			batch_sim_q_features = np.zeros((end_index - start_index, max_document_length), dtype=int)

			for i, batch_q_feature_raw in enumerate(batch_q_features_raw):

				# splited_batch_q_feature_raw = batch_q_feature_raw.split()
				current_max_length = min(len(batch_q_feature_raw), max_document_length)
				len_q_input.append(current_max_length)

				for j in range(current_max_length):
					batch_q_features[i, j] = int(batch_q_feature_raw[j])

			for i, batch_sim_q_feature_raw in enumerate(batch_sim_q_features_raw):

				# splited_batch_sim_q_feature_raw = batch_sim_q_feature_raw.split()
				current_max_length = min(len(batch_sim_q_feature_raw), max_document_length)
				len_sim_q_input.append(current_max_length)

				for j in range(current_max_length):
					batch_sim_q_features[i, j] = int(batch_sim_q_feature_raw[j])

			batch_labels_raw = labels[start_index:end_index]
			batch_labels = np.array(batch_labels_raw, dtype=int)
			# print(batch_labels)
			batch_labels = np.reshape(batch_labels, (-1, 1))  # rank변환
			# print(batch_labels)

			if onehot:
				print("onehot")
			else:
				# Convert to emb_seq
				def id_to_emb(batch_id):
					batch_emb = []
					for batch_seq in batch_id:
						# Tapi Vector가 없는 경우 방어 로직
						embed_seq = [token_to_embedding[e] for e in batch_seq]
						if sum(embed_seq[0]) == 0:
							# print(token_to_embedding[3])
							print("Embedding value error")
							# print(embed_seq)

						batch_emb.append(embed_seq)

					return batch_emb

				""" Embedding Vector """

				batch_q_features = id_to_emb(batch_q_features)
				batch_sim_q_features = id_to_emb(batch_sim_q_features)

			return batch_q_features, batch_sim_q_features, batch_labels, len_q_input, len_sim_q_input

def get_test_batch(q_features, sim_q_features, max_document_length):
	batch_q_features = np.zeros((1, max_document_length), dtype=int)
	for idx, id in enumerate(q_features):
		if idx < max_document_length:
			batch_q_features[0, idx] = int(id)
	batch_sim_q_features = np.zeros((1, max_document_length), dtype=int)
	for idx, id in enumerate(sim_q_features):
		if idx < max_document_length:
			batch_sim_q_features[0, idx] = int(id)
	return batch_q_features, batch_sim_q_features

# Load Embedding Dataset
def load_embeddings(vocab_path):
	vocab_tapi = []
	unk_vec = []

	with open(vocab_path, mode="rt", encoding="utf-8") as fh:
		tokens = fh.readline()
		while tokens:
			tokens = tokens.replace("\n", "")
			token_list = tokens.split("\u241D")
			tid = int(token_list[0])
			if token_list[1] == '':
				tapi_vec = []
			else:
				tapi_vec = [float(e) for e in token_list[1].split("\u241E")]
			if tid == 3:
				unk_vec = tapi_vec

			vocab_tapi.append([tid, tapi_vec])
			# id_embedding.append((int(token_list[0]), tapi_vec))
			# print(unk_vec)
			tokens = fh.readline()

		print("total embedding vocab size: {}".format(len(vocab_tapi)))

		vocab_tapi2 = [unk_vec for _ in range(len(vocab_tapi))]
		for (id, vector) in vocab_tapi:
			if len(vector) > 0:
				vocab_tapi2[id] = vector

		return vocab_tapi2

# def load_embeddings(vocab_path, emb_size):
#     id_embedding = []
#     with open(vocab_path, mode="rt", encoding="utf-8") as f:
#         word_list = f.readline()

#         while word_list:
#             word = word_list.replace("\n", "").split("\u241D")
#             # print(len(word[1]))
#             if word[1] == '':
#                 rand_vec = np.random.uniform(-1, 1, emb_size)
#                 embed_vec2 = [float(e) for e in rand_vec]
#             else:
#                 embed_vec = word[1].split("\u241E")
#                 embed_vec2 = [float(e) for e in embed_vec]
#             id_embedding.append((int(word[0]), embed_vec2))
#             word_list = f.readline()

			#     rand_vec = np.random.uniform(-1, 1, emb_size)
			#     embed_vec2 = [float(e) for e in rand_vec]
			# else:
			#     embed_vec = word[1].split("\u241E")
			#     embed_vec2 = [float(e) for e in embed_vec]
			# id_embedding.append((int(word[0]), embed_vec2))
			# word_list = f.readline()
	# temp_matrix = sorted(id_embedding, key=lambda x: x[0], reverse=False)
	# embed_matrix = [vector for idx, vector in temp_matrix]

	# return dict(id_embedding), embed_matrix

def read_cid(data_path):
	with open(data_path, mode='rt', encoding='utf-8') as fhu:

		gid_set = []
		qid_set = []

		user_utt = fhu.readline()
		counter = 0

		while user_utt:
			counter += 1
			if counter % 5000 == 0:
				print("  reading %s, line %d" % (data_path, counter))
				sys.stdout.flush()

			# label_set.append(labels.replace('\n',''))

			gid, qid = user_utt.split('\u241D')
			gid_set.append(gid.split("\u241E"))
			qid_set.append(qid.replace('\n', '').split("\u241E"))

			user_utt = fhu.readline()

		return gid_set, qid_set


def get_test_emb(q_features, token_to_embedding=None, max_document_length=10, onehot=False):
	np_q_features = np.zeros((len(q_features), max_document_length), dtype=int)

	for i, test in enumerate(q_features):
		for idx, id in enumerate(test):
			if idx < max_document_length:
				np_q_features[i, idx] = int(id)
		# np_q_features[0, idx] = int(id)

	def id_to_emb(batch_id):
		batch_emb = []
		for batch_seq in batch_id:
			# Tapi Vector가 없는 경우 방어 로직
			embed_seq = []
			# embed_seq = [token_to_embedding[e] for e in batch_seq]
			# batch_emb.append(embed_seq)
			for e in batch_seq:
				try:
					emb_id = token_to_embedding[e]
					embed_seq.append(emb_id)
				except Exception as e:
					# print("No tapi embedding {}".format(e))
					emb_id = token_to_embedding[3]
					embed_seq.append(emb_id)

			if sum(embed_seq[0]) == 0:
				print("error {}".format(sum(embed_seq[0])))
				sys.exit("Zero Value Exist")

			batch_emb.append(embed_seq)

		return batch_emb

	emb_mat = id_to_emb(np_q_features)

	return emb_mat


def read_query_ids(data_path):
	# Query값을 idx 로드한다.
	with open(data_path, mode='rt', encoding='utf-8') as fhu:

		gid_set = []
		qid_set = []

		user_utt = fhu.readline()
		counter = 0

		while user_utt:
			counter += 1
			if counter % 1000 == 0:
				print("  reading %s, line %d" % (data_path, counter))
				sys.stdout.flush()
			try:
				gids, qids = user_utt.split('\u241D')
			except:
				print(qids)
			gid_set.append(gids.split("\u241E"))
			qid_set.append(qids.replace('\n', '').split("\u241E"))

			user_utt = fhu.readline()

		return gid_set, qid_set

def load_group_map(g2q_map_path):
	group_map = []
	with open(g2q_map_path, mode="rt", encoding="utf-8") as fh:
		group_query = fh.readline()
		while group_query:
			group_query = group_query.replace("\n", "")
			group_query_id = group_query.split("\u241D")
			group_id = int(group_query_id[0])
			query_id_list = [int(x) for x in group_query_id[1].split("\u241E")]
			answer_id_list = [int(x) for x in group_query_id[2].split("\u241E")]

			answer_id_list = [int(x) for x in group_query_id[2].split("\u241E")]
			group_map.append([group_id, query_id_list, answer_id_list])

			# group_map.append([group_id, query_id_list])

			group_query = fh.readline()
	return group_map

def _read_tokens(token_user_ids_path, max_seq_len):
	token_ids_set = []
	token_len_set = []

	with open(token_user_ids_path, mode="rt", encoding="utf-8") as fhu:
		user_utt = fhu.readline()
		counter = 0
		while user_utt:
			counter += 1
			if counter % 10000 == 0:
				print("  reading %s, line %d" % (token_user_ids_path, counter))
				sys.stdout.flush()

			user_utt = user_utt.replace("\n", "")
			source_ids = user_utt.split("\u241D")
			cid = int(source_ids[0])
			token_seq = source_ids[1].split("\u241E")
			token_len = len(token_seq)
			
			# q1_batch = []
			# q2_batch = []
			# label_batch = []
			#
			# for q1, q2, label in batch:
			# 	q1_length, q2_length = len(q1), len(q2)
			#
			# 	q1_padding = [PAD_INDEX] * (max_len - q1_length)
			# 	q2_padding = [PAD_INDEX] * (max_len - q2_length)
			#
			# 	q1 = list(map(int, q1))
			# 	q2 = list(map(int, q2))
			#
			# 	q1_pad_seq, q2_pad_seq = (q1 + q1_padding), (q2 + q2_padding)
			# 	q1_pad_seq, q2_pad_seq = q1_pad_seq[:max_len], q2_pad_seq[:max_len]
			#
			# 	# input embed stuff
			# 	q1_pad_seq = emb_vector[q1_pad_seq]
			# 	q2_pad_seq = emb_vector[q2_pad_seq]
			#
			# # q1_pad_seq = pad_sequences(q1, maxlen=m

			token_seq = pad_sequences([token_seq], maxlen=max_seq_len, padding='post')

			# for idx in range(len(token_seq)):
			# 	if token_seq[idx] >= config.input_vocab_size:
			# 		token_seq[idx] = config.UNK_ID
				# if token_seq[idx] >= config.get('input_vocab_size'):
				# 	token_seq[idx] = config.get('UNK_ID')

			token_ids_set.append([cid, token_seq])
			token_len_set.append([cid, token_len])

			user_utt = fhu.readline()

	return dict(token_ids_set), dict(token_len_set)


def read_tokens_v2(token_user_ids_path):
	token_ids_set = []
	token_len_set = []

	with open(token_user_ids_path, mode="rt", encoding="utf-8") as fhu:
		user_utt = fhu.readline()
		counter = 0
		while user_utt:
			counter += 1
			if counter % 10000 == 0:
				print("  reading %s, line %d" % (token_user_ids_path, counter))
				sys.stdout.flush()

			user_utt = user_utt.replace("\n", "")
			source_ids = user_utt.split("\u241D")
			cid = int(source_ids[0])
			token_seq = source_ids[1].split("\u241E")
			token_len = len(token_seq)

			# q1_batch = []
			# q2_batch = []
			# label_batch = []
			#
			# for q1, q2, label in batch:
			# 	q1_length, q2_length = len(q1), len(q2)
			#
			# 	q1_padding = [PAD_INDEX] * (max_len - q1_length)
			# 	q2_padding = [PAD_INDEX] * (max_len - q2_length)
			#
			# 	q1 = list(map(int, q1))
			# 	q2 = list(map(int, q2))
			#
			# 	q1_pad_seq, q2_pad_seq = (q1 + q1_padding), (q2 + q2_padding)
			# 	q1_pad_seq, q2_pad_seq = q1_pad_seq[:max_len], q2_pad_seq[:max_len]
			#
			# 	# input embed stuff
			# 	q1_pad_seq = emb_vector[q1_pad_seq]
			# 	q2_pad_seq = emb_vector[q2_pad_seq]
			#
			# # q1_pad_seq = pad_sequences(q1, maxlen=m

			token_seq = pad_sequences([token_seq], maxlen=config.buckets[0], padding='post')

			# for idx in range(len(token_seq)):
			# 	if token_seq[idx] >= config.input_vocab_size:
			# 		token_seq[idx] = config.UNK_ID
			# if token_seq[idx] >= config.get('input_vocab_size'):
			# 	token_seq[idx] = config.get('UNK_ID')

			token_ids_set.append([cid, token_seq])
			token_len_set.append([cid, token_len])

			user_utt = fhu.readline()

	return dict(token_ids_set), dict(token_len_set)


def elmo_read_dataset(data_path):
	with open(data_path, 'r', encoding='utf-8') as fhu:
		cid_token_gid_set = []
		counter = 0

		for line in fhu:
			counter += 1
			if counter % 10000 == 0:
				print("  reading line %d" % counter)
				sys.stdout.flush()

			cid, query_tokens, gid = line.replace("\n", "").split("\u241D")
			query_tokens = query_tokens.split("\u241E")

			# cid = list(map(int, cid))
			# gid = list(map(int, gid))

			cid_token_gid_set.append([int(cid), query_tokens])

	return dict(cid_token_gid_set)

def embedding_converter(emb_path):

	import pandas as pd

	col_names = ['CID', 'QUERY', 'EMB']
	cid_emb = pd.read_csv(emb_path, header=None, sep='\u241D', encoding='utf-8', names=col_names)
	cid_emb = cid_emb[['CID', 'EMB']]
	empty_emb = np.zeros(128, dtype='float64')

	def convert(sent):
		try:
			wini_emb = np.asarray(sent.split('\u241E'), dtype='float64')
			wini_emb = wini_emb / np.linalg.norm(wini_emb)
		except Exception as e:
			print("Error with {}".format(sent))
			wini_emb = empty_emb

		return wini_emb

	cid_emb['EMB'] = cid_emb['EMB'].map(convert)
	cid_emb_dict = dict(zip(cid_emb.CID, cid_emb.EMB))

	return cid_emb_dict

class Batch_Creater_v2(object):
	def __init__(self, group_map, cid_qid, batch_size, cid_len=None, cid_emb_dict=None):

		self.cid_qid = cid_qid
		self.group_map = group_map
		self.batch_size = batch_size
		# self.num_half_batch = int(batch_size / 2)
		self.filtered_group = [x for x in group_map if len(x[1]) > 2]
		self.cid_emb_dict = cid_emb_dict
		self.cid_len = cid_len

	def create_weighted_pos_batch(self):

		group_len = [len(x[1]) for x in self.filtered_group]
		group_weights = np.array(group_len, dtype=np.double) / sum(group_len)
		self.pos_sample_group = random.choices(self.filtered_group, group_weights, k=self.batch_size)

		self.pos_random_group = [x[0] for x in self.pos_sample_group]
		self.pos_random_pair = [random.sample(i[1], 2) for i in self.pos_sample_group]

		return self.pos_random_pair

	def create_neg_batch(self):
		# neg_group = [x for x in filtered_group if x[0] not in pos_random_group]
		neg_candidate = [x[0] for x in self.filtered_group]
		neg_group_filtered = list(filter(lambda x: x not in self.pos_random_group, neg_candidate))
		neg_group_filtered = [[j for j in self.filtered_group if j[0] == i] for i in neg_group_filtered]

		# print(neg_group_filtered)
		#
		# group_len = [len(x[1]) for x in neg_group_filtered]
		# group_weights = np.array(group_len, dtype=np.double) / sum(group_len)

		# self.neg_group_sample = random.choices(neg_group_filtered, group_weights, k=self.num_half_batch)
		# self.neg_random_group = [x[0] for x in self.neg_group_sample]

		# neg_group_final = list(chain.from_iterable(self.neg_group_sample))
		neg_group_final = list(chain.from_iterable(neg_group_filtered))
		neg_random_pair = [random.sample(i[1], 1) for i in neg_group_final]
		neg_random = random.sample(neg_random_pair, self.batch_size)

		## Negative Set 서로 다른 그룹끼리 결합해 보자
		# neg_cand = [x[0] for x in neg_random]
		# neg_cand_list = list(combinations(neg_cand, 2))
		# neg_cand_pick = random.sample(neg_cand_list, self.num_neg_batch)
		# self.neg_neg_pair = [list(x) for x in neg_cand_pick]

		# print(neg_cand_pick)
		# print(len(neg_cand_pick))

		# #Negative 데이터셋 추출하여 Positive셋과 결합합니다.
		pos_random_q1 = [x[0] for x in self.pos_random_pair]
		neg_random_q2 = [x[0] for x in neg_random]
		pos_neg_pair = list(zip(neg_random_q2, pos_random_q1))
		self.pos_neg_pair = [list(x) for x in pos_neg_pair]

		# return neg_group_final, self.pos_neg_pair
		return self.pos_neg_pair
		# return self.neg_neg_pair
		# return self.pos_sample_group

	def create_batch(self):

		self.pos_random_pair = self.create_weighted_pos_batch()
		self.pos_neg_pair = self.create_neg_batch()
		#
		# print(self.pos_random_pair)
		# print(self.pos_neg_pair)

		self.pair_dataset = []

		for i in range(len(self.pos_random_pair)):
			pair_dataset = self.pos_random_pair[i] + self.pos_neg_pair[i]
			self.pair_dataset.append(pair_dataset)

		# print(raw_q_q_batch)
		q_list, pos_list, neg_1_list, neg_2_list = [], [], [], []
		q_len_list, pos_len_list, neg_1_len_list, neg_2_len_list = [], [], [], []

		for i in self.pair_dataset:

			q_list.append(self.cid_qid[i[0]])
			pos_list.append(self.cid_qid[i[1]])
			neg_1_list.append(self.cid_qid[i[2]])
			neg_2_list.append(self.cid_qid[i[3]])

			q_len_list.append(self.cid_len[i[0]])
			pos_len_list.append(self.cid_len[i[1]])
			neg_1_len_list.append(self.cid_len[i[2]])
			neg_2_len_list.append(self.cid_len[i[3]])

		q_bat, pos_bat, neg_1_bat, neg_2_bat = np.squeeze(q_list, axis=1), np.squeeze(pos_list, axis=1),\
		                                       np.squeeze(neg_1_list, axis=1), np.squeeze(neg_2_list, axis=1)
		q_len_bat, pos_len_bat, neg_1_len_bat, neg_2_len_bat = np.squeeze(q_len_list, axis=1), np.squeeze(pos_len_list, axis=1), \
		                                       np.squeeze(neg_1_len_list, axis=1), np.squeeze(neg_2_len_list, axis=1)

		# print("pos: ", len(pos_data), self.num_pos_batch)
		# print("neg: ", len(neg_data), self.num_neg_batch)

		# Define Label
		labels = [[1, 0, 0]] * self.batch_size
		labels = np.asarray(labels)
		# pos_label = np.ones(self.num_half_batch)  # pos batch label
		# neg_label = np.zeros(self.num_half_batch)
		# self.batch_label = np.hstack((pos_label, neg_label))

		return q_bat, pos_bat, neg_1_bat, neg_2_bat, labels


class Batch_Creater(object):
	def __init__(self, group_map, cid_qid, batch_size, cid_len=None, cid_emb_dict=None):

		self.cid_qid = cid_qid
		self.group_map = group_map
		self.batch_size = batch_size
		self.num_half_batch = int(batch_size / 2)
		self.filtered_group = [x for x in group_map if len(x[1]) > 2]
		self.cid_emb_dict = cid_emb_dict
		self.cid_len = cid_len

	def create_pos_batch(self):

		pos_sample_group = random.sample(self.filtered_group, self.num_half_batch)

		# Big
		# num_group = int(len(self.filtered_group) / 2)
		#
		# self.pos_sample_group = random.sample(self.filtered_group, num_group)

		self.pos_random_pair = [random.sample(i[1], 2) for i in pos_sample_group]
		self.pos_random_group = [j[0] for j in pos_sample_group]

		return self.pos_random_pair

	def create_weighted_pos_batch(self):

		group_len = [len(x[1]) for x in self.filtered_group]
		group_weights = np.array(group_len, dtype=np.double) / sum(group_len)
		self.pos_sample_group = random.choices(self.filtered_group, group_weights, k=self.num_half_batch)

		self.pos_random_group = [x[0] for x in self.pos_sample_group]
		self.pos_random_pair = [random.sample(i[1], 2) for i in self.pos_sample_group]

		return self.pos_random_pair

	def create_neg_batch(self):
		# neg_group = [x for x in filtered_group if x[0] not in pos_random_group]
		neg_candidate = [x[0] for x in self.filtered_group]
		neg_group_filtered = list(filter(lambda x: x not in self.pos_random_group, neg_candidate))
		neg_group_filtered = [[j for j in self.filtered_group if j[0] == i] for i in neg_group_filtered]

		# print(neg_group_filtered)
		#
		# group_len = [len(x[1]) for x in neg_group_filtered]
		# group_weights = np.array(group_len, dtype=np.double) / sum(group_len)

		# self.neg_group_sample = random.choices(neg_group_filtered, group_weights, k=self.num_half_batch)
		# self.neg_random_group = [x[0] for x in self.neg_group_sample]

		# neg_group_final = list(chain.from_iterable(self.neg_group_sample))
		neg_group_final = list(chain.from_iterable(neg_group_filtered))
		neg_random_pair = [random.sample(i[1], 1) for i in neg_group_final]
		neg_random = random.sample(neg_random_pair, self.num_half_batch)

		## Negative Set 서로 다른 그룹끼리 결합해 보자
		# neg_cand = [x[0] for x in neg_random]
		# neg_cand_list = list(combinations(neg_cand, 2))
		# neg_cand_pick = random.sample(neg_cand_list, self.num_neg_batch)
		# self.neg_neg_pair = [list(x) for x in neg_cand_pick]

		# print(neg_cand_pick)
		# print(len(neg_cand_pick))

		# #Negative 데이터셋 추출하여 Positive셋과 결합합니다.
		pos_random_q1 = [x[0] for x in self.pos_random_pair]
		neg_random_q2 = [x[0] for x in neg_random]
		pos_neg_pair = list(zip(neg_random_q2, pos_random_q1))
		self.pos_neg_pair = [list(x) for x in pos_neg_pair]

		# return neg_group_final, self.pos_neg_pair
		return self.pos_neg_pair
		# return self.neg_neg_pair
		# return self.pos_sample_group

	def create_neg_cos_sampling(self):
		""" 코사인 유사도 기반으로 Negative 답변을 추출한다"""
		neg_candidate = [x[0] for x in self.filtered_group]
		neg_group_filtered = list(filter(lambda x: x not in self.pos_random_group, neg_candidate))
		neg_group_filtered = [[j for j in self.filtered_group if j[0] == i] for i in neg_group_filtered]
		neg_group_filtered = list(chain.from_iterable(neg_group_filtered))

		self.pos_neg_pair_list = []

		# print(self.pos_sample_group)

		for i in self.pos_sample_group:

			# pos sample: batch size만큼의 데이터, 그룹과 answer를 가지고 있음
			pos_emb = self.cid_emb_dict[i[2][0]]
			# postive에 사용된 그룹들 중에 한개의 쿼리를 추출
			pos_pick = random.sample(i[1], 1)[0]

			# print("########")
			# print(pos_emb)
			# print(pos_pick)
			# print("########")

			best_cos_score = 0.0
			for j in neg_group_filtered:
				""" group map에서 Postive의 A와 가장 유사도가 높은 그룹 추출하기 """
				neg_emb = self.cid_emb_dict[j[2][0]]
				cos_score = self.cos_sim(pos_emb, neg_emb)

				if cos_score > best_cos_score:
					# 현재 Pos 데이터의 A의 유사도와 가장 높은 값이 있으면 뽑아서 업데이트 한다.
					print("Choose Group: {}, neg_group: {} with cos_sim {}".format(i[0], j[0], cos_score))
					best_cos_score = cos_score
					top_cos_neg = random.sample(j[1], 1)[0]
					pos_neg_pair = [pos_pick, top_cos_neg]
					# 가장 유사한 그룹을 업데이트 한다.
					top_answer_group, top_answer_cid = j[0], j[1]

			selected_pos_score = 0.0
			empty_emb = np.zeros(128, dtype='float64')

		for k in top_answer_cid:
			""" 선정된 Pos를 기반으로 가장 높은 Negative 쿼리 추출 """

			try:
				pos_pick_emb = self.cid_emb_dict[pos_pick]
				neg_pick_emb = self.cid_emb_dict[k]
			except:
				pos_pick_emb = empty_emb
				neg_pick_emb = empty_emb

			group_cos = self.cos_sim(pos_pick_emb, neg_pick_emb)

			if group_cos > selected_pos_score:
				# print("Choose Query: cos: {},: id: {}".format(group_cos, k))
				# print("update cos score pos group: {}, neg_group: {} with cos_sim {}".format(i[0], j[0], cos_score))
				selected_pos_score = group_cos
				neg_final_idx = k

			pos_neg_pair = [pos_pick, neg_final_idx] #최종 선정된 값 저장
			self.pos_neg_pair_list.append(pos_neg_pair)

		return self.pos_neg_pair_list

	def create_neg_big_batch(self):
		# neg_group = [x for x in filtered_group if x[0] not in pos_random_group]
		neg_candidate = [x[0] for x in self.filtered_group]
		neg_group_filtered = list(filter(lambda x: x not in self.pos_random_group, neg_candidate))
		neg_group_filtered = [[j for j in self.filtered_group if j[0] == i] for i in neg_group_filtered]

		neg_group_final = list(chain.from_iterable(neg_group_filtered))
		neg_random_pair = [random.sample(i[1], 1) for i in neg_group_final]

		# print(neg_random_pair)
		# print(len(neg_random_pair))

		# neg_random = random.sample(neg_random_pair, self.num_half_batch)

		## Negative Set 서로 다른 그룹끼리 결합해 보자
		neg_cand = [x[0] for x in neg_random_pair]
		neg_cand_list = list(combinations(neg_cand, 2))
		neg_cand_pick = random.sample(neg_cand_list, self.num_half_batch)
		self.neg_neg_pair = [list(x) for x in neg_cand_pick]

		# # print(neg_cand_pick)
		# # print(len(neg_cand_pick))
		#
		# # #Negative 데이터셋 추출하여 Positive셋과 결합합니다.
		# pos_random_q1 = [x[0] for x in self.pos_random_pair]
		# neg_random_q2 = [x[0] for x in neg_random]
		# pos_neg_pair = list(zip(neg_random_q2, pos_random_q1))
		# self.pos_neg_pair = [list(x) for x in pos_neg_pair]

		# return neg_group_final, self.pos_neg_pair
		return self.neg_neg_pair
		# return self.pos_sample_group

	def cos_sim(self, emb_1, emb_2):
		try:
			# top = np.dot(emb_1, emb_2)  # dot product
			# bottom = np.linalg.norm(emb_1) * np.linalg.norm(emb_2)
			# cos_score = top / bottom
			cos_score = np.dot(emb_1, emb_2)
		except Exception as e:
			print(e)
			print("Error Occurs: {}".format(e))
			cos_score = 0.0
			pass

		return cos_score


	def create_batch(self):

		RAND_COS = False

		# self.num_pos_batch
		self.pos_random_pairs, self.neg_pairs = [], []

		while (len(self.pos_random_pairs) < self.num_half_batch) and (len(self.neg_pairs) < self.num_half_batch):

			self.pos_random_pairs = self.pos_random_pairs + self.create_weighted_pos_batch()
			self.neg_pairs = self.neg_pairs + self.create_neg_batch()

			# if RAND_COS == True:
			# 	self.neg_pairs = self.neg_pairs + self.create_neg_cos_sampling()
			# else:
			# 	self.neg_pairs = self.neg_pairs + self.create_neg_batch()

		self.pos_random_pairs = self.pos_random_pairs[:self.num_half_batch]
		self.neg_pairs = self.neg_pairs[:self.num_half_batch]

		# while len(self.neg_pairs) < self.num_half_batch:
		# 	# self.neg_pairs = self.neg_pairs + self.create_neg_big_batch()
		# 	self.neg_pairs = self.neg_pairs + self.create_neg_cos_sampling()

		raw_q_q_batch = self.pos_random_pairs + self.neg_pairs

		# print(raw_q_q_batch)
		q1_list, q2_list = [], []
		q1_len, q2_len = [], []

		for i in raw_q_q_batch:
			q1_list.append(self.cid_qid[i[0]])
			q2_list.append(self.cid_qid[i[1]])

			q1_len.append(self.cid_len[i[0]])
			q2_len.append(self.cid_len[i[1]])

		self.q1_bat, self.q2_bat = np.squeeze(q1_list, axis=1), np.squeeze(q2_list, axis=1)
		self.q1_len, self.q2_len = np.squeeze(q1_len, axis=1), np.squeeze(q2_len, axis=1)

		# print("pos: ", len(pos_data), self.num_pos_batch)
		# print("neg: ", len(neg_data), self.num_neg_batch)

		# Define Label
		pos_label = np.ones(self.num_half_batch)  # pos batch label
		neg_label = np.zeros(self.num_half_batch)

		self.batch_label = np.hstack((pos_label, neg_label))

		return self.q1_bat, self.q2_bat, self.batch_label, self.q1_len, self.q2_len


def read_tokens_v2(token_user_ids_path, maxlen=20):
    q1_set, q2_set, label_set = [], [], []
    q1_len, q2_len = [], []

    with open(token_user_ids_path, mode="rt", encoding="utf-8") as fhu:
        user_utt = fhu.readline()
        counter = 0
        while user_utt:
            counter += 1
            if counter % 10000 == 0:
                print("  reading %s, line %d" % (token_user_ids_path, counter))
                sys.stdout.flush()

            q1, q2, labels = user_utt.split('\u241D')

            labels = labels.replace("\n", "")
            q1, q2 = q1.split("\u241E"), q2.split("\u241E")
            q1, q2 = list(map(int, q1)), list(map(int, q2))

            q1_set.append(q1)
            q2_set.append(q2)
            label_set.append(int(labels))
            q1_len.append(len(q1))
            q2_len.append(len(q2))

            user_utt = fhu.readline()

    q1_seq = pad_sequences(q1_set, maxlen=maxlen, padding='post')
    q2_seq = pad_sequences(q2_set, maxlen=maxlen, padding='post')

    query_set = np.stack((q1_seq, q2_seq), axis=1)
    query_len_set = np.stack((q1_len, q2_len), axis=1)

    return query_set, query_len_set, label_set
