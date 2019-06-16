import os
from glob import glob
from util.data_utils import read_dataset
import numpy as np
from sklearn.model_selection import train_test_split

def load_baloo_db(data_dir):
	train_data_path = glob(os.path.join(data_dir, 'table=QUERYTOKENSEQUENCE.text', 'part*'))[0]
	q1, q2, labels, pos_len, neg_len = read_dataset(train_data_path)

	train_valid_ratio = 0.2
	X = np.stack((q1, q2), axis=1)
	X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=train_valid_ratio, stratify=labels,
	                                                    random_state=712)
	q1_set, q2_set, label_set = X_train[:, 0], X_train[:, 1], y_train
	q1_test_set, q2_test_set, label_test_set = X_test[:, 0], X_test[:, 1], y_test

	train_unique, train_counts = np.unique(label_set, return_counts=True)
	val_unique, val_counts = np.unique(label_test_set, return_counts=True)

	train_neg_cnt, train_pos_cnt = train_counts[0], train_counts[1]
	val_neg_cnt, val_pos_cnt = val_counts[0], val_counts[1]

	return q1_set, q2_set, label_set, q1_test_set, q2_test_set, label_test_set

