import numpy as np
from tensorflow.keras.utils import to_categorical
import json

def extract_tokens_from_binary_parse(parse):
	return parse.replace('(', ' ').replace(')', ' ').replace('-LRB-', '(').replace('-RRB-', ')').split()

def yield_examples(fn, skip_no_majority=True, limit=None):
	for i, line in enumerate(open(fn)):
		if limit and i > limit:
			break
		data = json.loads(line)
		label = data['gold_label']
		s1 = ' '.join(extract_tokens_from_binary_parse(data['sentence1_binary_parse']))
		s2 = ' '.join(extract_tokens_from_binary_parse(data['sentence2_binary_parse']))
		if skip_no_majority and label == '-':
			continue
		yield (label, s1, s2)

def get_data(fn, limit=None):
	raw_data = list(yield_examples(fn=fn, limit=limit))
	left = [s1 for _, s1, s2 in raw_data]
	right = [s2 for _, s1, s2 in raw_data]
	print(max(len(x.split()) for x in left))
	print(max(len(x.split()) for x in right))

	LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
	Y = np.array([LABELS[l] for l, s1, s2 in raw_data])
	# Y = to_categorical(Y, len(LABELS))

	return left, right, Y