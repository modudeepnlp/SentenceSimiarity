import pandas as pd
import collections
import numpy as np
import pickle

from utils import configs
from pytorch_transformers import OpenAIGPTTokenizer, OpenAIGPTConfig

label_dict = {"neutral": 0, "entailment": 1, "contradiction": 2}

regxs = list("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=")

def strip_string(string):
	for regx in regxs:
		string = string.replace(regx, " " + regx + " ")
	return string.lower().strip()

def build_text(file, max_len):
	dataset = pd.read_csv(file, sep="\t")

	gold_label = []
	sentence1 = []
	sentence2 = []

	for i, row in dataset.iterrows():
		if row['gold_label'] == "-" or pd.isnull(row['sentence1']) or pd.isnull(row['sentence2']):
			continue

		# gold_label.append(label_dict[row['gold_label']])
		current_label = label_dict[row['gold_label']]

		line1 = strip_string(row['sentence1'])
		line2 = strip_string(row['sentence2'])

		half_len = int(max_len / 2)

		bos = tokenizer.convert_tokens_to_ids('<bos>')
		delim = tokenizer.convert_tokens_to_ids('<del>')
		eos = tokenizer.convert_tokens_to_ids('<eos>')
		pad = tokenizer.convert_tokens_to_ids('<pad>')

		line1_index = tokenizer.convert_tokens_to_ids(list(line1))
		line2_index = tokenizer.convert_tokens_to_ids(list(line2))

		if len(line1_index) + len(line2_index) >= (max_len - 4):
			line1_index = line1_index[:half_len - 2]
			line2_index = line2_index[:half_len - 1]

		entailment_sent = list([bos] + line1_index + [delim] + line2_index + [eos])


		if len(entailment_sent) > max_len:
			entailment_sent = entailment_sent[:max_len -1] + [eos]

		elif len(entailment_sent) < max_len:

			entail_list = []

			pad_list = [pad] * (max_len - len(entailment_sent))
			entail_list.extend(entailment_sent)
			entail_list.extend(pad_list)

			entailment_sent = entail_list

		print(len(entailment_sent))

		# entailment_index = line1_index
		# [: half_len - 2]
		# [: half_len - 1]
		# entailment_sent = list('<bos> ' + line1 + ' <del>' + line2 + ' <eos>')
		# index_sent = tokenizer.convert_tokens_to_ids(entailment_sent)

		# a = list("<bos> a young girl with blue and pink ribbons in her braids")
		# special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)\
		# index_sent = list(tokenizer.convert_tokens_to_ids(a) for token in special_tokens)

		gold_label.append(current_label)
		sentence1.append(entailment_sent)

	# sentence2.append(line2)

		# tokenizer.convert_tokens_to_ids(token)

	return gold_label, sentence1


if __name__ == "__main__":
	config_path = 'config/configs.transformer.json'
	args = configs.Config.load(config_path)

	path_train = args.data_path + "snli_1.0_train.txt"
	path_test = args.data_path + "snli_1.0_test.txt"
	path_dev = args.data_path + "snli_1.0_dev.txt"

	special_tokens = ['<bos>', '<del>', '<eos>', '<pad>']
	tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt', special_tokens=special_tokens)  # OpenAI용 토크나이저 불러오기
	tokenizer.add_tokens(special_tokens)
	tokenizer.max_len = args.max_len

	tokenizer.bos_token = '<bos>'
	tokenizer.eos_token = '<eos>'
	tokenizer.sep_token = '<del>'
	tokenizer.pad_token = '<pad>'

	label, sent = build_text(path_train, args.max_len)

	# len(tokenizer)
	# tokenizer.convert_ids_to_tokens('<del>')

	with open('train.pkl', 'wb') as f:
		pickle.dump((label, sent), f)
		print("save completed")

	with open('train.pkl', 'rb') as f:
		label, data_df = pickle.load(f)
		print("load compeleted")

	# special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)

	# tokenizer.cls_token = '[CLS]'
	# tokenizer.convert_tokens_to_ids('[CLS]')

	choices = [
		"<bos> Hello, my dog is cute <del> Hello, my cat is cute <eos>"]  # Assume you've added [CLS] to the vocabulary
