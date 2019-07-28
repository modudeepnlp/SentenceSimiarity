"""
OpenGPT Documentation: https://huggingface.co/pytorch-transformers/model_doc/gpt.html
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

from tqdm import tqdm, trange

import random
import numpy as np
import logging
import json
import matplotlib.pyplot as plt

from models.transformer import TransformerModel, DoubleHeadModel
from models.loss import ClassificationLossCompute
# from models.opt import OpenAIAdam

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils.data import Data
from utils import configs

import preprocessing.custom_dataset as custom_dataset
from torch.utils.data import DataLoader

import config as config
import pickle

from pytorch_transformers import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer, OpenAIGPTConfig, OpenAIGPTModel,
                                  AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME)
from pytorch_transformers.modeling_openai import OpenAIGPTPreTrainedModel
from pytorch_transformers.modeling_utils import SequenceSummary

config_path = 'config/configs.transformer.json'
args = configs.Config.load(config_path)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

if not os.path.exists(args.data_out):
	os.makedirs(args.data_out)

print('Building Model')

def load_pick(file_nm):
	with open(file_nm, 'rb') as f:
		df = pickle.load(f)
		print("load compeleted")
		return df

dev_data = load_pick('dev.pkl')

special_tokens = ['<bos>', '<del>', '<eos>']
tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)  # OpenAI용 토크나이저 불러오기
tokenizer.add_tokens(special_tokens)

config = OpenAIGPTConfig.from_pretrained('openai-gpt')
config.num_labels = 3
config.vocab_size = len(tokenizer)
config.summary_type = 'last'

tokenizer.bos_token = '<bos>'
tokenizer.eos_token = '<eos>'
tokenizer.sep_token = '<del>'

# model = OpenAIGPTModel(config)

class CustomClassifier(OpenAIGPTPreTrainedModel):

	def __init__(self, config):
		super(CustomClassifier, self).__init__(config)

		self.transformer = OpenAIGPTModel(config)
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		self.apply(self.init_weights)
		self.multiple_choice_head = SequenceSummary(config)
		self.tie_weights()

	def tie_weights(self):
		""" Make sure we are sharing the input and output embeddings.
			Export to TorchScript can't handle parameter sharing so we are cloning them instead.
		"""
		self._tie_or_clone_weights(self.lm_head,
		                           self.transformer.tokens_embed)

	def forward(self, input_ids, mc_token_ids=None, lm_labels=None, mc_labels=None, token_type_ids=None,
	            position_ids=None, head_mask=None):
		transformer_outputs = self.transformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
		                                       head_mask=head_mask)

		hidden_states = transformer_outputs[0]
		lm_logits = self.lm_head(hidden_states)
		# mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
		mc_logits = self.multiple_choice_head(hidden_states)

		outputs = (lm_logits, mc_logits) + transformer_outputs[1:]

		return mc_logits

# model = CustomClassifier.from_pretrained(args.model_name, num_special_tokens=special_tokens)
model = CustomClassifier(config)

# model.resize_token_embeddings(len(tokenizer))
# model.train()


# train_loader = DataLoader(train_data,
#                           batch_size=args.batch,
#                           shuffle=True,
#                           num_workers=config.cpu_processor,
#                           drop_last=True)
#
# test_loader = DataLoader(test_data,
#                          batch_size=args.batch,
#                          shuffle=False,
#                          num_workers=args.cpu_processor,
#                          drop_last=True)

dev_loader = DataLoader(dev_data,
                        batch_size=args.batch,
                        shuffle=False,
                        drop_last=True)

for n, (label, entailment_sent) in enumerate(dev_loader):
	label = Variable(label.to(device))
	entailment_sent = Variable(torch.stack(entailment_sent).to(device))

	print(entailment_sent)
outputs = model(entailment_sent)

dataset = custom_dataset.Custom_dataset()
train_data, test_data, dev_data = dataset.get_data()


train_loader = DataLoader(train_data,
                          batch_size=args.batch,
                          shuffle=True,
                          num_workers=config.cpu_processor,
                          drop_last=True)

test_loader = DataLoader(test_data,
                         batch_size=args.batch,
                         shuffle=False,
                         drop_last=True)

dev_loader = DataLoader(dev_data,
                        batch_size=args.batch,
                        shuffle=False,
                        drop_last=True)




model = TransformerModel(args)

n_ctx = 512

for n, (label, sent1, sent2) in enumerate(test_loader):
	print(sent1)

	label = Variable(label.to(device))
	sent1 = Variable(torch.stack(sent1).to(device))
	sent2 = Variable(torch.stack(sent2).to(device))

dh_model = DoubleHeadModel(args, sent1, 'inference', len(dataset.vocab_list), n_ctx)
for name, Parameter in dh_model.named_parameters():
	print(name, Parameter)

num_total_steps = 1000
num_warmup_steps = 100
warmup_propotion = float(num_warmup_steps) / float(num_total_steps)

optimizer = AdamW(dh_model.parameters(), lr=config.learning_rate, correct_bias=False)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler

loss = nn.CrossEntropyLoss()
dh_model.to(device)
dh_model = nn.DataParallel(dh_model)

for batch in train_data:
	loss = dh_model(sent1)
	loss.backward()
	scheduler.step()
	optimizer.step()

sys.exit(0)

# a = sent1.view(21,256)
# a.shape


model = Manhattan_LSTM(args.batch, args.hidden_size, [len(dataset.vocab_list), args.embedding_dim], use_embedding=False,
                       train_embedding=True)
model.to(device)
model.init_weights()

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
loss_function = nn.CrossEntropyLoss()

# 훈련
step_list = []
loss_list = []
acc_test_list = []
acc_dev_list = []
step = 0


def evaluation(loader):
	total = 0
	correct = 0
	for n, (label, sent1, sent2) in enumerate(loader):
		label = Variable(label.to(device))
		sent1 = Variable(torch.stack(sent1).to(device))
		sent2 = Variable(torch.stack(sent2).to(device))

		init_hidden = model.init_hidden(config.batch)
		out = model((sent1, sent2), init_hidden)
		_, pred = torch.max(out.data, 1)
		total += label.size(0)  # batch size
		correct += (pred == label).sum()
	acc = 100 * (correct.cpu().numpy() / total)
	return acc


for i in range(config.epoch):
	print("epoch = ", i)
	for n, (label, sent1, sent2) in enumerate(train_loader):
		optimizer.zero_grad()  # 초기화
		init_hidden = model.init_hidden(config.batch)

		label = Variable(label.to(device))
		sent1 = Variable(torch.stack(sent1).to(device))
		sent2 = Variable(torch.stack(sent2).to(device))

		# sent = [sent1, sent2]

		logit = model((sent1, sent2), init_hidden)
		loss = loss_function(logit, label)
		loss.backward()
		optimizer.step()
		step += 1
		if n % 500 == 0:
			print("epoch : ", i, " step : ", n, " loss : ", loss.item())
			step_list.append(step)
			loss_list.append(loss)
			acc_test = evaluation(test_loader)
			acc_dev = evaluation(dev_loader)
			acc_test_list.append(acc_test)
			acc_dev_list.append(acc_dev)

			torch.save(model, 'data_out/malstm.pth')

	# Loss 그래프
	# plt.plot(step_list, loss_list, 'r--')
	# plt.legend(['Training Loss'])
	# plt.xlabel('Step')
	# plt.ylabel('Loss')
	# plt.savefig('Train.png')
	#
	# # Acc 그래프
	# plt.plot(step_list, acc_test_list, 'b--')
	# plt.plot(step_list, acc_dev_list, 'g--')
	# plt.legend(['Test acc', 'dev acc'])
	# plt.savefig('Acc.png')

	print("dev acc : ", acc_dev_list)
	print("test acc : ", acc_test_list)


# outputs.shape
#
# lm_prediction_scores, mc_predictions_scroes = outputs[:2]
#
# # SentA
# a = outputs[0]
# # SentB
# sentA = a[0]
# sentB = a[1]

# sentA + sentB

# Start + Text1 + Delim + Text2 + Extract
# Start + Text2 + Delim + Text1 + Extract

# Start + Premise + Delim + Hypothese + Extract


# model = OpenAIGPTDoubleHeadsModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens))
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)
# choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]  # Assume you've added [CLS] to the vocabulary
# input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
# mc_labels_ids = torch.tensor([-1, -1]).unsqueeze(0)  # Batch size 1
#
# outputs = model(input_ids)


# last_hidden_states = outputs[0]

# model = OpenAIGPTDoubleHeadsModel(config)
# choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]  # Assume you've added [CLS] to the vocabulary
# input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
# mc_token_ids = torch.tensor([-1, -1]).unsqueeze(0)  # Batch size 1
# mc_token_ids = torch.tensor([0, 0, 1]).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids, mc_token_ids)
# lm_prediction_scores, mc_prediction_scores = outputs[:2]

# print('Model Parameters:')
# print('Hidden Size                  :', args.hidden_size)
# print('Batch Size                   :', args.batch_size)
# print('Max. input length            :', args.max_len)
# print('Learning rate                :', args.learning_rate)
# print('Number of Epochs             :', args.num_iters)
# print('--------------------------------------\n')
