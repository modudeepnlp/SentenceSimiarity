"""
OpenGPT Documentation: https://huggingface.co/pytorch-transformers/model_doc/gpt.html
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import csv

from tqdm import tqdm, trange

import random
import numpy as np
import logging
import json
# import matplotlib.pyplot as plt

from models.transformer import TransformerModel, DoubleHeadModel
from models.loss import ClassificationLossCompute
# from models.opt import OpenAIAdam

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils.data import Data
from utils import configs

import preprocessing.custom_dataset as custom_dataset
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import config as config
import pickle

from pytorch_transformers import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer, OpenAIGPTConfig, OpenAIGPTModel,
                                  AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME)
from pytorch_transformers.modeling_openai import OpenAIGPTPreTrainedModel
from pytorch_transformers.modeling_utils import SequenceSummary
import pandas as pd

config_path = 'config/configs.transformer.json'
args = configs.Config.load(config_path)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(filename='./data_out/log')
logger.addHandler(file_handler)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

logger.info('Building Model')

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def strip_string(string):
    regxs = list("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=")
    for regx in regxs:
        string = string.replace(regx, " " + regx + " ")
    return string.lower().strip()

def load_snli_dataset(dataset_path):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    label_dict = {"neutral": 0, "entailment": 1, "contradiction": 2}

    dataset = pd.read_csv(dataset_path, sep="\t")
    count = 0
    output = []

    for i, row in dataset.iterrows():
        if row['gold_label'] == "-" or pd.isnull(row['sentence1']) or pd.isnull(row['sentence2']):
            continue
        count += 1
        # gold_label.append(label_dict[row['gold_label']])
        current_label = label_dict[row['gold_label']]
        line1 = strip_string(row['sentence1'])
        line2 = strip_string(row['sentence2'])

        output.append((line1, line2, current_label))

    return output

def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)
        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
        lm_labels = np.full((n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
        mc_labels = np.zeros((n_batch,), dtype=np.int64)

        for i, (sent1, sent2, mc_label), in enumerate(dataset):
            try:
                with_sent1 = [start_token] + sent1[:cap_length] + [delimiter_token] + sent2[:cap_length] + [clf_token]
                with_sent2 = [start_token] + sent2[:cap_length] + [delimiter_token] + sent1[:cap_length] + [clf_token]
                input_ids[i, 0, :len(with_sent1)] = with_sent1
                input_ids[i, 1, :len(with_sent2)] = with_sent2
                mc_token_ids[i, 0] = len(with_sent1) - 1
                mc_token_ids[i, 1] = len(with_sent2) - 1
                lm_labels[i, 0, :len(with_sent1)] = with_sent1
                lm_labels[i, 1, :len(with_sent2)] = with_sent2
                mc_labels[i] = mc_label
            except Exception as e:
                print(e)
                pass

        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)

        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))

        # for i, (story, cont1, cont2, mc_label), in enumerate(dataset):
        #     with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        #     with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
        #     input_ids[i, 0, :len(with_cont1)] = with_cont1
        #     input_ids[i, 1, :len(with_cont2)] = with_cont2
        #     mc_token_ids[i, 0] = len(with_cont1) - 1
        #     mc_token_ids[i, 1] = len(with_cont2) - 1
        #     lm_labels[i, 0, :len(with_cont1)] = with_cont1
        #     lm_labels[i, 1, :len(with_cont2)] = with_cont2
        #     mc_labels[i] = mc_label
        # all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        # tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets

def tokenize_and_encode(obj):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    elif isinstance(obj, int):
        return obj
    return list(tokenize_and_encode(o) for o in obj)

# Load tokenizer and model
# This loading functions also add new tokens and embeddings called `special tokens`
special_tokens = ['_start_', '_delimiter_', '_classify_']
tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
model = OpenAIGPTDoubleHeadsModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens))
model.to(device)

# Load Snli dataset with pkl
path_train = args.data_path + "snli_1.0_train.txt"
path_test = args.data_path + "snli_1.0_test.txt"
path_dev = args.data_path + "snli_1.0_dev.txt"
pkl_path = 'data_in/snli.pkl'

if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        encoded_datasets = pickle.load(f)
        logger.info("load completed")
else:
    train_dataset = load_snli_dataset(path_train)
    dev_dataset = load_snli_dataset(path_dev)
    test_dataset = load_snli_dataset(path_test)

    datasets = (train_dataset, test_dataset)
    encoded_datasets = tokenize_and_encode(datasets)

    with open(pkl_path, 'wb') as f:
        pickle.dump(encoded_datasets, f)
        logger.info('save completed')

# for dataset in encoded_datasets:
#
#     for sent1, sent2, _ in dataset:
#         a = max(len(sent1[:max_length]), len(sent2[:max_length])) + 3
#         print(a)

max_length = model.config.n_positions // 2 - 2
input_length = max(len(sent1[:max_length]) + len(sent2[:max_length]) + 3 \
                   for dataset in encoded_datasets for sent1, sent2, _ in dataset)

input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model

# Prepare inputs tensors and dataloaders
tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length, *special_tokens_ids)
train_tensor_dataset, eval_tensor_dataset = tensor_datasets[0], tensor_datasets[1]

train_data = TensorDataset(*train_tensor_dataset)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

eval_data = TensorDataset(*eval_tensor_dataset)
eval_sampler = RandomSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=train_sampler, batch_size=args.batch_size)

args.do_train = True
args.do_eval = True

if args.do_train:
    if args.epoch > 0:
        t_total = args.epoch
        args.num_train_epochs = args.epoch //\
                                (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) \
                // args.gradient_accumulation_steps * args.num_train_epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=0, t_total=t_total)  # PyTorch scheduler

if args.do_train:
    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
    model.train()

    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_steps = 0
        tqdm_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(tqdm_bar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels = batch
            losses = model(input_ids, mc_token_ids, lm_labels, mc_labels)
            loss = args.lm_coef * losses[0] + losses[1]
            loss.backward()
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item()
            nb_tr_steps += 1
            tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, scheduler.get_lr()[0])
            logger.info = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, scheduler.get_lr()[0])

# Save a trained model
if args.do_train:
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(args.output_dir)

    # Load a trained model and vocabulary that you have fine-tuned
    model = OpenAIGPTDoubleHeadsModel.from_pretrained(args.output_dir)
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.output_dir)
    model.to(device)

if args.do_eval:
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels = batch
        with torch.no_grad():
           _, mc_loss, _, mc_logits = model(input_ids, mc_token_ids, lm_labels, mc_labels)

        mc_logits = mc_logits.detach().cpu().numpy()
        mc_labels = mc_labels.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

        eval_loss += mc_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    train_loss = tr_loss/nb_tr_steps if args.do_train else None
    result = {'eval_loss': eval_loss,
              'eval_accuracy': eval_accuracy,
              'train_loss': train_loss}

    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))











#
# class CustomClassifier(OpenAIGPTPreTrainedModel):
#
#     def __init__(self, config):
#         super(CustomClassifier, self).__init__(config)
#
#         self.transformer = OpenAIGPTModel(config)
#         self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
#         self.apply(self.init_weights)
#         self.multiple_choice_head = SequenceSummary(config)
#         self.tie_weights()
#
#     def tie_weights(self):
#         """ Make sure we are sharing the input and output embeddings.
#             Export to TorchScript can't handle parameter sharing so we are cloning them instead.
#         """
#         self._tie_or_clone_weights(self.lm_head,
#                                    self.transformer.tokens_embed)
#
#     def forward(self, input_ids, mc_token_ids=None, lm_labels=None, mc_labels=None, token_type_ids=None,
#                 position_ids=None, head_mask=None):
#         transformer_outputs = self.transformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
#                                                head_mask=head_mask)
#
#         hidden_states = transformer_outputs[0]
#         lm_logits = self.lm_head(hidden_states)
#         # mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
#         mc_logits = self.multiple_choice_head(hidden_states)
#
#         outputs = (lm_logits, mc_logits) + transformer_outputs[1:]
#
#         return mc_logits
#
#
# # model = CustomClassifier.from_pretrained(args.model_name, num_special_tokens=special_tokens)
# model = CustomClassifier(config)
# model.to(device)
#
# # model.resize_token_embeddings(len(tokenizer))
#
# def build_tensor(label, sentence, device, batch_size):
#     torch_label = torch.tensor(label, dtype=torch.long).to(device)
#     torch_sent = torch.tensor(sentence, dtype=torch.long).to(device)
#     dataset = torch.utils.data.TensorDataset(torch_label, torch_sent)
#     data_loader = DataLoader(dataset,
#                         batch_size=batch_size,
#                         shuffle=True,
#                         drop_last=True)
#     return data_loader
#
# loss_fn = torch.nn.CrossEntropyLoss()
#
# num_total_steps = 100000
# num_warmup_steps = 10000
# warmup_propotion = float(num_warmup_steps) / float(num_total_steps)
#
# optimizer = AdamW(model.parameters(), lr=args.learning_rate, correct_bias=False)
# scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_total_steps)  # PyTorch scheduler
#
# def eval(loader):
#     total = 0
#     correct = 0
#     for n, (eval_label, eval_entail_sent) in enumerate(loader):
#
#         eval_outputs = model(eval_entail_sent)
#         _, pred = torch.max(eval_outputs.data, 1)
#         total += eval_label.size(0)
#         correct += (pred == label).sum()
#
#     acc = 100 * (correct.cpu().numpy()/total)
#     return acc
#
#
# step_list = []
# loss_list = []
# acc_test_list = []
# acc_dev_list = []
#
# train_loss = 0
# step = 0
# for epoch in range(args.epoch):
#
#     # Training Phase
#     for n, (label, entailment_sent) in enumerate(train_loader):
#
#         model.train()
#
#         optimizer.zero_grad()
#         outputs = model(entailment_sent)
#         loss = loss_fn(outputs, label)
#         loss.backward()
#         scheduler.step()
#         optimizer.step()
#
#         train_loss += loss.item()
#
#         step += 1
#
#         if n % 50 == 0:
#
#             epoch_loss = train_loss / 50
#             print("epoch {}, loss: {}".format(n, epoch_loss))
#             train_loss = 0
#
#         if n % 500 == 0:
#
#             model.eval()
#             step_list.append(step)
#             loss_list.append(epoch_loss)
#             acc_test = eval(test_loader)
#             acc_dev = eval(dev_loader)
#             acc_test_list.append(acc_test)
#             acc_dev_list.append(acc_dev)
#
#             print(acc_test_list)
#             print(acc_dev_list)