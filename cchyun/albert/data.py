import sys
sys.path.append("..")

import os
from tqdm import tqdm, trange
import numpy as np
import pickle

import collections
from tempfile import TemporaryDirectory
from pathlib import Path
import shelve
from random import random, randrange, randint, shuffle, choice
import json
import logging

import torch
import torch.utils.data
import torch.nn.functional as F

import config as cfg
import global_data


"""
train dataset
"""
class BERTDataSet(torch.utils.data.Dataset):
    def __init__(self, labels, sentence1s, sentence2s):
        self.labels = labels
        self.sentence1s = sentence1s
        self.sentence2s = sentence2s
    
    def __len__(self):
        assert len(self.sentence1s) == len(self.sentence2s)
        return len(self.sentence1s)
    
    def __getitem__(self, uid):
        label = self.labels[uid]
        sentence1 = self.sentence1s[uid]
        sentence2 = self.sentence2s[uid]
        sentence = []
        sentence = []
        sentence.append(global_data.CLS_ID)
        sentence.extend(sentence1)
        sentence.append(global_data.SEP_ID)
        sentence.extend(sentence2)
        sentence.append(global_data.SEP_ID)
        segment = []
        segment.extend([0] * (len(sentence1) + 2))
        segment.extend([1] * (len(sentence2) + 1))
        return torch.tensor(label), torch.tensor(sentence), torch.tensor(segment)


"""
data loader 생성
"""
def build_data_loader(label, sentence1s, sentence2s, batch_size):
    dataset = BERTDataSet(label, sentence1s, sentence2s)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collate_fn)
    return loader


"""
data preprocessing
"""
def train_collate_fn(inputs):
    labels, sentences, segment = list(zip(*inputs))

    sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=global_data.PAD_ID)
    segment = torch.nn.utils.rnn.pad_sequence(segment, batch_first=True, padding_value=global_data.PAD_ID)

    batch = [
        torch.stack(labels, dim=0),
        sentences,
        segment,
    ]
    return batch


if __name__ == '__main__':
    prefix = 8000
    demp_pretrain(f"../data/m_snli_{prefix}.model", f"../data/pretrain_bert_{prefix}_0")
    pass

