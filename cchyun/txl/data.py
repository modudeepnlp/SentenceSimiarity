import sys
sys.path.append("..")

import os
from tqdm import tqdm, trange
import numpy as np
import pickle

import torch
import torch.utils.data
import torch.nn.functional as F

import global_data


"""
train dataset
"""
class TXLDataSet(torch.utils.data.Dataset):
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
        sentence.append(global_data.BOS_ID)
        sentence.extend(sentence1)
        sentence.append(global_data.SEP_ID)
        sentence.extend(sentence2)
        sentence.append(global_data.EOS_ID)
        return torch.tensor(label), torch.tensor(sentence)


"""
pretrain data iterator
"""
class TXLIterator(object):
    def __init__(self, config, token_ids):
        self.config = config
        n_step = len(token_ids) // config.n_batch
        token_ids = torch.LongTensor(token_ids[:n_step * config.n_batch]).to(config.device)
        self.token_ids = token_ids.view(config.n_batch, -1).contiguous()
        self.n_step = (n_step + self.config.n_dec_seq - 1) // self.config.n_dec_seq

    def get_batch(self, i):
        seq_len = min(self.config.n_dec_seq, self.token_ids.size(1) - 1 - i)
        beg_idx = i
        end_idx = beg_idx + seq_len

        inputs = self.token_ids[:,beg_idx:end_idx]
        labels = self.token_ids[:,beg_idx+1:end_idx+1]

        return inputs.contiguous(), labels.contiguous()

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.token_ids.size(1) - 1, self.config.n_dec_seq):
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()
    
    def __len__(self):
        return self.n_step


"""
data loader 생성
"""
def build_data_loader(label, sentence1s, sentence2s, batch_size):
    dataset = TXLDataSet(label, sentence1s, sentence2s)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader


"""
data preprocessing
"""
def collate_fn(inputs):
    labels, sentences = list(zip(*inputs))

    sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)

    batch = [
        torch.stack(labels, dim=0),
        sentences,
    ]
    return batch


"""
pretrain data load
"""
def load_pretrain(file):
    with open(file, 'rb') as f:
        token_ids = pickle.load(f)
    return token_ids


if __name__ == '__main__':
    pass
