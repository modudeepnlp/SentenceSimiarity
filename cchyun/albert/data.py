import sys
sys.path.append("..")

import os
from tqdm import tqdm, trange
import numpy as np
import pickle
import time

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


start_ids = dict()
def is_start_id(id, vocab):
    if id not in start_ids:
        start_ids[id] = vocab.id_to_piece(id).startswith(u"\u2581")
    return start_ids[id]


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    for (i, token) in enumerate(tokens):
        if token == global_data.CLS_ID or token == global_data.SEP_ID:
            continue
        if (len(cand_indices) >= 1 and not is_start_id(token, vocab)):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])
    
    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            if random() < 0.8:
                masked_token = global_data.MSK_ID
            else:
                if random() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append({"index": index, "label": tokens[index]})
            tokens[index] = masked_token
    
    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x["index"])
    mask_indices = [p["index"] for p in masked_lms]
    masked_token_labels = [p["label"] for p in masked_lms]

    return tokens, mask_indices, masked_token_labels

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def create_instances_from_document(
        doc_database, doc_idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab, vocab_list):
    document = doc_database[doc_idx]
    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = randint(2, max_num_tokens)

    instances = []
    current_chunk = []
    current_length = 0
    for i in range(len(document)):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if 1 < len(current_chunk):
                a_end = 1
                if len(current_chunk) > 1:
                    a_end = randrange(1, len(current_chunk))

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                
                tokens_b = []
                for j in range(a_end, len(current_chunk)):
                    tokens_b.extend(current_chunk[j])

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                if random() < 0.5:
                    is_normal_next = 1
                else:
                    is_normal_next = 0
                    temp = tokens_a
                    tokens_a = tokens_b
                    tokens_b = temp

                tokens = [global_data.CLS_ID] + tokens_a + [global_data.SEP_ID] + tokens_b + [global_data.SEP_ID]

                tokens, mask_indices, masked_token_labels = create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab, vocab_list)

                instance = {
                    "tokens": tokens,
                    "len_a": len(tokens_a),
                    "len_b": len(tokens_b),
                    "nsp_lbl": is_normal_next,
                    "mlm_pos": mask_indices,
                    "mlm_lbl": masked_token_labels
                }
                instances.append(instance)
            current_chunk = []
            current_length = 0
    return instances


"""
pretrain data create and dump
"""
def demp_pretrain(vocab_file, corpus, file):
    args = cfg.Config({
        "max_seq_len": 512,
        "short_seq_prob": 0.1,
        "masked_lm_prob": 0.15,
        "max_predictions_per_seq": 20,
    })
    vocab = global_data.load_vocab(vocab_file)

    print(f"read {corpus}, write {file}")
    docs = []
    with open(corpus) as f:
        doc = []
        for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
            line = line.strip()
            if line == "":
                if doc:
                    docs.append(doc)
                    doc = []
            else:
                tokens = vocab.encode_as_ids(line.lower())
                if tokens:
                    doc.append(tokens)
        if doc:
            docs.append(doc)
    if len(docs) <= 1:
        exit("ERROR: more documnet need")
    

    vocab_list = []
    for id in range(vocab.get_piece_size()):
        if not vocab.is_unknown(id):
            vocab_list.append(id)
    
    with open(file, "w") as f:
        with tqdm(total=len(docs), desc=f"Document") as pbar:
            for doc_idx in range(len(docs)):
                timestamp1 = time.time()
                doc_instances = create_instances_from_document(
                    docs, doc_idx,
                    max_seq_length=args.max_seq_len,
                    short_seq_prob=args.short_seq_prob,
                    masked_lm_prob=args.masked_lm_prob,
                    max_predictions_per_seq=args.max_predictions_per_seq,
                    vocab=vocab,
                    vocab_list=vocab_list)
                for instance in doc_instances:
                    f.write(json.dumps(instance))
                    f.write("\n")
                timestamp2 = time.time()
                if 60 < (timestamp2 - timestamp1):
                    print(f">>>> {(timestamp2 - timestamp1)}: {len(doc_instances)}")
                pbar.update(1)
                pbar.set_postfix_str(f"Instances: {len(doc_instances)}")


def build_pretrain_loader(file, vocab, n_batch):
    dataset = PregeneratedDataset(vocab, file)
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=n_batch, collate_fn=pretrain_collate_fn)
    return dataloader


class PregeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, vocab, file):
        self.vocab = vocab
        self.file = file
        # num_samples = 7013360
        num_samples = 0
        with open(file, "r") as f:
            for i, line in enumerate(f):
                num_samples += 1
        print(f"Number of line: {num_samples}")
        seq_len = 512

        self.input_ids = []
        self.segment_ids = []
        self.lm_label_ids = []
        self.is_nexts = []
        
        with open(file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                data = json.loads(line)
                self.input_ids.append(data["tokens"])
                self.segment_ids.append([0 for _ in range(data["len_a"] + 2)] + [1 for _ in range(data["len_b"] + 1)])
                lm_label_id = np.full(len(data["tokens"]), dtype=np.int, fill_value=-1)
                lm_label_id[data["mlm_pos"]] = data["mlm_lbl"]
                self.lm_label_ids.append(lm_label_id)
                self.is_nexts.append(data["nsp_lbl"])
                if i % 1000 == 0:
                    print(f"Read line: {i:5d}", end="\r")
        print(f"End of read file: {i + 1} lines")
    
    def __len__(self):
        assert len(self.input_ids) == len(self.segment_ids)
        assert len(self.segment_ids) == len(self.lm_label_ids)
        assert len(self.lm_label_ids) == len(self.is_nexts)
        return len(self.input_ids)

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item]),
                torch.tensor(self.segment_ids[item]),
                torch.tensor(self.lm_label_ids[item]),
                torch.tensor(self.is_nexts[item]))


"""
data preprocessing
"""
def pretrain_collate_fn(inputs):
    tokens, segment_ids, masked_labels, nsp_labels = list(zip(*inputs))
    
    tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=global_data.PAD_ID)
    segment_ids = torch.nn.utils.rnn.pad_sequence(segment_ids, batch_first=True, padding_value=global_data.PAD_ID)
    masked_labels = torch.nn.utils.rnn.pad_sequence(masked_labels, batch_first=True, padding_value=-1)

    batch = [
        tokens,
        segment_ids,
        masked_labels,
        torch.stack(nsp_labels, dim=0),
    ]
    return batch


if __name__ == '__main__':
    prefix = 8000
    # demp_pretrain(f"../data/m_snli_{prefix}.model", "../data/corpus.book.large.00.txt", f"../data/pretrain_albert_{prefix}_0.json")
    # demp_pretrain(f"../data/m_snli_{prefix}.model", "../data/corpus.book.large.01.txt", f"../data/pretrain_albert_{prefix}_1.json")
    demp_pretrain(f"../data/m_snli_{prefix}.model", "../data/corpus.book.large.02.txt", f"../data/pretrain_albert_{prefix}_2.json")
    demp_pretrain(f"../data/m_snli_{prefix}.model", "../data/corpus.book.large.03.txt", f"../data/pretrain_albert_{prefix}_3.json")
    pass

