import sys
sys.path.append("..")

import os
from tqdm import tqdm, trange
import numpy as np
import pickle

import torch
import torch.utils.data
import torch.nn.functional as F

import data
import tokenizer


def demp_pretrain(prefix):
    in_file = f"../data/corpus.{prefix}.txt"
    out_file = f"../data/txl_epoch_0_data.{prefix}.pkl"

    vocab, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2, max_sentence1, max_sentence2, max_sentence_all = data.load_data("../data/snli_data.pkl")

    token_ids = []
    with open(in_file) as f:
        for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
            line = line.strip()
            if line == "":
                pass
            else:
                tokens = tokenizer.tokenize(line)
                for token in tokens:
                    if token in vocab:
                        token_ids.append(vocab[token])
                    else:
                        token_ids.append(vocab["<unk>"])
    token_ids = np.array(token_ids)
    
    with open(out_file, 'wb') as f:
        pickle.dump((token_ids), f)


def load_pretrain(prefix):
    in_file = f"../data/txl_epoch_0_data.{prefix}.pkl"

    with open(in_file, 'rb') as f:
        token_ids = pickle.load(f)
    return token_ids


def make_corpus(prefix, count):
    filenames = os.listdir("/home/ubuntu/Dev/Research/Dnn/bookcorpus/out_txts")
    with open(f"../data/corpus.{prefix}.txt", "w") as corpus:
        for filename in filenames:
            with open("/home/ubuntu/Dev/Research/Dnn/bookcorpus/out_txts/" + filename, "r") as txt:
                corpus.write(txt.read())
                corpus.write("\n\n")
            count -= 1
            if count <= 0:
                break


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

        return inputs.contiguous(), labels.contiguous(), seq_len

    def get_fixlen_iter(self, start=0):
        for i in range(start, self.token_ids.size(1) - 1, self.config.n_dec_seq):
            yield self.get_batch(i)

    def __iter__(self):
        return self.get_fixlen_iter()
    
    def __len__(self):
        return self.n_step


class GptDataSet(torch.utils.data.Dataset):
    def __init__(self, labels, sentence1s, sentence2s, device):
        self.i_bos = 3 # bos
        self.i_cls = 4 # cls
        self.i_sep = 5 # sep
        self.i_eos = 6 # eos
        self.labels = labels
        self.sentence1s = sentence1s
        self.sentence2s = sentence2s
        self.device = device
    
    def __len__(self):
        assert len(self.sentence1s) == len(self.sentence2s)
        return len(self.sentence1s)
    
    def __getitem__(self, uid):
        label = self.labels[uid]
        sentence1 = self.sentence1s[uid]
        sentence2 = self.sentence2s[uid]
        sentence = []
        sentence.append(self.i_bos)
        sentence.extend(sentence1)
        sentence.append(self.i_sep)
        sentence.extend(sentence2)
        sentence.append(self.i_eos)
        return torch.tensor(label).to(self.device), torch.tensor(sentence).to(self.device)


def build_data_loader(label, sentence1s, sentence2s, device, batch_size):
    torch_labe = torch.tensor(label, dtype=torch.long).to(device)

    dataset = GptDataSet(label, sentence1s, sentence2s, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return loader


def collate_fn(inputs):
    labels, sentences = list(zip(*inputs))

    sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)

    batch = [
        torch.stack(labels, dim=0),
        sentences,
    ]
    return batch


if __name__ == '__main__':
    # make_corpus("small", 10)
    # demp_pretrain("small")
    pass