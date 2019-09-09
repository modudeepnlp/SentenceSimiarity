import pandas as pd
import collections
import numpy as np
import pickle

import torch
import torch.utils.data

import tokenizer


# label index 정의
label_dict = { "neutral": 0, "entailment": 1, "contradiction": 2 }


def build_text(file):
    dataset = pd.read_csv(file, sep="\t")

    gold_label = []
    sentence1 = []
    sentence2 = []
    for i, row in dataset.iterrows():
        if row['gold_label'] == "-" or pd.isnull(row['sentence1']) or pd.isnull(row['sentence2']):
            continue

        gold_label.append(label_dict[row['gold_label']])

        line1 = tokenizer.tokenize(row['sentence1'])
        sentence1.append(line1)

        line2 = tokenizer.tokenize(row['sentence2'])
        sentence2.append(line2)
    
    return gold_label, sentence1, sentence2


def build_vocab(texts):
    tokens = []
    for text in texts:
        for line in text:
            tokens.extend(line)

    counter = collections.Counter(tokens)
    vocab = { "<pad>": 0, "<unk>": 1, "<msk>": 2, "<bos>": 3, "<cls>": 4, "<sep>": 5, "<eos>": 6 }
    index = len(vocab)
    for key, _ in counter.items():
        vocab[key] = index
        index += 1
    return vocab


def text_to_index(vocab, text):
    index = []
    for line in text:
        line_index = []
        for token in line:
            if token in vocab:
                line_index.append(vocab[token])
            else:
                line_data.append(vocab["<unk>"])
        index.append(line_index)
    return index


def dump_data(train, valid, test, save):
    train_label, train_sentence1, train_sentence2 = build_text(train)
    valid_label, valid_sentence1, valid_sentence2 = build_text(valid)
    test_label, test_sentence1, test_sentence2 = build_text(test)

    vocab = build_vocab([train_sentence1, train_sentence2, valid_sentence1, valid_sentence2, test_sentence1, test_sentence2])

    datas = []
    for data in [train_sentence1, train_sentence2, valid_sentence1, valid_sentence2, test_sentence1, test_sentence2]:
        datas.append(text_to_index(vocab, data))
    
    max_sentence1 = 0
    for i in range(0, len(datas), 2):
        for line in datas[i]:
            max_sentence1 = max(max_sentence1, len(line))
    
    max_sentence2 = 0
    for i in range(1, len(datas), 2):
        for line in datas[i]:
            max_sentence2 = max(max_sentence2, len(line))
    
    max_sentence_all = 0
    for i in range(0, len(datas), 2):
        for j in range(len(datas[i])):
            max_sentence_all = max(max_sentence_all, len(datas[i][j]) + len(datas[i + 1][j]))
    
    print(max_sentence1, max_sentence2, max_sentence_all)

    with open(save, 'wb') as f:
        pickle.dump((vocab, train_label, datas[0], datas[1], valid_label, datas[2], datas[3], test_label, datas[4], datas[5], max_sentence1, max_sentence2, max_sentence_all), f)


def load_data(file):
    with open(file, 'rb') as f:
        vocab, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2, max_sentence1, max_sentence2, max_sentence_all = pickle.load(f)
    return vocab, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2, max_sentence1, max_sentence2, max_sentence_all


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
        return torch.tensor(uid).to(self.device), torch.tensor(label).to(self.device), torch.tensor(sentence).to(self.device)


def collate_fn(inputs):
    uids, labels, sentence = list(zip(*inputs))

    sentence = torch.nn.utils.rnn.pad_sequence(sentence, batch_first=True, padding_value=0)

    batch = [
        torch.stack(uids, dim=0),
        torch.stack(labels, dim=0),
        sentence,
    ]
    return batch


if __name__ == "__main__":
    dump_data("data/snli_1.0/snli_1.0_train.txt", "data/snli_1.0/snli_1.0_dev.txt", "data/snli_1.0/snli_1.0_test.txt", "data/snli_data.pkl")

