import pandas as pd
import collections
import numpy as np
import pickle


# label index 정의
label_dict = { "neutral": 0, "entailment": 1, "contradiction": 2 }


regxs = list("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=")
def strip_string(string):
    for regx in regxs:
        string = string.replace(regx, " " + regx + " ")
    return string.lower().strip()


def build_text(file):
    dataset = pd.read_csv(file, sep="\t")

    gold_label = []
    sentence1 = []
    sentence2 = []
    for i, row in dataset.iterrows():
        if row['gold_label'] == "-" or pd.isnull(row['sentence1']) or pd.isnull(row['sentence2']):
            continue

        gold_label.append(label_dict[row['gold_label']])

        line1 = strip_string(row['sentence1']).split()
        sentence1.append(line1)

        line2 = strip_string(row['sentence2']).split()
        sentence2.append(line2)
    
    return gold_label, sentence1, sentence2


def build_vocab(texts):
    tokens = []
    for text in texts:
        for line in text:
            tokens.extend(line)

    counter = collections.Counter(tokens)
    vocab = { "<pad>": 0, "<unk>": 1, "<bos>": 2, "<dlm>": 3, "<eos>": 4 }
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


def index_pad(indexs, i_pad):
    length = 0
    for index in indexs:
        for line in index:
            length = max(length, len(line))

    pads = []
    for index in indexs:
        pad = []
        for line in index:
            line_pad = []
            line_pad.extend(line)
            line_pad.extend([i_pad] * (length - len(line_pad)))
            pad.append(line_pad[0:length])
        pads.append(pad)
    return pads


# <bos> text1 <dlm> text2 <eos>
def index_concat(index1, index2, i_bos, i_dlm, i_eos):
    concat = []
    for i in range(len(index1)):
        line_concat = []
        line_concat.append(i_bos)
        line_concat.extend(index1[i])
        line_concat.append(i_dlm)
        line_concat.extend(index2[i])
        line_concat.append(i_eos)
        concat.append(line_concat)
    return concat


def dump_data(train, valid, test, save):
    train_label, train_sentence1, train_sentence2 = build_text(train)
    valid_label, valid_sentence1, valid_sentence2 = build_text(valid)
    test_label, test_sentence1, test_sentence2 = build_text(test)

    vocab = build_vocab([train_sentence1, train_sentence2, valid_sentence1, valid_sentence2, test_sentence1, test_sentence2])

    data_i = [train_sentence1, train_sentence2, valid_sentence1, valid_sentence2, test_sentence1, test_sentence2]
    data_s = []
    for data in data_i:
        data_s.append(text_to_index(vocab, data))

    data_c = []
    for i in range(0, len(data_s), 2):
        data_c.append(index_concat(data_s[i], data_s[i + 1], vocab["<bos>"], vocab["<dlm>"], vocab["<eos>"]))
        data_c.append(index_concat(data_s[i + 1], data_s[i], vocab["<bos>"], vocab["<dlm>"], vocab["<eos>"]))
    
    data_s = index_pad(data_s, vocab["<pad>"])
    data_c = index_pad(data_c, vocab["<pad>"])

    with open(save, 'wb') as f:
        pickle.dump((vocab, train_label, data_s[0], data_s[1], data_c[0], data_c[1], valid_label, data_s[2], data_s[3], data_c[2], data_c[3], test_label, data_s[4], data_s[5], data_c[4], data_c[5]), f)


def load_data(file):
    with open(file, 'rb') as f:
        vocab, train_label, train_sentence1, train_sentence2, train_1_2, train_2_1, valid_label, valid_sentence1, valid_sentence2, valid_1_2, valid_2_1, test_label, test_sentence1, test_sentence2, test_1_2, test_2_1 = pickle.load(f)
    return vocab, train_label, train_sentence1, train_sentence2, train_1_2, train_2_1, valid_label, valid_sentence1, valid_sentence2, valid_1_2, valid_2_1, test_label, test_sentence1, test_sentence2, test_1_2, test_2_1


if __name__ == "__main__":
    dump_data("data/snli_1.0/snli_1.0_train.txt", "data/snli_1.0/snli_1.0_dev.txt", "data/snli_1.0/snli_1.0_test.txt", "data/snli_data.pkl")

