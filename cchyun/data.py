"""
학습할 데이터을 읽어 들인다.
"""
from vocab import strip_string
import numpy as np
import pandas as pd

# label index 정의
label_dict = { "neutral": 0, "entailment": 1, "contradiction": 2 }


"""
pickle로 vocab 로드
"""
def load_vocab(file="data/vocab.txt"):
    vocab = {}
    index = 0
    with open(file, "r") as f:
        for token in f:
            vocab[token.strip()] = index
            index += 1
    return vocab


"""
파일로 부터 vocab을 이용해 데이터 로드
"""
def load_data(file, vocab):
    gold_label = []
    sentence1 = []
    sentence2 = []

    dataset = pd.read_csv(file, sep="\t")
    for i, row in dataset.iterrows():
        if row['gold_label'] == "-" or pd.isnull(row['sentence1']) or pd.isnull(row['sentence2']):
            continue

        gold_label.append(label_dict[row['gold_label']])

        line1 = []
        for token in strip_string(row['sentence1']).split():
            line1.append(vocab[token])
        while len(line1) < 82:
            line1.append(1) # <pad>
        sentence1.append(line1)

        line2 = []
        for token in strip_string(row['sentence2']).split():
            line2.append(vocab[token])
        while len(line2) < 63:
            line2.append(1) # <pad>
        sentence2.append(line2)
    
    return gold_label, sentence1, sentence2


"""
테스트 스크립트
"""
def zero_count(labels):
    zero_count = 0
    for label in labels:
        if label == 0:
            zero_count += 1
    return zero_count

def sentence_len(sentence):
    len_sentence = 0
    for sss in sentence:
        len_sentence = max(len_sentence, len(sss))
    return len_sentence

def main():
    vocab = load_vocab()
    print("vocab size: {}".format(len(vocab)))

    train_labe, train_sentence1, train_sentence2 = load_data("data/snli_1.0/snli_1.0_train.txt", vocab)
    print("train zero: {} / {}".format(zero_count(train_labe), len(train_labe)))
    print("train sentence1: {}".format(sentence_len(train_sentence1)))
    print("train sentence2: {}".format(sentence_len(train_sentence2)))

    dev_labe, dev_sentence1, dev_sentence2 = load_data("data/snli_1.0/snli_1.0_dev.txt", vocab)
    print("dev zero: {} / {}".format(zero_count(dev_labe), len(dev_labe)))
    print("dev sentence1: {}".format(sentence_len(dev_sentence1)))
    print("dev sentence2: {}".format(sentence_len(dev_sentence2)))

    test_labe, test_sentence1, test_sentence2 = load_data("data/snli_1.0/snli_1.0_test.txt", vocab)
    print("dev zero: {} / {}".format(zero_count(test_labe), len(test_labe)))
    print("dev sentence1: {}".format(sentence_len(test_sentence1)))
    print("dev sentence2: {}".format(sentence_len(test_sentence2)))


if __name__ == "__main__":
    main()

