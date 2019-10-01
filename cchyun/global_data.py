import os
import sentencepiece as spm
import pandas as pd
import collections
import numpy as np
import pickle

import torch
import torch.utils.data


PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SEP_ID = 4
CLS_ID = 5
MSK_ID = 6

# label index 정의
label_dict = { "neutral": 0, "entailment": 1, "contradiction": 2 }


"""
학습 corpus 생성
"""
def make_book_corpus(output, count=None):
    filenames = os.listdir("/home/ubuntu/Dev/Research/Dnn/bookcorpus/out_txts")
    with open(output, "w") as corpus:
        for filename in filenames:
            with open("/home/ubuntu/Dev/Research/Dnn/bookcorpus/out_txts/" + filename, "r") as txt:
                corpus.write(txt.read())
                corpus.write("\n\n")
            if count is not None:
                count -= 1
                if count <= 0:
                    break


"""
학습 corpus 생성
"""
def make_snli_corpus(srcs, dst):
    datasets = []
    for src in srcs:
        datasets.append(pd.read_csv(src, sep="\t"))
    with open(dst, "w") as corpus:
        for dataset in datasets:
            for i, row in dataset.iterrows():
                corpus.write(row["sentence1"].lower())
                corpus.write("\n")
                corpus.write(row["sentence1"].lower())
                corpus.write("\n")


"""
vocab 생성
"""
def build_vocab(input, prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        f"--input={input} --model_prefix={prefix} --vocab_size={vocab_size + 7}" + 
        " --model_type=bpe" +
        # " --max_sentence_length=10000" +
        # " --input_sentence_size=12000000 --shuffle_input_sentence=true" +
        " --pad_id=0 --pad_piece=[PAD]" +
        " --unk_id=1 --unk_piece=[UNK]" +
        " --bos_id=2 --bos_piece=[BOS]" +
        " --eos_id=3 --eos_piece=[EOS]" +
        " --user_defined_symbols=[SEP],[CLS],[MASK]" )


"""
loading vocab
"""
def load_vocab(file):
    vocab = spm.SentencePieceProcessor()
    vocab.load(file)
    return vocab


"""
read snli txt
"""
def read_snli(vocab, file):
    dataset = pd.read_csv(file, sep="\t")

    gold_label = []
    sentence1 = []
    sentence2 = []
    for i, row in dataset.iterrows():
        if row["gold_label"] == "-" or pd.isnull(row["sentence1"]) or pd.isnull(row["sentence2"]):
            continue

        gold_label.append(label_dict[row["gold_label"]])
        sentence1.append(vocab.encode_as_ids(row["sentence1"]))
        sentence2.append(vocab.encode_as_ids(row["sentence2"]))
    
    return gold_label, sentence1, sentence2


def dump_snli(train, valid, test, voc, save):
    vocab = load_vocab(voc)

    train_label, train_sentence1, train_sentence2 = read_snli(vocab, train)
    valid_label, valid_sentence1, valid_sentence2 = read_snli(vocab, valid)
    test_label, test_sentence1, test_sentence2 = read_snli(vocab, test)

    with open(save, 'wb') as f:
        pickle.dump((train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2), f)


def load_snli(file):
    with open(file, 'rb') as f:
        train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2 = pickle.load(f)
    return train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2


if __name__ == "__main__":
    # make_book_corpus("data/corpus.book.small.txt", 100)
    # make_book_corpus("data/corpus.book.large.txt", 4000)
    # make_snli_corpus(["data/snli_1.0/snli_1.0_train.txt", "data/snli_1.0/snli_1.0_dev.txt", "data/snli_1.0/snli_1.0_test.txt"], "data/corpus.snli.txt")
    # build_vocab("data/corpus.snli.txt", "m_snli_16000", 16000)
    # dump_snli("data/snli_1.0/snli_1.0_train.txt", "data/snli_1.0/snli_1.0_dev.txt", "data/snli_1.0/snli_1.0_test.txt", "data/m_snli_16000.model", "data/snli_16000_data.pkl")
    pass

