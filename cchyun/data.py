import pandas as pd
import gluonnlp as nlp
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
    length1 = 0
    sentence2 = []
    length2 = 0
    for i, row in dataset.iterrows():
        if row['gold_label'] == "-" or pd.isnull(row['sentence1']) or pd.isnull(row['sentence2']):
            continue

        gold_label.append(label_dict[row['gold_label']])

        line1 = strip_string(row['sentence1']).split()
        sentence1.append(line1)
        length1 = max(length1, len(line1))

        line2 = strip_string(row['sentence2']).split()
        sentence2.append(line2)
        length2 = max(length2, len(line2))
    
    return gold_label, sentence1, length1, sentence2, length2


def build_vocab(texts):
    tokens = []
    for text in texts:
        for line in text:
            tokens.extend(line)

    counter = nlp.data.count_tokens(tokens)
    vocab = nlp.Vocab(counter, bos_token=None, eos_token=None, min_freq=1)

    return vocab


def text_to_data(vocab, text, length):
    data = []
    for line in text:
        line_data = []
        for token in line:
            if token in vocab:
                line_data.append(vocab[token])
            else:
                line_data.append(vocab["<unk>"])
        line_data.extend([1] * (length - len(line_data)))
        data.append(line_data)

    return data


def dump_data(train, valid, test, save):
    train_label, train_sentence1, train_length1, train_sentence2, train_length2 = build_text(train)
    valid_label, valid_sentence1, valid_length1, valid_sentence2, valid_length2 = build_text(valid)
    test_label, test_sentence1, test_length1, test_sentence2, test_length2 = build_text(test)
    length = max(train_length1, train_length2, valid_length1, valid_length2, test_length1, test_length2)

    vocab = build_vocab([train_sentence1, train_sentence2, valid_sentence1, valid_sentence2, test_sentence1, test_sentence2])

    vocab_fw = {}
    vocab_bw = {}
    index = 0
    for token in vocab.idx_to_token:
        vocab_fw[token] = index
        vocab_bw[index] = token
        index += 1

    train_sentence1 = np.array(text_to_data(vocab_fw, train_sentence1, length))
    train_sentence2 = np.array(text_to_data(vocab_fw, train_sentence2, length))
    valid_sentence1 = np.array(text_to_data(vocab_fw, valid_sentence1, length))
    valid_sentence2 = np.array(text_to_data(vocab_fw, valid_sentence2, length))
    test_sentence1 = np.array(text_to_data(vocab_fw, test_sentence1, length))
    test_sentence2 = np.array(text_to_data(vocab_fw, test_sentence2, length))

    with open(save, 'wb') as f:
        pickle.dump((vocab_fw, vocab_bw, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2), f)


def load_data(file):
    with open(file, 'rb') as f:
        vocab_fw, vocab_bw, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2 = pickle.load(f)
    return vocab_fw, vocab_bw, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2


if __name__ == "__main__":
    dump_data("data/snli_1.0/snli_1.0_train.txt", "data/snli_1.0/snli_1.0_dev.txt", "data/snli_1.0/snli_1.0_test.txt", "data/snli_data.pkl")

