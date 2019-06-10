import gluonnlp as nlp
import pandas as pd


regxs = list("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=")

def strip_string(string):
    for regx in regxs:
        string = string.replace(regx, " " + regx + " ")
    return string.lower().strip()


def load_tokens(file):
    tokens = []
    dataset = pd.read_csv(file, sep="\t")
    for i, row in dataset.iterrows():
        if i != 0 and i % 1000 == 0:
                print("{0} : {1} sentences".format(file, i))
        if pd.isnull(row['sentence1']) == False:
            for token in strip_string(row['sentence1']).split():
                tokens.append(token)
        if pd.isnull(row['sentence2']) == False:
            for token in strip_string(row['sentence2']).split():
                tokens.append(token)
    return tokens


def main():
    tokens = []
    tokens.extend(load_tokens("data/snli_1.0/snli_1.0_train.txt"))
    tokens.extend(load_tokens("data/snli_1.0/snli_1.0_dev.txt"))
    tokens.extend(load_tokens("data/snli_1.0/snli_1.0_test.txt"))

    counter = nlp.data.count_tokens(tokens)
    vocab = nlp.Vocab(counter, bos_token=None, eos_token=None)
    with open("data/vocab.txt", "w") as f:
        for token in vocab.idx_to_token:
            f.write(token)
            f.write("\n")


if __name__ == "__main__":
    main()