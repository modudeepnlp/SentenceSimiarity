import gluonnlp as nlp


regxs = list("[@_!#$%^&*()<>?/\|}{~:]'\".,-;`+=")

def strip_string(string):
    for regx in regxs:
        string = string.replace(regx, " " + regx + " ")
    return string.lower().strip()


def load_tokens(file):
    tokens = []
    with open(file, "r") as f:
        index = 0
        for line in f:
            if 0 < index:
                parts = line.strip().split('\t')
                for token in strip_string(parts[5]).split():
                    tokens.append(token)
                for token in strip_string(parts[6]).split():
                    tokens.append(token)
            index += 1
    return tokens


def main():
    tokens = []
    tokens.extend(load_tokens("Data/snli_1.0/snli_1.0_train.txt"))
    tokens.extend(load_tokens("Data/snli_1.0/snli_1.0_dev.txt"))
    tokens.extend(load_tokens("Data/snli_1.0/snli_1.0_test.txt"))

    counter = nlp.data.count_tokens(tokens)
    vocab = nlp.Vocab(counter, bos_token=None, eos_token=None)
    with open("Data/vocab.txt", "w") as f:
        for token in vocab.idx_to_token:
            f.write(token)
            f.write("\n")


if __name__ == "__main__":
    main()