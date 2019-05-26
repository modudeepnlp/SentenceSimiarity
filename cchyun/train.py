import data
import model
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data


def build_tensor(label, sentence1, sentence2, batch_size=256):
    torch_labe = torch.tensor(label, dtype=torch.long)
    torch_sentence1 = torch.tensor(sentence1, dtype=torch.long)
    torch_sentence2 = torch.tensor(sentence2, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(torch_labe, torch_sentence1, torch_sentence2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def main():
    vocab = data.load_vocab()
    train_labe, train_sentence1, train_sentence2 = data.load_data("Data/snli_1.0/snli_1.0_train.txt", vocab)
    dev_labe, dev_sentence1, dev_sentence2 = data.load_data("Data/snli_1.0/snli_1.0_dev.txt", vocab)
    test_labe, test_sentence1, test_sentence2 = data.load_data("Data/snli_1.0/snli_1.0_test.txt", vocab)

    config = { "n_embed": len(vocab), "d_embed": 32, "n_output": 3 }
    snli = model.SNLI(config=config)

    # seed = 1029
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(snli.parameters(), lr=0.01)

    train_loader = build_tensor(train_labe, train_sentence1, train_sentence2)
    dev_loader = build_tensor(dev_labe, dev_sentence1, dev_sentence2)
    test_loader = build_tensor(test_labe, test_sentence1, test_sentence2)

    for epoch in range(1):
        snli.train()
        index = 0
        for batch_label, batch_sentence1, batch_sentence2 in tqdm(train_loader, disable=True):
            optimizer.zero_grad()

            pred_label = snli(batch_label, batch_sentence1, batch_sentence2)

            loss = loss_fn(pred_label, batch_label)
            loss.backward()
            optimizer.step()

            print("epoch: {}, index: {}, loss: {}".format(epoch, index, loss.item()))
            index += len(batch_label)


if __name__ == "__main__":
    main()

