import data
import model
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data


def build_tensor(label, sentence1, sentence2, device, batch_size=256):
    torch_labe = torch.tensor(label, dtype=torch.long).to(device)
    torch_sentence1 = torch.tensor(sentence1, dtype=torch.long).to(device)
    torch_sentence2 = torch.tensor(sentence2, dtype=torch.long).to(device)
    dataset = torch.utils.data.TensorDataset(torch_labe, torch_sentence1, torch_sentence2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def main():
    # cuda or cpu
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    vocab = data.load_vocab("data/vocab.txt")
    print("load_vocab .......................... %d" % len(vocab))
    train_labe, train_sentence1, train_sentence2 = data.load_data("data/snli_1.0/snli_1.0_train.txt", vocab)
    # train_labe, train_sentence1, train_sentence2 = data.load_data("data/snli_1.0/snli_1.0_test.txt", vocab) ## fast test only
    print("load_data train ..................... %d" % len(train_labe))
    dev_labe, dev_sentence1, dev_sentence2 = data.load_data("data/snli_1.0/snli_1.0_dev.txt", vocab)
    print("load_data dev ....................... %d" % len(dev_labe))
    test_labe, test_sentence1, test_sentence2 = data.load_data("data/snli_1.0/snli_1.0_test.txt", vocab)
    print("load_data test ...................... %d" % len(test_labe))

    config = model.SNLIConfig({
        "device": device, # cpu 또는 gpu 사용
        "n_embed": len(vocab), "d_embed": 32, "n_output": 3, "n_epoch": 10, "n_batch": 256,
        "n_layer": 1, "cells": 2, "dropout": 0.1 ## HBMP
    })
    snli = model.SNLI(config=config)
    snli.to(config.device)

    seed = 1029
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(snli.parameters(), lr=0.01)

    train_loader = build_tensor(train_labe, train_sentence1, train_sentence2, config.device, config.n_batch)
    dev_loader = build_tensor(dev_labe, dev_sentence1, dev_sentence2, config.device, config.n_batch)
    test_loader = build_tensor(test_labe, test_sentence1, test_sentence2, config.device, config.n_batch)

    epochs = []
    dev_score = []
    test_score = []

    for epoch in range(config.n_epoch):
        epochs.append(epoch + 1)
        
        #
        # train
        #
        snli.train()
        train_loss = 0
        for i, value in enumerate(train_loader, 0):
            batch_label, batch_sentence1, batch_sentence2 = value

            optimizer.zero_grad()

            pred_label = snli(batch_sentence1, batch_sentence2)
            loss = loss_fn(pred_label, batch_label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        snli.eval()
    
        #
        # dev evaluate
        #
        dev_match = np.zeros((len(dev_labe)))
        for i, value in enumerate(dev_loader, 0):
            batch_label, batch_sentence1, batch_sentence2 = value
            pred_label = snli(batch_sentence1, batch_sentence2)
            _, indices = pred_label.max(1)
            match = torch.eq(indices, batch_label).detach()
            dev_match[i * config.n_batch:(i+1) * config.n_batch] = match.cpu()
        dev_accuracy = np.sum(dev_match) * 100 / len(dev_match)
        dev_score.append(dev_accuracy)

        #
        # test evaluate
        #
        test_match = np.zeros((len(test_labe)))
        for i, value in enumerate(test_loader, 0):
            batch_label, batch_sentence1, batch_sentence2 = value
            pred_label = snli(batch_sentence1, batch_sentence2)
            _, indices = pred_label.max(1)
            match = torch.eq(indices, batch_label).detach()
            test_match[i * config.n_batch:(i+1) * config.n_batch] = match.cpu()
        test_accuracy = np.sum(test_match) * 100 / len(test_match)
        test_score.append(test_accuracy)
        
        print("[%d], loss: %.3f, dev: %.3f, test: %.3f" % (epoch + 1, train_loss, dev_accuracy, test_accuracy))


    df = pd.DataFrame(data=np.array([dev_score, test_score]), columns=epochs, index=["dev", "test"])
    print(df)

    # plt.plot(epochs, dev_score, label='dev')
    # plt.plot(epochs, test_score, label='test')
    # plt.show()


if __name__ == "__main__":
    main()

