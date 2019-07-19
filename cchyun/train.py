import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
from datetime import datetime
import logging

import data
import config as cfg
import hbmp
import transformer

import torch
import torch.utils.data


def build_tensor(label, sentence1, sentence2, device, batch_size):
    torch_labe = torch.tensor(label, dtype=torch.long).to(device)
    torch_sentence1 = torch.tensor(sentence1, dtype=torch.long).to(device)
    torch_sentence2 = torch.tensor(sentence2, dtype=torch.long).to(device)
    dataset = torch.utils.data.TensorDataset(torch_labe, torch_sentence1, torch_sentence2)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def train_model(config, train_loader, valid_loader, test_loader, log=True):
    # snli = hbmp.SNLI(config=config)
    snli = transformer.SNLI(config=config)
    snli.to(config.device)

    seed = 1029
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    loss_fn = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(snli.parameters(), lr=config.learning_rate)
    optimizer = transformer.ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, snli.parameters()), betas=(0.9, 0.98), eps=1e-09),
        config.d_embed, 4000)

    epochs = []
    valid_score = []
    test_score = []

    min_loss = 100
    max_dev = 0
    max_test = 0

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
            _, indices = pred_label.max(1)
            # print(indices)
            loss = loss_fn(pred_label, batch_label)
            loss.backward()
            # optimizer.step()
            optimizer.step_and_update_lr()

            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        snli.eval()
    
        #
        # dev evaluate
        #
        valid_match = []
        for i, value in enumerate(valid_loader, 0):
            batch_label, batch_sentence1, batch_sentence2 = value
            pred_label = snli(batch_sentence1, batch_sentence2)
            _, indices = pred_label.max(1)
            match = torch.eq(indices, batch_label).detach()
            valid_match.extend(match.cpu())
        valid_accuracy = np.sum(valid_match) * 100 / len(valid_match)
        valid_score.append(valid_accuracy)

        #
        # test evaluate
        #
        test_match = []
        for i, value in enumerate(test_loader, 0):
            batch_label, batch_sentence1, batch_sentence2 = value
            pred_label = snli(batch_sentence1, batch_sentence2)
            _, indices = pred_label.max(1)
            match = torch.eq(indices, batch_label).detach()
            test_match.extend(match.cpu())
        test_accuracy = np.sum(test_match) * 100 / len(test_match)
        test_score.append(test_accuracy)
        
        min_loss = min(min_loss, train_loss)
        max_dev = max(max_dev, valid_accuracy)
        max_test = max(max_test, test_accuracy)
        if log:
            logging.warning("[%2d], loss: %.3f, dev: %.3f, test: %.3f" % (epoch + 1, train_loss, valid_accuracy, test_accuracy))
    
    del snli

    # if log:
    #     df = pd.DataFrame(data=np.array([valid_score, test_score]), columns=epochs, index=["dev", "test"])
    #     logging.warning(df)
    # plt.plot(epochs, valid_score, label='dev')
    # plt.plot(epochs, test_score, label='test')
    # plt.show()

    return min_loss, max_dev, max_test
    

def main():
    # config = cfg.Config.load("config.hbmp.json")
    config = cfg.Config.load("config.transformer.json")

    vocab, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2 = data.load_data("data/snli_data.pkl")
 
    # cuda or cpu
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # config.n_vocab = len(vocab)
    config.n_enc_vocab = len(vocab)
    config.n_dec_vocab = len(vocab)
    config.n_enc_seq = max(len(train_sentence1[0]), len(train_sentence2[0]))
    config.n_dec_seq = max(len(train_sentence1[0]), len(train_sentence2[0]))
    config.i_pad = vocab["<pad>"]

    train_loader = build_tensor(train_label, train_sentence1, train_sentence2, config.device, config.n_batch)
    # train_loader = build_tensor(test_label, test_sentence1, test_sentence2, config.device, config.n_batch) ## only for fast test
    valid_loader = build_tensor(valid_label, valid_sentence1, valid_sentence2, config.device, config.n_batch)
    test_loader = build_tensor(test_label, test_sentence1, test_sentence2, config.device, config.n_batch)

    configs = []
    configs.append(config)

    timestamp = datetime.today().strftime("%Y%m%d%H%M%S")
    logging.basicConfig(filename='log/train-{}.log'.format(timestamp), format='%(asctime)s %(message)s', level=logging.DEBUG)
    logging.getLogger().addHandler(logging.StreamHandler())

    for config in configs:
        logging.warning("{}".format(config))
        train_loss, valid_accuracy, test_accuracy = train_model(config, train_loader, valid_loader, test_loader)
        logging.warning("loss: %.3f, dev: %.3f, test: %.3f" % (train_loss, valid_accuracy, test_accuracy))


if __name__ == "__main__":
    main()

