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
import gpt
import optimizer as optim

import torch
import torch.utils.data


def build_tensor(label, sentence1s, sentence2s, device, batch_size):
    torch_labe = torch.tensor(label, dtype=torch.long).to(device)

    dataset = data.SimDataset(label, sentence1s, sentence2s, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data.collate_fn)
    return loader


def train_model(config, train_loader, valid_loader, test_loader, log=True):
    # snli = hbmp.SNLI(config=config)
    # snli = transformer.SNLI(config=config)
    snli = gpt.SNLI(config=config)
    snli.to(config.device)

    seed = 1029
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.i_pad, reduction='mean')
    snli_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    # optimizer = torch.optim.Adam(snli.parameters(), lr=config.learning_rate)
    optimizer = optim.ScheduledOptim(
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
        lm_coef = config.lm_coef
        for i, value in enumerate(train_loader, 0):
            uids, snli_label, sentences = value
            lm_label = sentences[:, 1:].contiguous()

            optimizer.zero_grad()

            lm_logit, snli_logit = snli(sentences)

            lm_loss = lm_loss_fn(lm_logit.view(-1, lm_logit.size(2)), lm_label.view(-1))
            snli_loss = snli_loss_fn(snli_logit, snli_label)

            if 0 < lm_coef:
                loss = snli_loss + lm_coef * lm_loss
            else:
                loss = snli_loss

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
            uids, snli_label, sentences = value
            lm_logit, snli_logit = snli(sentences)
            _, indices = snli_logit.max(1)
            match = torch.eq(indices, snli_label).detach()
            valid_match.extend(match.cpu())
        valid_accuracy = np.sum(valid_match) * 100 / len(valid_match)
        valid_score.append(valid_accuracy)

        #
        # test evaluate
        #
        test_match = []
        for i, value in enumerate(test_loader, 0):
            uids, snli_label, sentences = value
            lm_logit, snli_logit = snli(sentences)
            _, indices = snli_logit.max(1)
            match = torch.eq(indices, snli_label).detach()
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
    # config = cfg.Config.load("config.transformer.json")
    config = cfg.Config.load("config.gpt.json")

    vocab, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2, max_sentence1, max_sentence2, max_sentence_all = data.load_data("data/snli_data.pkl")

    # cuda or cpu
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.n_vocab = len(vocab)
    config.n_enc_vocab = len(vocab)
    config.n_dec_vocab = len(vocab)
    config.n_enc_seq = max_sentence_all
    config.n_dec_seq = max_sentence_all
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

