import pickle, os, random, collections
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from datetime import datetime
from pathlib import Path
import logging

import torch
import torch.utils.data
import torch.nn.functional as F

import config as cfg
import data
import optimizer as optim
import bert_data
import mtdnn_data
import mtdnn_model


def eval_epoch(model, data_loader, mode):
    matchs = []
    model.eval()

    with tqdm(total=len(data_loader), desc=f"{mode}") as pbar:
        for i, value in enumerate(data_loader):
            uids, label, sentences, segments = value

            logit = model(sentences, segments, None, "snli")
            _, indices = logit.max(1)

            match = torch.eq(indices, label).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) * 100 / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) * 100 / len(matchs) if 0 < len(matchs) else 0


def train_epoch(epoch, model, loss_fn, optimizer, data_loader):
    losses = collections.deque(maxlen=len(data_loader))
    model.train()

    with tqdm(total=len(data_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(data_loader):
            uids, label, sentences, segments = value

            optimizer.zero_grad()

            logit = model(sentences, segments, None, "snli")

            loss = loss_fn(logit, label)
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            # optimizer.step()
            optimizer.step_and_update_lr()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)


def train_model(config, vocab, model, train_loader, valid_loader, test_loader):
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09),
        config.d_embed, 4000)

    for epoch in trange(config.n_epoch, desc="Epoch"):
        train_epoch(epoch, model, loss_fn, optimizer, train_loader)
        eval_epoch(model, valid_loader, "Valid")
        eval_epoch(model, test_loader, "Test")

        model.save("mtdnn_final.pth")


def main():
    config = cfg.Config.load("mtdnn_config.json")

    vocab, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2, max_sentence1, max_sentence2, max_sentence_all = data.load_data("data/snli_data.pkl")

    # cuda or cpu
    config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    config.n_vocab = len(vocab)
    config.n_enc_vocab = len(vocab)
    config.n_dec_vocab = len(vocab)
    config.i_pad = vocab["<pad>"]

    task_defs = mtdnn_data.TaskDefs("mtdnn_task_def.yml")
    
    model = mtdnn_model.MTDNNModel(config, task_defs)
    if os.path.isfile("mtdnn_final.pth"):
        model.load("mtdnn_final.pth")
        print(">>>> load state dict from: ", "mtdnn_final.pth")
    model.to(config.device)

    train_loader = bert_data.build_data_loader(train_label, train_sentence1, train_sentence2, config.device, config.n_batch)
    # train_loader = bert_data.build_data_loader(test_label, test_sentence1, test_sentence2, config.device, config.n_batch) ## only for fast test
    valid_loader = bert_data.build_data_loader(valid_label, valid_sentence1, valid_sentence2, config.device, config.n_batch)
    test_loader = bert_data.build_data_loader(test_label, test_sentence1, test_sentence2, config.device, config.n_batch)

    print(config)
    train_model(config, vocab, model, train_loader, valid_loader, test_loader)


if __name__ == "__main__":
    main()