import sys
sys.path.append("..")

import pickle, os, random, collections
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from datetime import datetime
from pathlib import Path
import logging

import config as cfg
import data
import gpt_data
import gpt_model
import optimizer as optim

import torch
import torch.utils.data
import torch.nn.functional as F


def train_epoch(config, epoch, model, lm_loss_fn, ns_loss_fn, optimizer, data_loader):
    losses = collections.deque(maxlen=len(data_loader))
    model.train()

    with tqdm(total=len(data_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(data_loader):
            value = tuple(t.to(config.device) for t in value)
            input_ids, segment_ids, lm_label_ids, is_next = value
            lm_label = lm_label_ids[:, 1:].contiguous()

            optimizer.zero_grad()

            lm_logit, snli_logit = model(input_ids, segment_ids)

            lm_loss = lm_loss_fn(lm_logit.view(-1, lm_logit.size(2)), lm_label.view(-1))
            ns_loss = ns_loss_fn(snli_logit.view(-1, 2), is_next.view(-1))
            loss = lm_loss + ns_loss
            loss_val = loss.item()
            losses.append(loss_val)
            
            loss.backward()
            optimizer.step_and_update_lr()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")


def train_model(config, vocab, model):
    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    ns_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09),
        # torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=0.0001),
        config.d_embed, 4000)

    for epoch in trange(config.n_epoch, desc="Epoch"):
        train_loader = gpt_data.build_pretrain_loader(epoch, vocab, config.n_batch)

        train_epoch(config, epoch, model, lm_loss_fn, ns_loss_fn, optimizer, train_loader)

        model.decoder.save(epoch, "gpt_pretrain_final.pth")


def main():
    config = cfg.Config.load("gpt_config.json")

    vocab, train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2, max_sentence1, max_sentence2, max_sentence_all = data.load_data("../data/snli_data.pkl")

    # cuda or cpu
    config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    config.n_vocab = len(vocab)
    config.n_enc_vocab = len(vocab)
    config.n_dec_vocab = len(vocab)
    config.i_pad = vocab["<pad>"]
    config.n_batch = 64
    config.n_epoch = 100

    model = gpt_model.GPTPretrain(config)
    if os.path.isfile("gpt_pretrain_final.pth"):
        model.decoder.load("gpt_pretrain_final.pth")
        print(">>>> load state dict from: ", "gpt_pretrain_final.pth")
    model.to(config.device)

    print(config)
    train_model(config, vocab, model)


if __name__ == "__main__":
    main()

