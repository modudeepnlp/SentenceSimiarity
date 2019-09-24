import sys
sys.path.append("..")

import os
from tqdm import tqdm, trange
import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F

import config as cfg
import data
import optimizer as optim
import txl_data
import model as txl_model


def train_epoch(config, epoch, model, loss_fn, optimizer, train_iter):
    losses = []
    model.train()

    mems = tuple()
    with tqdm(total=len(train_iter), desc=f"Train {epoch}") as pbar:
        for i, (inputs, labels, seq_len) in enumerate(train_iter):
            optimizer.zero_grad()

            logit, mems = model(inputs, *mems)

            loss = loss_fn(logit.view(-1, logit.size(2)), labels.view(-1))
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step_and_update_lr()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")


def train_model():
    config = cfg.Config.load("config.json")

    vocab = data.load_data("../data/snli_data.pkl")[0]
    token_ids = txl_data.load_pretrain("large")

    config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    config.n_vocab = len(vocab)
    config.n_enc_vocab = len(vocab)
    config.n_dec_vocab = len(vocab)
    config.i_pad = vocab["<pad>"]
    config.n_batch = 64
    config.n_epoch = 100

    offset = 0
    model = txl_model.TransformerXL(config)
    if os.path.isfile("save_pretrain_final.pth"):
        offset = model.decoder.load("save_pretrain_final.pth") + 1
        print(">>>> load state dict from: ", "save_pretrain_final.pth")
    model.to(config.device)

    train_iter = txl_data.TXLIterator(config, token_ids)

    print(config)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = optim.ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09),
        config.d_embed, 4000)
    
    for step in trange(config.n_epoch, desc="Epoch"):
        epoch = step + offset
        train_epoch(config, epoch, model, loss_fn, optimizer, train_iter)
        model.decoder.save(epoch, "save_pretrain_final.pth")


if __name__ == '__main__':
    train_model()
