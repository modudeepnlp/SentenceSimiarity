import sys
sys.path.append("..")

import os
from tqdm import tqdm, trange
import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F

import config as cfg
import global_data
import optimizer as optim
import data
import model as txl_model


def train_epoch(config, epoch, model, loss_fn, optimizer, scheduler, train_iter):
    losses = []
    model.train()

    mems = tuple()
    with tqdm(total=len(train_iter), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(train_iter):
            inputs, labels = map(lambda v: v.to(config.device), value)
            optimizer.zero_grad()

            logit, mems = model(inputs, *mems)

            loss = loss_fn(logit.view(-1, logit.size(2)), labels.view(-1))
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")


def train_model():
    config = cfg.Config.load("config.json")

    vocab = global_data.load_vocab(vocab_file)
    token_ids = data.load_pretrain(data_pkl)

    config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    config.n_vocab = len(vocab)
    config.n_enc_vocab = len(vocab)
    config.n_dec_vocab = len(vocab)
    config.i_pad = global_data.PAD_ID
    config.n_batch = 64
    config.n_epoch = 3

    offset = 0
    model = txl_model.TXLPretrain(config)
    if os.path.isfile(save_pretrain_file):
        offset = model.decoder.load(save_pretrain_file) + 1
        print(">>>> load state dict from: ", save_pretrain_file)
    model.to(config.device)

    train_iter = data.TXLIterator(config, token_ids)

    print(config)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    
    t_total = len(train_iter) * config.n_epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = optim.WarmupLinearSchedule(optimizer, warmup_steps=config.warmup_steps, t_total=t_total)
    
    for step in trange(config.n_epoch, desc="Epoch"):
        epoch = step + offset
        train_epoch(config, epoch, model, loss_fn, optimizer, scheduler, train_iter)
        model.decoder.save(epoch, save_pretrain_file)


prefix = 8000
vocab_file = f"../data/m_snli_{prefix}.model"
data_pkl = f"../data/pretrain_gpt_{prefix}_0.pkl"
save_pretrain_file = f"save_pretrain_{prefix}.pth"


if __name__ == '__main__':
    train_model()
