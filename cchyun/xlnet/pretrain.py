import sys
sys.path.append("..")

import os, argparse
from tqdm import tqdm, trange
import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F

import config as cfg
import global_data
import optimizer as optim
import data
import model as xlnet_model


def train_epoch(config, epoch, model, loss_fn, optimizer, scheduler, train_iter):
    losses = []
    model.train()

    mems = tuple()
    with tqdm(total=len(train_iter), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(train_iter):
            inp_k, seg_id, target, perm_mask, target_mapping, inp_q, target_mask = map(lambda v: v.to(config.device), value)
            optimizer.zero_grad()

            logit, mems = model(inp_k, mems, perm_mask)
            loss = loss_fn(logit.view(-1, logit.size(2)), target.view(-1))
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")


def train_model(cuda, vocab_file, data_pkl, save_pretrain_file):
    config = cfg.Config.load("config.json")

    vocab = global_data.load_vocab(vocab_file)
    features = data.load_pretrain(data_pkl)

    config.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    config.n_vocab = len(vocab)
    config.n_enc_vocab = len(vocab)
    config.n_dec_vocab = len(vocab)
    config.i_pad = global_data.PAD_ID
    config.n_batch = 64
    config.n_epoch = 30
    print(config)

    offset = 0
    model = xlnet_model.XLNETPretrain(config)
    if os.path.isfile(save_pretrain_file):
        offset = model.decoder.load(save_pretrain_file) + 1
        print(">>>> load state dict from: ", save_pretrain_file)
    model.to(config.device)

    train_iter = data.build_pretrain_loader(features, config.n_batch)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="cuda", type=str, required=False,
                        help="cuda device # cuda / cuda:0 / cuda:1")
    parser.add_argument("--vocab", default="8000", type=str, required=False,
                        help="vocab size # 8000 / 1600")
    args = parser.parse_args()

    vocab_file = f"../data/m_snli_{args.vocab}.model"
    data_pkl = f"../data/pretrain_xlnet_{args.vocab}_0.pkl"
    save_pretrain_file = f"save_pretrain_{args.vocab}.pth"

    train_model(args.cuda, vocab_file, data_pkl, save_pretrain_file)
