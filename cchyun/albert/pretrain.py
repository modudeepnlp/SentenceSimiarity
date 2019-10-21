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
import model as albert_model


def train_epoch(config, epoch, model, loss_fn, optimizer, scheduler, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            input_ids, segment_ids, lm_label_ids, is_next = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()

            prediction_scores, seq_relationship_score, _ = model(input_ids, segment_ids)

            masked_lm_loss = loss_fn(prediction_scores.view(-1, config.n_vocab), lm_label_ids.view(-1))
            next_sentence_loss = loss_fn(seq_relationship_score.view(-1, 2), is_next.view(-1))
            loss = masked_lm_loss + next_sentence_loss
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")


def train_model(cuda, vocab_file, data_pkls, save_pretrain_file):
    config = cfg.Config.load("config.json")

    vocab = global_data.load_vocab(vocab_file)

    config.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    config.n_vocab = len(vocab)
    config.i_pad = global_data.PAD_ID
    config.n_batch = 24
    config.n_epoch = 3
    print(config)

    offset = 0
    model = albert_model.AlBertPretrain(config)
    if os.path.isfile(save_pretrain_file):
        offset = model.bert.load(save_pretrain_file) + 1
        print(">>>> load state dict from: ", save_pretrain_file)
    model.to(config.device)

    train_loader = None

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = None
    scheduler = None

    for step in trange(config.n_epoch, desc="Epoch"):
        epoch = step + offset
        if train_loader is not None:
            del train_loader
        data_pkl = data_pkls[epoch % len(data_pkls)]
        print(f"load pretrain data from {data_pkl}")
        train_loader = data.build_pretrain_loader(data_pkl, vocab, config.n_batch)
        if optimizer is None or scheduler is None:
            t_total = len(train_loader) * config.n_epoch
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
            optimizer = optim.RAdam(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
            scheduler = optim.WarmupLinearSchedule(optimizer, warmup_steps=config.warmup_steps, t_total=t_total)

        train_epoch(config, epoch, model, loss_fn, optimizer, scheduler, train_loader)
        model.bert.save(epoch, save_pretrain_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="cuda", type=str, required=False,
                        help="cuda device # cuda / cuda:0 / cuda:1")
    parser.add_argument("--vocab", default="8000", type=str, required=False,
                        help="vocab size # 8000 / 1600")
    args = parser.parse_args()

    vocab_file = f"../data/m_snli_{args.vocab}.model"
    data_pkls = [
        f"../data/pretrain_albert_{args.vocab}_0.json",
        f"../data/pretrain_albert_{args.vocab}_1.json",
        f"../data/pretrain_albert_{args.vocab}_2.json",
        f"../data/pretrain_albert_{args.vocab}_3.json"
    ]
    save_pretrain_file = f"save_pretrain_{args.vocab}.pth"

    train_model(args.cuda, vocab_file, data_pkls, save_pretrain_file)

