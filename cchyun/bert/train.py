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
import model as bert_model


def eval_epoch(config, epoch, model, data_loader, mode):
    matchs = []
    model.eval()

    with tqdm(total=len(data_loader), desc=f"{mode}") as pbar:
        for i, value in enumerate(data_loader):
            snli_label, sentences, segment = map(lambda v: v.to(config.device), value)
            lm_label = sentences[:, 1:].contiguous()

            ogit = model(sentences, segment)
            _, indices = ogit.max(1)

            match = torch.eq(indices, snli_label).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) * 100 / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) * 100 / len(matchs) if 0 < len(matchs) else 0


def train_epoch(config, epoch, model, loss_fn, optimizer, scheduler, data_loader):
    losses = []
    model.train()

    with tqdm(total=len(data_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(data_loader):
            snli_label, sentences, segment = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()

            logit = model(sentences, segment)

            loss = loss_fn(logit, snli_label)
        
            loss_val = loss.item()
            losses.append(loss_val)

            loss.backward()

            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)


def train_model(cuda, vocab_file, data_pkl, save_file, save_pretrain_file):
    config = cfg.Config.load("config.json")

    vocab = global_data.load_vocab(vocab_file)
    train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2 = global_data.load_snli(data_pkl)

    # cuda or cpu
    config.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
    config.n_vocab = len(vocab)
    config.n_enc_vocab = len(vocab)
    config.n_dec_vocab = len(vocab)
    config.i_pad = global_data.PAD_ID
    print(config)

    best_epoch, best_loss, best_val, best_test = 0, 0, 0, 0
    model = bert_model.SNLI(config)
    if os.path.isfile(save_file):
        model.load(save_file)
        print(">>>> load state dict from: ", save_file)
    elif os.path.isfile(save_pretrain_file):
        epoch = model.bert.load(save_pretrain_file)
        print(">>>> load state dict from: ", save_pretrain_file, "epoch:", epoch)
    model.to(config.device)

    train_loader = data.build_data_loader(train_label, train_sentence1, train_sentence2, config.n_batch)
    # train_loader = data.build_data_loader(test_label, test_sentence1, test_sentence2, config.n_batch) ## only for fast test
    valid_loader = data.build_data_loader(valid_label, valid_sentence1, valid_sentence2, config.n_batch)
    test_loader = data.build_data_loader(test_label, test_sentence1, test_sentence2, config.n_batch)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    t_total = len(train_loader) * config.n_epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.learning_rate, eps=config.adam_epsilon)
    scheduler = optim.WarmupLinearSchedule(optimizer, warmup_steps=config.warmup_steps, t_total=t_total)
    
    best_epoch, best_loss, best_val, best_test = None, None, None, None
    for epoch in trange(config.n_epoch, desc="Epoch"):
        score_loss = train_epoch(config, epoch, model, loss_fn, optimizer, scheduler, train_loader)
        score_val = eval_epoch(config, epoch, model, valid_loader, "Valid")
        score_test = eval_epoch(config, epoch, model, test_loader, "Test")

        if best_test is None or best_test < score_test:
            model.save(epoch, score_loss, score_val, score_test, save_file)
            best_epoch, best_loss, best_val, best_test = epoch, score_loss, score_val, score_test
            print(f">>>>>>> model saved at {save_file} {best_epoch} {best_loss:.3f} {best_val:.3f} {best_test:.3f}")
        else:
            print(f">>>>>>> model not seved under accuracy {best_epoch} {best_loss:.3f} {best_val:.3f} {best_test:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="cuda", type=str, required=False,
                        help="cuda device # cuda / cuda:0 / cuda:1")
    parser.add_argument("--vocab", default="8000", type=str, required=False,
                        help="vocab size # 8000 / 1600")
    args = parser.parse_args()

    vocab_file = f"../data/m_snli_{args.vocab}.model"
    data_pkl = f"../data/snli_{args.vocab}_data.pkl"
    save_file = f"save_{args.vocab}.pth"
    save_pretrain_file = f"save_pretrain_{args.vocab}.pth"

    train_model(args.cuda, vocab_file, data_pkl, save_file, save_pretrain_file)
