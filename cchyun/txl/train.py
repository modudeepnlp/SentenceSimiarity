import sys
sys.path.append("..")

import os, collections
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


def eval_epoch(config, epoch, model, data_loader, mode):
    matchs = []
    model.eval()

    mems = tuple()
    with tqdm(total=len(data_loader), desc=f"{mode}") as pbar:
        for i, value in enumerate(data_loader):
            snli_label, sentences = map(lambda v: v.to(config.device), value)
            lm_label = sentences[:, 1:].contiguous()

            lm_logit, snli_logit, _ = model(sentences, *mems)
            _, indices = snli_logit.max(1)

            match = torch.eq(indices, snli_label).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) * 100 / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) * 100 / len(matchs) if 0 < len(matchs) else 0


def train_epoch(config, epoch, model, lm_coef, lm_loss_fn, snli_loss_fn, optimizer, scheduler, data_loader):
    losses = collections.deque(maxlen=len(data_loader))
    model.train()

    mems = tuple()
    with tqdm(total=len(data_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(data_loader):
            snli_label, sentences = map(lambda v: v.to(config.device), value)
            lm_label = sentences[:, 1:].contiguous()

            optimizer.zero_grad()

            lm_logit, snli_logit, _ = model(sentences, *mems)

            lm_loss = lm_loss_fn(lm_logit.view(-1, lm_logit.size(2)), lm_label.view(-1))
            snli_loss = snli_loss_fn(snli_logit, snli_label)
            if 0 < lm_coef:
                loss = snli_loss + lm_coef * lm_loss
            else:
                loss = snli_loss
        
            loss_val = snli_loss.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)


def train_model():
    config = cfg.Config.load("config.json")

    vocab = global_data.load_vocab(vocab_file)
    train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2 = global_data.load_snli(data_pkl)

    # cuda or cpu
    config.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    config.n_vocab = len(vocab)
    config.n_enc_vocab = len(vocab)
    config.n_dec_vocab = len(vocab)
    config.i_pad = global_data.PAD_ID

    best_epoch, best_loss, best_val, best_test = 0, 0, 0, 0
    model = txl_model.SNLI(config)
    if os.path.isfile(save_file):
        model.load(save_file)
        print(">>>> load state dict from: ", save_file)
    elif os.path.isfile(save_pretrain_file):
        epoch = model.decoder.load(save_pretrain_file)
        print(">>>> load state dict from: ", save_pretrain_file, "epoch:", epoch)
    model.to(config.device)

    train_loader = data.build_data_loader(train_label, train_sentence1, train_sentence2, config.n_batch)
    # train_loader = data.build_data_loader(test_label, test_sentence1, test_sentence2, config.n_batch) ## only for fast test
    valid_loader = data.build_data_loader(valid_label, valid_sentence1, valid_sentence2, config.n_batch)
    test_loader = data.build_data_loader(test_label, test_sentence1, test_sentence2, config.n_batch)

    print(config)

    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.i_pad, reduction='mean')
    snli_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    
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
        score_loss = train_epoch(config, epoch, model, config.lm_coef, lm_loss_fn, snli_loss_fn, optimizer, scheduler, train_loader)
        score_val = eval_epoch(config, epoch, model, valid_loader, "Valid")
        score_test = eval_epoch(config, epoch, model, test_loader, "Test")

        if best_test is None or best_test < score_test:
            model.save(epoch, score_loss, score_val, score_test, save_file)
            best_epoch, best_loss, best_val, best_test = epoch, score_loss, score_val, score_test
            print(f">>>>>>> model saved at {save_file} {best_epoch} {best_loss:.3f} {best_val:.3f} {best_test:.3f}")
        else:
            print(f">>>>>>> model not seved under accuracy {best_epoch} {best_loss:.3f} {best_val:.3f} {best_test:.3f}")


prefix = 16000
vocab_file = f"../data/m_snli_{prefix}.model"
data_pkl = f"../data/snli_{prefix}_data.pkl"
save_file = f"save_{prefix}.pth"
save_pretrain_file = f"save_pretrain_{prefix}.pth"


if __name__ == '__main__':
    train_model()
