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


def eval_epoch(model, data_loader, mode):
    matchs = []
    model.eval()

    mems = tuple()
    with tqdm(total=len(data_loader), desc=f"{mode}") as pbar:
        for i, value in enumerate(data_loader):
            snli_label, sentences = value
            lm_label = sentences[:, 1:].contiguous()

            lm_logit, snli_logit, _ = model(sentences, *mems)
            _, indices = snli_logit.max(1)

            match = torch.eq(indices, snli_label).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) * 100 / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) * 100 / len(matchs) if 0 < len(matchs) else 0


def train_epoch(epoch, model, lm_coef, lm_loss_fn, snli_loss_fn, optimizer, data_loader):
    losses = []
    model.train()

    mems = tuple()
    with tqdm(total=len(data_loader), desc=f"Train {epoch}") as pbar:
        for i, value in enumerate(data_loader):
            snli_label, sentences = value
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
            optimizer.step_and_update_lr()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)


def train_model():
    config = cfg.Config.load("config.json")

    vocab = global_data.load_vocab("../data/m_book.model")
    train_label, train_sentence1, train_sentence2, valid_label, valid_sentence1, valid_sentence2, test_label, test_sentence1, test_sentence2 = global_data.load_snli("../data/snli_data.pkl")

    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.n_vocab = len(vocab)
    config.n_enc_vocab = len(vocab)
    config.n_dec_vocab = len(vocab)
    config.i_pad = global_data.PAD_ID

    best_epoch, best_loss, best_val, best_test = 0, 0, 0, 0
    model = txl_model.SNLI(config)
    if os.path.isfile("save_final.pth"):
        best_epoch, best_loss, best_val, best_test = model.load("save_final.pth")
        print(">>>> load state dict from: ", "save_final.pth")
    elif os.path.isfile("save_pretrain_final.pth"):
        offset = model.decoder.load("save_pretrain_final.pth")
        print(">>>> load state dict from: ", "save_pretrain_final.pth", "epoch:", offset)
    model.to(config.device)

    train_loader = data.build_data_loader(train_label, train_sentence1, train_sentence2, config.device, config.n_batch)
    # train_loader = data.build_data_loader(test_label, test_sentence1, test_sentence2, config.device, config.n_batch) ## only for fast test
    valid_loader = data.build_data_loader(valid_label, valid_sentence1, valid_sentence2, config.device, config.n_batch)
    test_loader = data.build_data_loader(test_label, test_sentence1, test_sentence2, config.device, config.n_batch)

    print(config)

    lm_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=config.i_pad, reduction='mean')
    snli_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09),
        config.d_embed, 4000)
    
    for epoch in trange(config.n_epoch, desc="Epoch"):
        score_loss = train_epoch(epoch, model, config.lm_coef, lm_loss_fn, snli_loss_fn, optimizer, train_loader)
        score_val = eval_epoch(model, valid_loader, "Valid")
        score_test = eval_epoch(model, test_loader, "Test")

        if best_test is None or best_test < score_test:
            model.save(epoch, score_loss, score_val, score_test, "save_final.pth")
            best_epoch, best_loss, best_val, best_test = epoch, score_loss, score_val, score_test
            print(f">>>>>>> model saved at save_final.pth {best_epoch} {best_loss:.3f} {best_val:.3f} {best_test:.3f}")
        else:
            print(f">>>>>>> model not seved under accuracy {best_epoch} {best_loss:.3f} {best_val:.3f} {best_test:.3f}")


if __name__ == '__main__':
    train_model()
