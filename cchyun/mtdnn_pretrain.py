import pickle, os, random, collections
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from datetime import datetime
from pathlib import Path
import logging

import config as cfg
import data
import mtdnn_data
import mtdnn_model
import optimizer as optim

import torch
import torch.utils.data
import torch.nn.functional as F


def train_epoch(epoch, mtdnn, optimizer, all_indices, train_iters):
    losses = collections.deque(maxlen=len(all_indices))
    mtdnn.train()

    with tqdm(total=len(all_indices), desc=f"Train {epoch}") as pbar:
        for i, task_id in enumerate(all_indices):
            batch_meta, batch_data = next(train_iters[task_id])

            labels = batch_data[batch_meta['label']]
            task_id = batch_meta['task_id']
            task = batch_meta['task']
            task_type = batch_meta['task_type']
            inputs = batch_data[:batch_meta['input_len']]

            input_ids = inputs[0]
            token_type_ids = inputs[1]
            attention_mask = inputs[2]

            optimizer.zero_grad()

            logits = mtdnn(input_ids, token_type_ids, attention_mask, task)
            if task_type == mtdnn_data.TaskType.Regression:
                loss = torch.mean(F.mse_loss(logits.squeeze(), labels))
            else:
                loss = F.cross_entropy(logits, labels)
            loss_val = loss.item()
            losses.append(loss_val)
            
            loss.backward()
            optimizer.step_and_update_lr()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")


def train_model(config, vocab, model, train_data_list):
    optimizer = optim.ScheduledOptim(
        torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), betas=(0.9, 0.98), eps=1e-09),
        config.d_embed, 4000)

    train_iters = [iter(item) for item in train_data_list]

    for epoch in trange(config.n_epoch, desc="Epoch"):
        for train_data in train_data_list:
            train_data.reset()

        all_indices = []
        for i in range(len(train_data_list)):
            all_indices += [i] * len(train_data_list[i])
        # random.shuffle(all_indices)
        train_epoch(epoch, model, optimizer, all_indices, train_iters)

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
    config.n_batch = 128
    config.n_epoch = 10

    task_defs = mtdnn_data.TaskDefs("mtdnn_task_def.yml")
    
    model = mtdnn_model.MTDNNModel(config, task_defs)
    if os.path.isfile("mtdnn_final.pth"):
        model.load("mtdnn_final.pth")
        print(">>>> load state dict from: ", "mtdnn_final.pth")
    elif os.path.isfile("bert_pretrain_final.pth"):
        model.bert.load("bert_pretrain_final.pth")
        print(">>>> load state dict from: ", "bert_pretrain_final.pth")
    model.to(config.device)

    datasets = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "scitail", "snli", "sst", "stsb", "wnli"]
    train_data_list = mtdnn_data.build_data_loader(config, vocab, task_defs, "train", datasets)
    # val_data_list = mtdnn_data.build_data_loader(config, vocab, task_defs, "dev", datasets)
    # test_data_list = mtdnn_data.build_data_loader(config, vocab, task_defs, "test", datasets)

    print(config)
    train_model(config, vocab, model, train_data_list)


if __name__ == "__main__":
    main()