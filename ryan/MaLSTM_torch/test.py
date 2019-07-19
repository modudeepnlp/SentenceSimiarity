"""
MALSTM: https://github.com/fionn-mac/Manhattan-LSTM/blob/master/PyTorch/main.py

"""

import os
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable

import random
import numpy as np
import logging
import json
import matplotlib.pyplot as plt

from models.malstm import Manhattan_LSTM
from utils.data import Data
from utils import configs

import preprocessing.custom_dataset as custom_dataset
from torch.utils.data import DataLoader

import config as config


config_path = 'config/configs.malstm.json'
args = configs.Config.load(config_path)

print('Model Parameters:')
print('Hidden Size                  :', args.hidden_size)
print('Batch Size                   :', args.batch_size)
print('Max. input length            :', args.max_len)
print('Learning rate                :', args.learning_rate)
print('Number of Epochs             :', args.num_iters)
print('--------------------------------------\n')

# print('Reading Data.')
# data = Data(args.data_name, args.data_file, args.training_ratio, args.max_len)

# print('\n')
# print('Number of training samples        :', len(data.x_train))
# print('Number of validation samples      :', len(data.x_val))
# print('Maximum sequence length           :', args.max_len)
# print('\n')

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))

if not os.path.exists(args.data_out):
	os.makedirs(args.data_out)

print('Building Model')

dataset = custom_dataset.Custom_dataset()
train_data, test_data, dev_data = dataset.get_data()

train_loader = DataLoader(train_data,
                        batch_size=args.batch ,
                        shuffle=True,
                        num_workers=config.cpu_processor,
                        drop_last=True)

test_loader = DataLoader(test_data,
                        batch_size=config.batch,
                        shuffle=False,
                        num_workers=config.cpu_processor,
                        drop_last=True)

dev_loader = DataLoader(dev_data,
                    batch_size=config.batch,
                    shuffle=False,
                    num_workers=config.cpu_processor,
                    drop_last=True)


model = Manhattan_LSTM(args.batch, args.hidden_size, [len(dataset.vocab_list), args.embedding_dim], use_embedding=False, train_embedding=True)
model.to(device)
model.init_weights()

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
loss_function = nn.CrossEntropyLoss()

# 훈련
step_list = []
loss_list = []
acc_test_list = []
acc_dev_list = []
step = 0

def evaluation(loader):
    total = 0
    correct = 0
    for n, (label, sent1, sent2) in enumerate(loader):
        label = Variable(label.to(device))
        sent1 = Variable(torch.stack(sent1).to(device))
        sent2 = Variable(torch.stack(sent2).to(device))

        init_hidden = model.init_hidden(config.batch)
        out =  model((sent1, sent2), init_hidden)
        _, pred = torch.max(out.data, 1)
        total += label.size(0) # batch size
        correct += (pred == label).sum()
    acc = 100 * (correct.cpu().numpy()/total)
    return acc

for i in range(config.epoch):
    print("epoch = ", i)
    for n, (label, sent1, sent2) in enumerate(train_loader):
        optimizer.zero_grad()  # 초기화
        init_hidden = model.init_hidden(config.batch)

        label = Variable(label.to(device))
        sent1 = Variable(torch.stack(sent1).to(device))
        sent2 = Variable(torch.stack(sent2).to(device))

        # sent = [sent1, sent2]

        logit = model((sent1, sent2), init_hidden)
        loss = loss_function(logit, label)
        loss.backward()
        optimizer.step()
        step += 1
        if n % 500 == 0:
            print("epoch : ", i, " step : ", n, " loss : ", loss.item())
            step_list.append(step)
            loss_list.append(loss)
            acc_test = evaluation(test_loader)
            acc_dev = evaluation(dev_loader)
            acc_test_list.append(acc_test)
            acc_dev_list.append(acc_dev)

            print("dev acc : ", acc_dev_list)
            print("test acc : ", acc_test_list)


    # Loss 그래프
    # plt.plot(step_list, loss_list, 'r--')
    # plt.legend(['Training Loss'])
    # plt.xlabel('Step')
    # plt.ylabel('Loss')
    # plt.savefig('Train.png')
    #
    # # Acc 그래프
    # plt.plot(step_list, acc_test_list, 'b--')
    # plt.plot(step_list, acc_dev_list, 'g--')
    # plt.legend(['Test acc', 'dev acc'])
    # plt.savefig('Acc.png')

    print("dev acc : ", acc_dev_list)
    print("test acc : ", acc_test_list)
