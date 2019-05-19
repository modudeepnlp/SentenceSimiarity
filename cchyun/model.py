"""
SNLI Simple model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SNLI(nn.Module):
    def __init__(self, config):
        super(SNLI, self).__init__()

        self.embed = nn.Embedding(config["n_embed"], config["d_embed"])
        self.layer1 = nn.Linear(config["d_embed"], 32)
        self.layer2 = nn.Linear(config["d_embed"], 32)
        self.output = nn.Linear(32, config["n_output"])

    def forward(self, label, sentence1, sentence2):
        sentence1_embed = self.embed(sentence1)
        sentence1_ctx = self.layer1(sentence1_embed)
        sentence1_ctx = sentence1_ctx.mean(1)

        sentence2_embed = self.embed(sentence2)
        sentence2_ctx = self.layer2(sentence2_embed)
        sentence2_ctx = sentence2_ctx.mean(1)

        output = self.output(sentence1_ctx - sentence2_ctx)
        return output

