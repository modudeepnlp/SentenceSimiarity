import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import bert_model


class MTDNNModel(nn.Module):
    def __init__(self, config, task_defs):
        super().__init__()
        self.config = config
        self.task_defs = task_defs

        self.bert = bert_model.BertModel(config)
        self.scoring_list = nn.ModuleList()

        self.task_dict = {}
        task_indx = 0
        for task, define in task_defs._task_def_dic.items():
            self.task_dict[task] = task_indx
            task_indx += 1
            out_proj = nn.Linear(self.config.d_embed, define["n_class"])
            self.scoring_list.append(out_proj)
    
    def forward(self, input_ids, token_type_ids, attention_mask, task):
        enc_outputs, pooled_output, enc_self_attns = self.bert(input_ids, token_type_ids)
        out_proj = self.scoring_list[self.task_dict[task]]
        logits = out_proj(pooled_output)
        return logits
    
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])

