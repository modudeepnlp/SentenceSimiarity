import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import bert_model


class Classifier(nn.Module):
    def __init__(self, x_size, y_size, opt, prefix='decoder', dropout=None):
        super(Classifier, self).__init__()
        self.opt = opt
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(prefix), 0))
        else:
            self.dropout = dropout
        self.merge_opt = opt.get('{}_merge_opt'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)

        if self.merge_opt == 1:
            self.proj = nn.Linear(x_size * 4, y_size)
        else:
            self.proj = nn.Linear(x_size * 2, y_size)

        if self.weight_norm_on:
            self.proj = weight_norm(self.proj)

    def forward(self, x1, x2, mask=None):
        if self.merge_opt == 1:
            x = torch.cat([x1, x2, (x1 - x2).abs(), x1 * x2], 1)
        else:
            x = torch.cat([x1, x2], 1)
        x = self.dropout(x)
        scores = self.proj(x)
        return scores


class FlatSimV2(nn.Module):
    def __init__(self, x_size, y_size, opt={}, prefix='seqatt', dropout=None):
        super(FlatSimV2, self).__init__()
        assert x_size == y_size
        self.opt = opt
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.linear = nn.Linear(x_size * 4, 1)
        if self.weight_norm_on:
            self.linear = weight_norm(self.linear)
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout

    def forward(self, x, y, x_mask):
        """
        x = batch * len * h1
        y = batch * h2
        x_mask = batch * len
        """
        x = self.dropout(x)
        y = self.dropout(y)
        y = y.unsqueeze(1).expand_as(x)

        flat_x = torch.cat([x, y, x * y, torch.abs(x - y)], 2).contiguous().view(x.size(0) * x.size(1), -1)
        flat_scores = self.linear(flat_x)
        scores = flat_scores.contiguous().view(x.size(0), -1)
        scores.data.masked_fill_(x_mask.data, -float('inf'))

        return scores


class DropoutWrapper(nn.Module):
    """
    This is a dropout wrapper which supports the fix mask dropout
    """
    def __init__(self, dropout_p=0, enable_vbp=True):
        super(DropoutWrapper, self).__init__()
        """variational dropout means fix dropout mask
        ref: https://discuss.pytorch.org/t/dropout-for-rnns/633/11
        """
        self.enable_variational_dropout = enable_vbp
        self.dropout_p = dropout_p

    def forward(self, x):
        """
            :param x: batch * len * input_size
        """
        if self.training == False or self.dropout_p == 0:
            return x

        if len(x.size()) == 3:
            mask = 1.0 / (1-self.dropout_p) * torch.bernoulli((1-self.dropout_p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1))
            mask.requires_grad = False
            return mask.unsqueeze(1).expand_as(x) * x
        else:
            return F.dropout(x, p=self.dropout_p, training=self.training)


class FlatSimilarityWrapper(nn.Module):
    def __init__(self, x1_dim, x2_dim, prefix='attention', opt={}, dropout=None):
        super(FlatSimilarityWrapper, self).__init__()
        self.score_func_str = opt.get('{}_att_type'.format(prefix), 'none').lower()
        self.att_dropout = DropoutWrapper(opt.get('{}_att_dropout'.format(prefix), 0))
        self.score_func = None
        if self.score_func_str == 'bilinear':
            self.score_func = BilinearFlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'simple':
            self.score_func = SimpleFlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        elif self.score_func_str == 'flatsim':
            self.score_func = FlatSim(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)
        else:
            self.score_func = FlatSimV2(x1_dim, x2_dim, prefix=prefix, opt=opt, dropout=dropout)

    def forward(self, x1, x2, mask):
        scores = self.score_func(x1, x2, mask)
        return scores


class LinearSelfAttn(nn.Module):
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """
    def __init__(self, input_size, dropout=None):
        super(LinearSelfAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)
        self.dropout = dropout

    def forward(self, x, x_mask):
        x = self.dropout(x)
        x_flat = x.contiguous().view(-1, x.size(-1))
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, 1)
        return alpha.unsqueeze(1).bmm(x).squeeze(1)


class SelfAttnWrapper(nn.Module):
    def __init__(self, input_size, prefix='attn_sum', opt={}, dropout=None):
        super(SelfAttnWrapper, self).__init__()
        """
        Self att wrapper, support linear and MLP
        """
        attn_type = opt.get('{}_type'.format(prefix), 'linear')
        if attn_type == 'mlp':
            self.att = MLPSelfAttn(input_size, prefix, opt, dropout)
        else:
            self.att = LinearSelfAttn(input_size, dropout)

    def forward(self, x, x_mask):
        return self.att(x, x_mask)


class SANClassifier(nn.Module):
    """Implementation of Stochastic Answer Networks for Natural Language Inference, Xiaodong Liu, Kevin Duh and Jianfeng Gao
    https://arxiv.org/abs/1804.07888
    """
    def __init__(self, x_size, h_size, label_size, opt={}, prefix='decoder', dropout=None):
        super(SANClassifier, self).__init__()
        if dropout is None:
            self.dropout = DropoutWrapper(opt.get('{}_dropout_p'.format(self.prefix), 0))
        else:
            self.dropout = dropout
        self.prefix = prefix
        self.query_wsum = SelfAttnWrapper(x_size, prefix='mem_cum', opt=opt, dropout=self.dropout)
        self.attn = FlatSimilarityWrapper(x_size, h_size, prefix, opt, self.dropout)
        self.rnn_type = '{}{}'.format(opt.get('{}_rnn_type'.format(prefix), 'gru').upper(), 'Cell')
        self.rnn =getattr(nn, self.rnn_type)(x_size, h_size)
        self.num_turn = opt.get('{}_num_turn'.format(prefix), 5)
        self.opt = opt
        self.mem_random_drop = opt.get('{}_mem_drop_p'.format(prefix), 0)
        self.mem_type = opt.get('{}_mem_type'.format(prefix), 0)
        self.weight_norm_on = opt.get('{}_weight_norm_on'.format(prefix), False)
        self.label_size = label_size
        self.dump_state = opt.get('dump_state_on', False)
        self.alpha = Parameter(torch.zeros(1, 1), requires_grad=False)
        if self.weight_norm_on:
            self.rnn = WN(self.rnn)

        self.classifier = Classifier(x_size, self.label_size, opt, prefix=prefix, dropout=self.dropout)

    def forward(self, x, h0, x_mask=None, h_mask=None):
        h0 = self.query_wsum(h0, h_mask)
        if type(self.rnn) is nn.LSTMCell:
            c0 = h0.new(h0.size()).zero_()
        scores_list = []
        for turn in range(self.num_turn):
            att_scores = self.attn(x, h0, x_mask)
            x_sum = torch.bmm(F.softmax(att_scores, 1).unsqueeze(1), x).squeeze(1)
            scores = self.classifier(x_sum, h0)
            scores_list.append(scores)
            # next turn
            if self.rnn is not None:
                h0 = self.dropout(h0)
                if type(self.rnn) is nn.LSTMCell:
                    h0, c0 = self.rnn(x_sum, (h0, c0))
                else:
                    h0 = self.rnn(x_sum, h0)
        if self.mem_type == 1:
            mask = generate_mask(self.alpha.data.new(x.size(0), self.num_turn), self.mem_random_drop, self.training)
            mask = [m.contiguous() for m in torch.unbind(mask, 1)]
            tmp_scores_list = [mask[idx].view(x.size(0), 1).expand_as(inp) * F.softmax(inp, 1) for idx, inp in enumerate(scores_list)]
            scores = torch.stack(tmp_scores_list, 2)
            scores = torch.mean(scores, 2)
            scores = torch.log(scores)
        else:
            scores = scores_list[-1]
        if self.dump_state:
            return scores, scores_list
        else:
            return scores


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
            if define["task_type"] == "Classification":
                # out_proj = SANClassifier(self.config.d_embed, self.config.d_embed, define["n_class"], prefix='answer', dropout=config.dropout)
                out_proj = nn.Linear(self.config.d_embed, define["n_class"])
            else:
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

