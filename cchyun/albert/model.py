import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# 참고: https://github.com/graykode/nlp-tutorial
#      https://github.com/jadore801120/attention-is-all-you-need-pytorch
#      https://github.com/JayParks/transformer
#      https://github.com/modudeepnlp/code_implementation/blob/master/codes/transformer/Transformer-Torch.py


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
    return pad_attn_mask.bool()


def get_attn_subsequent_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask.bool()


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)
    
    def forward(self, Q, K, V, attn_mask):
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)
        attn = nn.Softmax(dim=-1)(scores)
        # attn = self.dropout(attn)
        # (bs, n_head, n_q_seq, d_head)
        context = torch.matmul(attn, V)
        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_hidn, self.config.d_head * self.config.n_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.d_head * self.config.n_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.d_head * self.config.n_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        # self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        # (bs, n_head, n_q_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        # (bs, n_head, n_k_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)

        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head)
        # (bs, n_head, n_q_seq, e_embd)
        output = self.linear(context)
        # output = self.dropout(output)
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.activ = gelu
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, d_ff, n_seq)
        output = self.activ(self.conv1(inputs.transpose(1, 2)))
        # (bs, n_seq, d_embd)
        output = self.conv2(output).transpose(1, 2)
        # output = self.dropout(output)
        # (bs, n_seq, d_embd)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.multi_head_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
    
    def forward(self, inputs, attn_mask):
        # (bs, n_seq, d_hidn), (bs, n_head, n_seq, n_seq)
        att_outputs, attn = self.multi_head_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (bs, n_seq, d_hidn), (bs, n_head, n_seq, n_seq)
        return ffn_outputs, attn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.wrd_emb = nn.Embedding(self.config.n_vocab, self.config.d_embd)
        self.pos_emb = nn.Embedding(self.config.n_seq + 1, self.config.d_hidn)
        self.seg_emb = nn.Embedding(4, self.config.d_hidn)
        self.e_fecto = nn.Linear(self.config.d_embd, self.config.d_hidn)

        self.layer = EncoderLayer(self.config)

    def forward(self, inputs, segs):
        positions = torch.arange(inputs.size(1), device=inputs.device, dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)) + 1
        pos_mask = inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)

        # (bs, n_seq, d_embd)
        wrd_emb = self.wrd_emb(inputs)
        # (bs, n_seq, d_hidn)
        wrd_emb = self.e_fecto(wrd_emb)
        # (bs, n_seq, d_hidn)
        outputs = wrd_emb + self.pos_emb(positions) + self.seg_emb(segs)

        # (bs, n_seq, n_seq)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        attn_probs = []
        for _ in range(self.config.n_layer):
            # (bs, n_seq, d_hidn), (bs, n_head, n_seq, n_seq)
            outputs, attn_prob = self.layer(outputs, attn_mask)
            attn_probs.append(attn_prob)

        # (bs, n_seq, d_hidn), [(bs, n_head, n_seq, n_seq)]
        return outputs, attn_probs


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
    
    def forward(self, input_ids, token_type_ids):
        outputs, attn_probs = self.encoder(input_ids, token_type_ids)
        pooled_output = outputs[:,0]

        return outputs, pooled_output, attn_probs
    
    def save(self, epoch, path):
        torch.save({
            "epoch": epoch,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"]


class BertPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert = BertModel(config)
        self.mlm_prediction = nn.Linear(config.d_hidn, config.n_output)
        self.mlm_prediction.weight = self.bert.encoder.wrd_emb.weight
        self.nsp_prediction = nn.Linear(config.d_hidn, config.n_vocab)
    
    def forward(self, input_ids, token_type_ids):
        outputs, pooled_output, attn_probs = self.bert(input_ids, token_type_ids)

        prediction_scores, seq_relationship_score = self.cls(outputs, pooled_output)
 
        return prediction_scores, seq_relationship_score, attn_probs


class SNLI(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert = BertModel(config)
        self.snli_prediction = nn.Linear(config.d_hidn, config.n_output)
    
    def forward(self, input_ids, token_type_ids):
        outputs, pooled_output, attn_probs = self.bert(input_ids, token_type_ids)

        snli_prediction = self.snli_prediction(pooled_output)
 
        return snli_prediction
    
    def save(self, epoch, score_loss, score_val, score_test, path):
        torch.save({
            "epoch": epoch,
            "score_loss": score_loss,
            "score_val": score_val,
            "score_test": score_test,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])



