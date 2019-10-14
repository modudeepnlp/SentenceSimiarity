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
        self.dropout = nn.Dropout(config.dropout)

        self.scale = 1 / (self.config.d_head ** 0.5)
    
    def forward(self, Q, K, V, attn_mask):
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2)).mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)
        # (bs, n_head, n_q_seq, d_head)
        context = torch.matmul(attn, V)
        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_embed, self.config.d_head * self.config.n_head)
        self.W_K = nn.Linear(self.config.d_embed, self.config.d_head * self.config.n_head)
        self.W_V = nn.Linear(self.config.d_embed, self.config.d_head * self.config.n_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_embed)

        self.dropout = nn.Dropout(config.dropout)
    
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
        output = self.dropout(output)
        # (bs, n_q_seq, d_embed), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_embed, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_embed, kernel_size=1)
        self.activ = gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, d_ff, n_seq)
        output = self.activ(self.conv1(inputs.transpose(1, 2)))
        # gelu ??
        # (bs, n_seq, d_embed)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        # (bs, n_seq, d_embed)
        return output


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_self_attn = MultiHeadAttention(self.config)
        # GPT2: layer normal (위치변경)
        self.layer_norm1 = nn.LayerNorm(self.config.d_embed, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        # GPT2: layer normal (위치변경)
        self.layer_norm2 = nn.LayerNorm(self.config.d_embed, eps=self.config.layer_norm_epsilon)
    
    def forward(self, enc_inputs, enc_self_attn_mask):
        # (bs, n_seq, d_embed), (bs, n_head, n_seq, n_seq)
        att_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        att_outputs = self.layer_norm1(enc_inputs + att_outputs)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (bs, n_seq, d_embed), (bs, n_head, n_seq, n_seq)
        return ffn_outputs, attn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(self.config.n_vocab, self.config.d_embed)
        self.pos_emb = nn.Embedding(self.config.n_seq + 1, self.config.d_embed)
        self.seg_emb = nn.Embedding(2, self.config.d_embed)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, enc_inputs, enc_segs):
        positions = torch.arange(enc_inputs.size(1), device=enc_inputs.device, dtype=enc_inputs.dtype).expand(enc_inputs.size(0), enc_inputs.size(1)) + 1
        pos_mask = enc_inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)
        # (bs, n_seq, d_embed)
        enc_outputs = self.enc_emb(enc_inputs) + self.pos_emb(positions) + self.seg_emb(enc_segs)

        # (bs, n_seq, n_seq)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs, self.config.i_pad)

        enc_self_attns = []
        for layer in self.layers:
            # (bs, n_seq, d_embed), (bs, n_head, n_seq, n_seq)
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        # (bs, n_seq, d_embed), [(bs, n_head, n_seq, n_seq)]
        return enc_outputs, enc_self_attns


class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(config)
    
    def forward(self, input_ids, token_type_ids):
        enc_outputs, enc_self_attns = self.encoder(input_ids, token_type_ids)
        pooled_output = enc_outputs[:,0]

        return enc_outputs, pooled_output, enc_self_attns
    
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
        self.mlm_prediction = nn.Linear(config.d_embd, config.n_output)
        self.mlm_prediction.weight = self.bert.encoder.wrd_emb.weight
        self.nsp_prediction = nn.Linear(config.d_embd, config.n_vocab)
    
    def forward(self, input_ids, token_type_ids):
        outputs, pooled_output, attn_probs = self.bert(input_ids, token_type_ids)

        prediction_scores, seq_relationship_score = self.cls(outputs, pooled_output)
 
        return prediction_scores, seq_relationship_score, attn_probs


class SNLI(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.bert = BertModel(config)
        self.seq_relationship = nn.Linear(config.d_embed, config.n_output)
    
    def forward(self, input_ids, token_type_ids):
        enc_outputs, pooled_output, enc_self_attns = self.bert(input_ids, token_type_ids)

        seq_relationship_score = self.seq_relationship(pooled_output)
 
        return seq_relationship_score
    
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



