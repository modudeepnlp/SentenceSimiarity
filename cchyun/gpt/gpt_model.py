import numpy as np
import math

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
    return pad_attn_mask.byte()


def get_attn_subsequent_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask.byte()


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask):
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.config.d_k)
        scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn, V)
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_embed, self.config.d_k * self.config.n_heads)
        self.W_K = nn.Linear(self.config.d_embed, self.config.d_k * self.config.n_heads)
        self.W_V = nn.Linear(self.config.d_embed, self.config.d_k * self.config.n_heads)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_heads * self.config.d_v, self.config.d_embed)

        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)

        # (bs, n_head, n_q_seq, d_k)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1,2)
        # (bs, n_head, n_k_seq, d_k)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_heads, self.config.d_k).transpose(1,2)
        # (bs, n_head, n_v_seq, d_v)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_heads, self.config.d_v).transpose(1,2)

        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_heads, 1, 1)

        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_k_seq)
        context, attn = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (bs, n_head, n_q_seq, h_head * d_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_heads * self.config.d_v)
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
        # (bs, n_enc_seq, d_embed), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        att_outputs = self.layer_norm1(enc_inputs + att_outputs)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (bs, n_enc_seq, d_embed), (bs, n_head, n_enc_seq, n_enc_seq)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_embed)
        self.pos_emb = nn.Embedding(self.config.n_enc_seq + 1, self.config.d_embed)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, enc_inputs):
        positions = torch.arange(enc_inputs.size(1), device=enc_inputs.device, dtype=enc_inputs.dtype).expand(enc_inputs.size(0), enc_inputs.size(1)) + 1
        pos_mask = enc_inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)
        # (bs, n_enc_seq, d_embed)
        enc_outputs = self.enc_emb(enc_inputs) + self.pos_emb(positions)

        # (bs, n_enc_seq, n_enc_seq)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs, self.config.i_pad)

        enc_self_attns = []
        for layer in self.layers:
            # (bs, n_enc_seq, d_embed), (bs, n_head, n_enc_seq, n_enc_seq)
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        # (bs, n_enc_seq, d_embed), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_self_attn = MultiHeadAttention(self.config)
        # GPT2: layer normal (위치변경)
        self.layer_norm1 = nn.LayerNorm(self.config.d_embed, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        # GPT2: layer normal (위치변경)
        self.layer_norm2 = nn.LayerNorm(self.config.d_embed, eps=self.config.layer_norm_epsilon)
    
    def forward(self, dec_inputs, dec_self_attn_mask):
        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_dec_seq)
        att_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        att_outputs = self.layer_norm1(dec_inputs + att_outputs)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_dec_seq)
        return ffn_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_embed)
        self.pos_emb = nn.Embedding(self.config.n_dec_seq + 1, self.config.d_embed)
        self.seg_emb = nn.Embedding(3, self.config.d_embed)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])

        # GPT2: layer normal 추가
        self.layer_norm = nn.LayerNorm(config.d_embed, eps=config.layer_norm_epsilon)
    
    def forward(self, dec_inputs, dec_segs):
        positions = torch.arange(dec_inputs.size(1), device=dec_inputs.device, dtype=dec_inputs.dtype).expand(dec_inputs.size(0), dec_inputs.size(1)) + 1
        pos_mask = dec_inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)
        # (bs, n_dec_seq, d_embed)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(positions) + self.seg_emb(dec_segs)

        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # (bs, n_dec_seq, n_dec_seq)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_subsequent_mask), 0)

        dec_self_attns = []
        # GPT2: layer normal 추가
        # dec_outputs = self.layer_norm(dec_outputs)
        for layer in self.layers:
            # (bs, n_dec_seq, d_embed), (bs, n_dec_seq, n_dec_seq)
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        # (bs, n_dec_seq, d_embed), [(bs, n_dec_seq, n_dec_seq)]
        return dec_outputs, dec_self_attns
    
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])


class GPTPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.decoder = Decoder(self.config)

        self.projection_lm = nn.Linear(self.config.d_embed, self.config.n_dec_vocab, bias=False)
        # tie_weights
        self.projection_lm.weight = self.decoder.dec_emb.weight

        self.projection_ns = nn.Linear(config.d_embed, 2)

        self.dropout = nn.Dropout(config.dropout)
     
    def forward(self, sentences, segments):
        # (bs, n_dec_seq, d_embed) -> (bs, n_dec_seq * d_embed)
        sentence_ctx, _ = self.decoder(sentences, segments)
        
        lm_logit = self.projection_lm(sentence_ctx)

        ns_logit = sentence_ctx[:, -1]
        ns_logit = self.dropout(ns_logit)
        ns_logit = self.projection_ns(ns_logit)
        ns_logit = torch.tanh(ns_logit)
        ns_logit = self.dropout(ns_logit)
        
        return lm_logit[:, :-1, :].contiguous(), ns_logit


class SNLI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.decoder = Decoder(self.config)

        self.projection_lm = nn.Linear(self.config.d_embed, self.config.n_dec_vocab, bias=False)
        # tie_weights
        self.projection_lm.weight = self.decoder.dec_emb.weight

        self.projection_snli = nn.Linear(config.d_embed, config.n_output)

        self.dropout = nn.Dropout(config.dropout)
     
    def forward(self, sentences, segments):
        # (bs, n_dec_seq, d_embed) -> (bs, n_dec_seq * d_embed)
        sentence_ctx, _ = self.decoder(sentences, segments)
        
        lm_logit = self.projection_lm(sentence_ctx)

        snli_logit = sentence_ctx[:, -1]
        snli_logit = self.dropout(snli_logit)
        snli_logit = self.projection_snli(snli_logit)
        snli_logit = torch.tanh(snli_logit)
        snli_logit = self.dropout(snli_logit)
        
        return lm_logit[:, :-1, :].contiguous(), snli_logit
    
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
