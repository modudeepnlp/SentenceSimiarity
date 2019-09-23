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
    
    def forward(self, Q, K, V, r_w_bias, r_r_bias, attn_mask):
        rw_Q = (Q.transpose(1, 2) + r_w_bias).transpose(1, 2)
        # (Q + u) * K: (bs, n_head, qlen, klen)
        AC = torch.matmul(rw_Q, K.transpose(-1, -2))

        rr_Q = (Q.transpose(1, 2) + r_r_bias).transpose(1, 2)
        # (Q + v) * K: (bs, n_head, qlen, klen)
        BD = torch.matmul(rr_Q, K.transpose(-1, -2))

        print(f">>>>>>>> {Q.size()}, {rr_Q.size()}, {BD.size()}")
        # # (bs, n_head, qlen, klen)
        # attn_score = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.config.d_head)
        # attn_score.masked_fill_(attn_mask, -1e9)
        # # (bs, n_head, qlen, klen)
        # attn_prob = nn.Softmax(dim=-1)(attn_score)
        # attn_prob = self.dropout(attn_prob)
        # # (bs, n_head, qlen, d_head)
        # context = torch.matmul(attn_prob, V)
        # # (bs, n_head, qlen, d_head), (bs, n_head, qlen, klen)
        # return context, attn_prob


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_embed, self.config.d_head * self.config.n_head)
        self.W_K = nn.Linear(self.config.d_embed, self.config.d_head * self.config.n_head)
        self.W_V = nn.Linear(self.config.d_embed, self.config.d_head * self.config.n_head)

        self.r_net = nn.Linear(self.config.d_embed, self.config.n_head * self.config.d_head, bias=False)
        # self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_embed)

        self.dropout = nn.Dropout(config.dropout)

        self.scale = 1 / (self.config.d_head ** 0.5)
    
    def forward(self, Q, K, V, pos_emb, r_w_bias, r_r_bias, attn_mask, mems):
        n_batch = Q.size(0)

        # E * Wq: (bs, qlen, n_head, d_head)
        q_s = self.W_Q(Q).view(n_batch, -1, self.config.n_head, self.config.d_head)
        # E * Wq + u: (bs, n_head, qlen, d_head)
        Q_rw = (q_s + r_w_bias).transpose(1, 2)
        # E * Wq + v: (bs, n_head, qlen, d_head)
        Q_rr = (q_s + r_r_bias).transpose(1, 2)
        # E * Wk: (bs, n_head, klen, d_head)
        k_s = self.W_K(K).view(n_batch, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        # (bs, n_head, klen, d_head)
        v_s = self.W_V(V).view(n_batch, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        # Wr * R: (n_head, klen, d_head)
        r_s = self.r_net(pos_emb).view(-1, self.config.n_head, self.config.d_head).transpose(0, 1)

        # (bs, n_head, qlen, klen)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        # (E * Wq + u) * (E * Wk)T: (bs, n_head, qlen, klen)
        AC = torch.matmul(Q_rw, k_s.transpose(-1, -2))
        # (E * Wq + v) * (Wr * R)T: (bs, n_head, qlen, klen)
        BD = torch.matmul(Q_rr, r_s.transpose(-1, -2))
        BD = self._rel_shift(BD)

        # (bs, n_head, qlen, klen)
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        attn_score.masked_fill_(attn_mask, -1e9)

        # (bs, n_head, qlen, klen)
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)

        # (bs, n_head, qlen, d_head)
        attn_vec = torch.matmul(attn_prob, v_s)
        # (bs, qlen, h_head * d_head)
        attn_vec = attn_vec.contiguous().view(n_batch, -1, self.config.n_head * self.config.d_head)
        # (bs, qlen, d_embed)
        attn_vec = self.linear(attn_vec)
        attn_vec = self.dropout(attn_vec)

        # (bs, qlen, d_embed), (bs, n_head, qlen, klen)
        return attn_vec, attn_prob

    def _rel_shift(self, x):
        # (bs, 1, qlen, klen)
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        # (bs, n_head + 1, qlen, klen)
        x_padded = torch.cat([zero_pad, x], dim=1)
        # (n_head + 1, bs, qlen, klen)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        x_shift = x_padded[1:].view_as(x)
        # print(f">>>>>>>> {x[0][0]}")
        # print(f">>>>>>>> {x_shift[0][0]}")
        return x_shift


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_embed, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_embed, kernel_size=1)
        self.activ = gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, d_ff, n_seq), glue
        output = self.activ(self.conv1(inputs.transpose(1, 2)))
        # (bs, n_seq, d_embed)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        # (bs, n_seq, d_embed)
        return output


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_embed, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_embed, eps=self.config.layer_norm_epsilon)
    
    def forward(self, dec_inputs, pos_emb, r_w_bias, r_r_bias, dec_attn_mask, mems):
        q_inputs = dec_inputs
        kv_inputs = torch.cat([mems, dec_inputs], dim=1)

        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_dec_seq)
        att_outputs, dec_attn_prob = self.dec_self_attn(q_inputs, kv_inputs, kv_inputs, pos_emb, r_w_bias, r_r_bias, dec_attn_mask, mems)
        att_outputs = self.layer_norm1(dec_inputs + att_outputs)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_dec_seq)
        return ffn_outputs, dec_attn_prob


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_embed)

        # (e_embed / 2)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, self.config.d_embed, 2.0) / self.config.d_embed))
        self.register_buffer('inv_freq', inv_freq)

        self.r_w_bias = nn.Parameter(torch.Tensor(self.config.n_head, self.config.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.config.n_head, self.config.d_head))

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])

        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_embed, eps=config.layer_norm_epsilon)
    
    def init_mems(self):
        mems = []
        param = next(self.parameters())
        for i in range(self.config.n_layer + 1):
            empty = torch.empty(0, dtype=param.dtype, device=param.device)
            mems.append(empty)
        return mems
    
    def _update_mems(self, hids, mems, qlen, mlen):
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        with torch.no_grad():
            new_mems = []
            for i in range(len(hids)):
                new_mems.append(hids[i].detach())
    
        return new_mems

    def forward(self, dec_inputs, mems):
        if not mems:
            mems = self.init_mems()
    
        # (bs, n_dec_seq, d_embed)
        n_batch, qlen = dec_inputs.size()
        word_emb = self.word_emb(dec_inputs)

        mlen = mems[0].size(1) if 1 < mems[0].dim() else 0

        klen = mlen + qlen

        # (qlen, klen, 1)
        dec_attn_mask = torch.triu(word_emb.new_ones(n_batch, qlen, klen), diagonal=1+mlen).bool()

        # [klen - 1, ... , 0], (klen)
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        # (klen, d_embed / 2)
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        # (klen, d_embed)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        dec_outputs = self.dropout(word_emb)
        pos_emb = self.dropout(pos_emb)

        hids = []
        hids.append(dec_outputs)
        for i, layer in enumerate(self.layers):
            # (bs, n_dec_seq, d_embed), (bs, n_dec_seq, n_dec_seq)
            dec_outputs, dec_attn_prob = layer(dec_outputs, pos_emb, self.r_w_bias, self.r_r_bias, dec_attn_mask, mems[i])
            hids.append(dec_outputs)
        dec_outputs = self.dropout(dec_outputs)

        # (bs, n_dec_seq, d_embed)
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        
        # (bs, n_dec_seq, d_embed), n_layer+1 hiddens
        return dec_outputs, new_mems
    
    def save(self, epoch, path):
        torch.save({
            "epoch": epoch,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"]


class TransformerXL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.decoder = Decoder(config)

        self.projection_lm = nn.Linear(self.config.d_embed, self.config.n_dec_vocab, bias=False)
        # tie_weights
        self.projection_lm.weight = self.decoder.word_emb.weight
    
    def forward(self, dec_inputs, *mems):
        dec_outputs, new_mems = self.decoder(dec_inputs, mems)
        logit = self.projection_lm(dec_outputs)
        return logit, new_mems


class SNLI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.decoder = Decoder(self.config)

        self.projection_lm = nn.Linear(self.config.d_embed, self.config.n_dec_vocab, bias=False)
        # tie_weights
        self.projection_lm.weight = self.decoder.word_emb.weight

        self.projection_snli = nn.Linear(config.d_embed, config.n_output)

        self.dropout = nn.Dropout(config.dropout)
     
    def forward(self, sentences, *mems):
        # (bs, n_dec_seq, d_embed) -> (bs, n_dec_seq * d_embed)
        sentence_ctx, new_mems = self.decoder(sentences, mems)
        
        lm_logit = self.projection_lm(sentence_ctx)

        snli_logit = sentence_ctx[:, -1]
        snli_logit = self.dropout(snli_logit)
        snli_logit = self.projection_snli(snli_logit)
        snli_logit = torch.tanh(snli_logit)
        snli_logit = self.dropout(snli_logit)
        
        return lm_logit[:, :-1, :].contiguous(), snli_logit
    
    def save(self, epoch, score_val, score_test, path):
        torch.save({
            "epoch": epoch,
            "score_val": score_val,
            "score_test": score_test,
            "state_dict": self.state_dict()
        }, path)
    
    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])
        return save["epoch"], save["score_val"], save["score_test"]
