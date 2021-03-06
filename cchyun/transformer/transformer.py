import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# 참고: https://github.com/graykode/nlp-tutorial
#      https://github.com/jadore801120/attention-is-all-you-need-pytorch
#      https://github.com/JayParks/transformer
#      https://github.com/modudeepnlp/code_implementation/blob/master/codes/transformer/Transformer-Torch.py


def get_sinusoid_encoding_table(n_seq, d_embed):
    def cal_angle(position, i_embed):
        return position / np.power(10000, 2 * (i_embed // 2) / d_embed)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_embed) for i_embed in range(d_embed)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return sinusoid_table


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad).unsqueeze(1).expand(batch_size, len_q, len_k)  # <pad>
    return pad_attn_mask.byte()


def get_attn_subsequent_mask(seq):
    subsequent_mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    subsequent_mask = subsequent_mask.triu(diagonal=1) # upper triangular part of a matrix(2-D)
    return subsequent_mask.byte()


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
        self.layer_norm = nn.LayerNorm(self.config.d_embed)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)

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
        return self.layer_norm(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_embed, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_embed, kernel_size=1)
        self.layer_norm = nn.LayerNorm(self.config.d_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, n_seq, d_embd)
        residual = inputs

        # (bs, d_ff, n_seq)
        output = F.relu(self.conv1(inputs.transpose(1, 2)))
        # (bs, n_seq, d_embed)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        # (bs, n_seq, d_embed)
        return self.layer_norm(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_self_attn = MultiHeadAttention(self.config)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
    
    def forward(self, enc_inputs, enc_self_attn_mask):
        # (bs, n_enc_seq, d_embed), (bs, n_head, n_enc_seq, n_enc_seq)
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        # (bs, n_enc_seq, d_embed), (bs, n_head, n_enc_seq, n_enc_seq)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_embed)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_enc_seq + 1, self.config.d_embed))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, enc_inputs):
        possitions = torch.cumsum(torch.ones(enc_inputs.size(1), dtype=torch.long).to(self.config.device), dim=0) * (1 - enc_inputs.eq(self.config.i_pad)).to(torch.long)
        # (bs, n_enc_seq, d_embed)
        enc_outputs = self.enc_emb(enc_inputs) + self.pos_emb(possitions)

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
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
    
    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_dec_seq)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_enc_seq)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_dec_seq), (bs, n_head, n_dec_seq, n_enc_seq)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_embed)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.config.d_embed))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        possitions = torch.cumsum(torch.ones(dec_inputs.size(1), dtype=torch.long).to(self.config.device), dim=0) * (1 - dec_inputs.eq(self.config.i_pad)).to(torch.long)
        # (bs, n_dec_seq, d_embed)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(possitions)

        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # (bs, n_dec_seq, n_dec_seq)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_subsequent_mask), 0)
        # (bs, n_dec_seq, n_enc_seq)
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs, self.config.i_pad)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # (bs, n_dec_seq, d_embed), (bs, n_dec_seq, n_dec_seq), (bs, n_dec_seq, n_enc_seq)
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # (bs, n_dec_seq, d_embed), [(bs, n_dec_seq, n_dec_seq)], [(bs, n_dec_seq, n_enc_seq)]S
        return dec_outputs, dec_self_attns, dec_enc_attns


class DecoderSNLILayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_self_attn = MultiHeadAttention(self.config)
        self.dec_enc_attn = MultiHeadAttention(self.config)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
    
    def forward(self, dec_inputs, dec_self_attn_mask):
        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_dec_seq)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_dec_seq)
        return dec_outputs, dec_self_attn


class DecoderSNLI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_embed)
        sinusoid_table = torch.FloatTensor(get_sinusoid_encoding_table(self.config.n_dec_seq + 1, self.config.d_embed))
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_table, freeze=True)

        self.layers = nn.ModuleList([DecoderSNLILayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, dec_inputs):
        possitions = torch.cumsum(torch.ones(dec_inputs.size(1), dtype=torch.long).to(self.config.device), dim=0) * (1 - dec_inputs.eq(self.config.i_pad)).to(torch.long)
        # (bs, n_dec_seq, d_embed)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(possitions)

        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)
        # (bs, n_dec_seq, n_dec_seq)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_subsequent_mask), 0)

        dec_self_attns = []
        for layer in self.layers:
            # (bs, n_dec_seq, d_embed), (bs, n_dec_seq, n_dec_seq)
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        # (bs, n_dec_seq, d_embed), [(bs, n_dec_seq, n_dec_seq)]
        return dec_outputs, dec_self_attns


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.projection = nn.Linear(self.config.d_embed, self.config.n_dec_vocab, bias=False)
    
    def forward(self, enc_inputs, dec_inputs):
        # (bs, n_enc_seq, d_embed), [(bs, n_head, n_enc_seq, n_enc_seq)]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # (bs, n_seq, d_embed), [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        # (bs, n_dec_seq, n_dec_vocab)
        dec_logits = self.projection(dec_outputs)
        # (bs, n_dec_seq, n_dec_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)], [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns


"""
    promise, hypersis:        loss: 0.599, dev: 75.086, test: 75.428
                              loss: 0.657, dev: 65.474, test: 64.780
    <s>p<d>h<e>, <s>h<d>p<e>: loss: 0.643, dev: 73.359, test: 73.137
"""
class SNLITransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)
        self.decoder.dec_emb = self.encoder.enc_emb
        n_hidden = config.n_dec_seq * config.d_embed
        self.projection = nn.Linear(n_hidden, self.config.n_output, bias=False)
    
    def forward(self, sentence1, sentence2):
         # (bs, n_enc_seq, d_embed), [(bs, n_head, n_enc_seq, n_enc_seq)]
        enc_outputs, enc_self_attns = self.encoder(sentence1)
        # (bs, n_seq, d_embed), [(bs, n_head, n_dec_seq, n_dec_seq)], [(bs, n_head, n_dec_seq, n_enc_seq)]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(sentence2, sentence1, enc_outputs)
        # (bs, n_dec_seq, n_dec_vocab)
        output = self.projection(dec_outputs.view(dec_outputs.size()[0], -1))
        return output


"""
    promise, hypersis, [h;p,|h-p|,h*p]: loss: 0.549, dev: 76.245, test: 76.079
    <s>p<d>h<e>, <s>h<d>p<e>          : loss: 0.567, dev: 75.147, test: 74.898
    <s>p<d>h<e>                       : loss: 0.218, dev: 79.628, test: 79.591
"""
class SNLIEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.dropout1 = nn.Dropout(p=config.dropout)
        n_hidden = config.n_enc_seq * config.d_embed
        self.layout1 = nn.Linear(n_hidden, n_hidden // 4)
        self.dropout2 = nn.Dropout(p=config.dropout)
        self.layout2 = nn.Linear(n_hidden // 4, n_hidden // 16)
        self.layout3 = nn.Linear(n_hidden // 16, config.n_output)
    
    def forward(self, sentence1, sentence2):
        # (bs, n_enc_seq, d_embed) -> (bs, n_enc_seq * d_embed)
        sentence1_ctx, _ = self.encoder(sentence1)
        sentence1_ctx = sentence1_ctx.view(sentence1_ctx.size()[0], -1)
        # sentence2_ctx, _ = self.encoder(sentence2)
        # sentence2_ctx = sentence2_ctx.view(sentence2_ctx.size()[0], -1)

        # # (bs, n_enc_seq * d_embed * 4)
        # output = torch.cat([sentence1_ctx, sentence2_ctx, torch.abs(sentence1_ctx - sentence2_ctx), sentence1_ctx * sentence2_ctx], 1)
        output = sentence1_ctx

        output = self.dropout1(output)
        output = F.relu(self.layout1(output))
        output = self.dropout2(output)
        output = F.relu(self.layout2(output))
        output = self.layout3(output)
        return output


"""
    promise, hypersis, [h;p,|h-p|,h*p]: loss: 0.664, dev: 72.424, test: 72.445
    <s>p<d>h<e>, <s>h<d>p<e>          : loss: 0.671, dev: 72.069, test: 72.64
    <s>p<d>h<e>                       : loss: 0.403, dev: 78.998, test: 78.491
"""
class SNLIDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.decoder = DecoderSNLI(self.config)
        self.dropout1 = nn.Dropout(p=config.dropout)
        n_hidden = config.n_dec_seq * config.d_embed
        self.layout1 = nn.Linear(n_hidden, n_hidden // 4)
        self.dropout2 = nn.Dropout(p=config.dropout)
        self.layout2 = nn.Linear(n_hidden // 4, n_hidden // 16)
        self.layout3 = nn.Linear(n_hidden // 16, config.n_output)
    
    def forward(self, sentence1, sentence2):
        # (bs, n_dec_seq, d_embed) -> (bs, n_dec_seq * d_embed)
        sentence1_ctx, _ = self.decoder(sentence1)
        sentence1_ctx = sentence1_ctx.view(sentence1_ctx.size()[0], -1)
        # sentence2_ctx, _ = self.decoder(sentence2)
        # sentence2_ctx = sentence2_ctx.view(sentence2_ctx.size()[0], -1)

        # # (bs, n_dec_seq * d_embed * 4)
        # output = torch.cat([sentence1_ctx, sentence2_ctx, torch.abs(sentence1_ctx - sentence2_ctx), sentence1_ctx * sentence2_ctx], 1)
        output = sentence1_ctx

        output = self.dropout1(output)
        output = F.relu(self.layout1(output))
        output = self.dropout2(output)
        output = F.relu(self.layout2(output))
        output = self.layout3(output)
        return output


class SNLI(SNLIEncoder):
    def __init__(self, config):
        super().__init__(config)

