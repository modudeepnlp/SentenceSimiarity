import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformer


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_embed)
        self.pos_emb = nn.Embedding(self.config.n_enc_seq + 1, self.config.d_embed)

        self.layers = nn.ModuleList([transformer.EncoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, enc_inputs):
        possitions = torch.cumsum(torch.ones(enc_inputs.size(1), dtype=torch.long).to(self.config.device), dim=0) * (1 - enc_inputs.eq(self.config.i_pad)).to(torch.long)
        # (bs, n_enc_seq, d_embed)
        enc_outputs = self.enc_emb(enc_inputs) + self.pos_emb(possitions)

        # (bs, n_enc_seq, n_enc_seq)
        enc_self_attn_mask = transformer.get_attn_pad_mask(enc_inputs, enc_inputs, self.config.i_pad)

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

        self.dec_self_attn = transformer.MultiHeadAttention(self.config)
        self.pos_ffn = transformer.PoswiseFeedForwardNet(self.config)
    
    def forward(self, dec_inputs, dec_self_attn_mask):
        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_dec_seq)
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        # (bs, n_dec_seq, d_embed), (bs, n_head, n_dec_seq, n_dec_seq)
        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.dec_emb = nn.Embedding(self.config.n_dec_vocab, self.config.d_embed)
        self.pos_emb = nn.Embedding(self.config.n_dec_seq + 1, self.config.d_embed)

        self.layers = nn.ModuleList([DecoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, dec_inputs):
        possitions = torch.cumsum(torch.ones(dec_inputs.size(1), dtype=torch.long).to(self.config.device), dim=0) * (1 - dec_inputs.eq(self.config.i_pad)).to(torch.long)
        # (bs, n_dec_seq, d_embed)
        dec_outputs = self.dec_emb(dec_inputs) + self.pos_emb(possitions)

        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_pad_mask = transformer.get_attn_pad_mask(dec_inputs, dec_inputs, self.config.i_pad)
        # (bs, n_dec_seq, n_dec_seq)
        dec_attn_subsequent_mask = transformer.get_attn_subsequent_mask(dec_inputs)
        # (bs, n_dec_seq, n_dec_seq)
        dec_self_attn_mask = torch.gt((dec_attn_pad_mask + dec_attn_subsequent_mask), 0)

        dec_self_attns = []
        for layer in self.layers:
            # (bs, n_dec_seq, d_embed), (bs, n_dec_seq, n_dec_seq)
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_mask)
            dec_self_attns.append(dec_self_attn)
        # (bs, n_dec_seq, d_embed), [(bs, n_dec_seq, n_dec_seq)]
        return dec_outputs, dec_self_attns


class SNLIEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)

        self.projection_lm = nn.Linear(self.config.d_embed, self.config.n_enc_vocab, bias=False)
        # tie_weights
        self.projection_lm.weight = self.encoder.enc_emb.weight

        self.projection_snli = nn.Linear(config.n_enc_seq * config.d_embed, config.n_output)
     
    def forward(self, sentence):
        # (bs, n_enc_seq, d_embed) -> (bs, n_enc_seq * d_embed)
        sentence_ctx, _ = self.encoder(sentence)

        lm_logit = self.projection_lm(sentence_ctx)
        snli_logit = self.projection_snli(sentence_ctx.view(sentence_ctx.size()[0], -1))
        
        return lm_logit[:, :-1, :].contiguous(), snli_logit


class SNLIDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.decoder = Decoder(self.config)

        self.projection_lm = nn.Linear(self.config.d_embed, self.config.n_dec_vocab, bias=False)
        # tie_weights
        self.projection_lm.weight = self.decoder.dec_emb.weight

        self.projection_snli = nn.Linear(config.n_dec_seq * config.d_embed, config.n_output)
     
    def forward(self, sentence):
        # (bs, n_dec_seq, d_embed) -> (bs, n_dec_seq * d_embed)
        sentence_ctx, _ = self.decoder(sentence)
        
        lm_logit = self.projection_lm(sentence_ctx)
        snli_logit = self.projection_snli(sentence_ctx.view(sentence_ctx.size()[0], -1))
        
        return lm_logit[:, :-1, :].contiguous(), snli_logit


class SNLI(SNLIDecoder):
    def __init__(self, config):
        super().__init__(config)

