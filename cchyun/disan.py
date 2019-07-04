import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def masked_softmax(vec, mask, dim=1):
	masked_vec = vec * mask.float()
	max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
	exps = torch.exp(masked_vec - max_vec)
	masked_exps = exps * mask.float()
	masked_sums = masked_exps.sum(dim, keepdim=True)
	zeros = (masked_sums == 0)
	masked_sums += zeros.float()
	return masked_exps / (masked_sums + 1e-20)


def get_direct_mask_tile(direction, seq_len, device):
    mask = torch.FloatTensor(seq_len, seq_len).to(device)
    mask.data.fill_(1)
    if direction == 'fw':
        mask = torch.triu(mask, diagonal=1)
    elif direction == 'bw':
        mask = torch.tril(mask, diagonal=-1)
    else:
        raise NotImplementedError('only forward or backward mask is allowed!')
    mask.unsqueeze_(0)
    return mask


def get_rep_mask_tile(rep_mask, device):
	batch_size, seq_len = rep_mask.size()
	mask = rep_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)

	return mask


# """
# [n_batch, n_seq]
# padding: 0, not_padding: 1
# """
def get_rep_mask(lengths, sentence_len, device):
	rep_mask = torch.FloatTensor(len(lengths), sentence_len).to(device)
	rep_mask.data.fill_(1)
	for i in range(len(lengths)):
		rep_mask[i, lengths[i]:] = 0

	return rep_mask


class Source2Token(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.d_hidden * 2, config.d_hidden * 2)
        self.fc2 = nn.Linear(config.d_hidden * 2, config.d_hidden * 2)
        init.xavier_uniform_(self.fc1.weight.data)
        init.constant_(self.fc1.bias.data, 0)
        init.xavier_uniform_(self.fc2.weight.data)
        init.constant_(self.fc2.bias.data, 0)

        self.elu = nn.ELU()
        self.softmax = nn.Softmax(dim=-2)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, inputs, rep_mask):
        inputs = self.dropout(inputs)
        map1 = self.elu(self.fc1(inputs))
        map2 = self.fc2(self.dropout(map1))

        soft = masked_softmax(map2, rep_mask.unsqueeze(-1), dim=1)
        out = torch.sum(inputs * soft, dim=1)
        
        return out


# DiSA
class DiSA(nn.Module):
    def __init__(self, config, direction):
        super().__init__()
        self.config = config
        self.direction = direction

        self.fc = nn.Linear(config.d_embed, config.d_hidden)
        init.xavier_uniform_(self.fc.weight.data)
        init.constant_(self.fc.bias.data, 0)

        self.w_1 = nn.Linear(config.d_hidden, config.d_hidden)
        self.w_2 = nn.Linear(config.d_hidden, config.d_hidden)
        init.xavier_uniform_(self.w_1.weight)
        init.xavier_uniform_(self.w_2.weight)
        init.constant_(self.w_1.bias, 0)
        init.constant_(self.w_2.bias, 0)
        self.w_1.bias.requires_grad = False
        self.w_2.bias.requires_grad = False

        self.b_1 = nn.Parameter(torch.zeros(config.d_hidden))
        self.c = nn.Parameter(torch.Tensor([5.0]), requires_grad=False)

        self.w_f1 = nn.Linear(config.d_hidden, config.d_hidden)
        self.w_f2 = nn.Linear(config.d_hidden, config.d_hidden)
        init.xavier_uniform_(self.w_f1.weight)
        init.xavier_uniform_(self.w_f2.weight)
        init.constant_(self.w_f1.bias, 0)
        init.constant_(self.w_f2.bias, 0)
        self.w_f1.bias.requires_grad = False
        self.w_f2.bias.requires_grad = False
        self.b_f = nn.Parameter(torch.zeros(config.d_hidden))

        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs, rep_mask):
        n_batch, n_seq, d_embd = inputs.size()
        # (n_batch, n_seq, n_seq)
        rep_mask_tile = get_rep_mask_tile(rep_mask, self.config.device)
        # (1, n_seq, n_seq)
        direct_mask_tile = get_direct_mask_tile(self.direction, n_seq, self.config.device)
        # (n_batch, n_seq, n_seq)
        mask = rep_mask_tile * direct_mask_tile
        # (n_batch, n_seq, n_seq, 1)
        mask.unsqueeze_(-1)

        inputs = self.dropout(inputs)
        # Fully Connected Layer: (n_batch, n_seq, d_hidden)
        rep_map = self.elu(self.fc(inputs))
        # (n_batch, n_seq, n_seq, d_hidden)
        rep_map_tile = rep_map.unsqueeze(1).expand(n_batch, n_seq, n_seq, self.config.d_hidden)
        rep_map = self.dropout(rep_map)

        # (n_batch, 1, n_seq, d_hidden)
        dependent_etd = self.w_1(rep_map).unsqueeze(1)
        # (n_batch, n_seq, 1, d_hidden)
        head_etd = self.w_2(rep_map).unsqueeze(2)

        # (n_batch, n_seq, n_seq, d_hidden)
        logits = self.c * self.tanh((dependent_etd + head_etd + self.b_1) / self.c)

        # Attention scores (n_batch, n_seq, n_seq, d_hidden)
        attn_score = masked_softmax(logits, mask, dim=2)
        attn_score = attn_score * mask

        # Attention results (n_batch, n_seq, d_hidden)
        attn_result = torch.sum(attn_score * rep_map_tile, dim=2)

        # Fusion gate: combination with input (n_batch, n_seq, d_hidden)
        fusion_gate = self.sigmoid(self.w_f1(self.dropout(rep_map)) + self.w_f2(self.dropout(attn_result)) + self.b_f)
        out = fusion_gate * rep_map + (1-fusion_gate) * attn_result

        out = out * rep_mask.unsqueeze(-1)
        return out


# DiSAN
class DiSAN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fw_DiSA = DiSA(config, direction='fw')
        self.bw_DiSA = DiSA(config, direction='bw')

        self.source2token = Source2Token(config)
    
    def forward(self, inputs, rep_mask):
        # Forward and backward DiSA (n_batch, n_seq, d_hidden)
        fw_u = self.fw_DiSA(inputs, rep_mask)
        bw_u = self.bw_DiSA(inputs, rep_mask)

        # Concat (n_batch, n_seq, d_hidden * 2)
        u = torch.cat([fw_u, bw_u], dim=-1)

        # Source2Token
        s = self.source2token(u, rep_mask)
        
        return s


# SNLI
class SNLI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.n_vocab, config.d_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.elu = nn.ELU()

        self.disan = DiSAN(config)

        self.fc = nn.Linear(config.d_hidden * 8, config.d_hidden)
        self.fc_out = nn.Linear(config.d_hidden, config.n_output)
    
    def forward(self, sentence1, sentence2):
        sentence1_pre_len = sentence1.ne(self.config.i_pad).sum(dim=1)
        sentence2_pre_len = sentence2.ne(self.config.i_pad).sum(dim=1)
        
        # Get representation masks for sentences of variable lengths
        _, sentence1_seq_len = sentence1.size()
        sentence1_mask = get_rep_mask(sentence1_pre_len, sentence1_seq_len, self.config.device)
        _, sentence2_seq_len = sentence2.size()
        sentence2_mask = get_rep_mask(sentence2_pre_len, sentence2_seq_len, self.config.device)

        # Embedding (n_batch, n_seq, d_embed)
        sentence1_embed = self.embed(sentence1)
        sentence2_embed = self.embed(sentence2)

        # DiSAN
        sentence1_s = self.disan(sentence1_embed, sentence1_mask)
        sentence2_s = self.disan(sentence2_embed, sentence2_mask)

        outs = torch.cat([sentence1_s, sentence2_s, torch.abs(sentence1_s - sentence2_s), sentence1_s * sentence2_s], dim=-1)
        outs = self.elu(self.fc(self.dropout(outs)))
        outs = self.fc_out(self.dropout(outs))

        return outs

