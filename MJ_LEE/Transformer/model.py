import torch
import torch.nn as nn
import config as config
import numpy as np
from torch.autograd import Variable
import define_func as func

class embedding(nn.Module):
    def __init__(self, vocab_list, emb_weight):
        super(embedding, self).__init__()
        vocab_length = len(vocab_list)
        self.word_emb = nn.Embedding(vocab_length, config.embedding_dim)
        if config.use_glove:
            self.word_emb = nn.Embedding.from_pretrained(emb_weight)
        pos_weight = func.get_sinusoid_encoding_table(vocab_length)
        self.pos_emb = nn.Embedding.from_pretrained(pos_weight, freeze=True)
        
    def forward(self, sentence):
        sentence = sentence.permute(1, 0) # [21, 256] -> [256, 21]
        max_pos = sentence.size(1)
        a = torch.cumsum(torch.ones(max_pos, dtype=torch.long).to(config.gpu), dim=0)
        # b = (1 - sentence.eq(1)).to(torch.long)
        # position =  a * b 
        position = a # no mask
        return self.word_emb(sentence) + self.pos_emb(position)

        #         position shape :  torch.Size([256, 21])
        # a shape :  torch.Size([21])
        # b shape :  torch.Size([256, 21])
        # position :  tensor([[1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         ...,
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0],
        #         [1, 2, 3,  ..., 0, 0, 0]], device='cuda:0')
        # a :  tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
        #         19, 20, 21], device='cuda:0')
        # b :  tensor([[1, 1, 1,  ..., 0, 0, 0],
        #         [1, 1, 1,  ..., 0, 0, 0],
        #         [1, 1, 1,  ..., 0, 0, 0],
        #         ...,
        #         [1, 1, 1,  ..., 0, 0, 0],
        #         [1, 1, 1,  ..., 0, 0, 0],
        #         [1, 1, 1,  ..., 0, 0, 0]], device='cuda:0')


        # self.pos_emb(position) : 
        # tensor([[ 8.4147e-01,  5.4030e-01,  8.0782e-01,  ...,  1.0000e+00,
                #   1.0633e-04,  1.0000e+00],
                # [ 9.0930e-01, -4.1615e-01,  9.5231e-01,  ...,  1.0000e+00,
                #   2.1267e-04,  1.0000e+00],
                # [ 1.4112e-01, -9.8999e-01,  3.1481e-01,  ...,  1.0000e+00,
                #   3.1900e-04,  1.0000e+00],
                # ...,
                # [ 1.4988e-01,  9.8870e-01, -8.3111e-01,  ...,  1.0000e+00,
                #   2.0203e-03,  1.0000e+00],
                # [ 9.1295e-01,  4.0808e-01, -4.0648e-02,  ...,  1.0000e+00,
                #   2.1267e-03,  1.0000e+00],
                # [ 8.3666e-01, -5.4773e-01,  7.8319e-01,  ...,  1.0000e+00,
                #   2.2330e-03,  1.0000e+00]], device='cuda:0')

    
class scaled_dot_product_attention(nn.Module):
    def __init__(self):
        super(scaled_dot_product_attention, self).__init__()

    def forward(self, q, k, v):
        d_k = k.size(-1)
        k = k.transpose(-1, -2)
        att = torch.matmul(q, k) / np.sqrt(d_k)
        prob = nn.Softmax(dim=-1)(att)
        score = torch.matmul(prob, v)
        return score

class multihead_att(nn.Module):
    def __init__(self):
        super(multihead_att, self).__init__()
        self.head = 8
        self.hidden_size = 512
        self.dim = int(512 / 8) # 64
        self.scaled_dot_product_att = scaled_dot_product_attention()

        h2 = 128
        self.dense = nn.Linear(self.hidden_size, h2)
        self.relu = nn.ReLU(inplace=True)        

    def forward(self, q, k, v):
        # [256, 21, 512] -> [256, 21, 8, 64]
        q = q.view(config.batch, -1, self.head, self.dim) # view는 split을 대신할 수 있는가
        k = k.view(config.batch, -1, self.head, self.dim)
        v = v.view(config.batch, -1, self.head, self.dim)

        att = self.scaled_dot_product_att(q, k, v) # [256, 21, 8, 64]
        att = att.view(config.batch, -1, self.hidden_size) # view는 concat을 대신할 수 있는가 # [256, 21, 512]
        fc = self.dense(att) # [256, 21, 128]
        fc = self.relu(fc)
        return fc  

class add_norm(nn.Module):
    def __init__(self):
        super(add_norm, self).__init__()
        self.layer_norm = nn.LayerNorm(128)
    def forward(self, residual, output):
        return self.layer_norm(residual + output)

class positionwise_fcn(nn.Module):
    def __init__(self):
        super(positionwise_fcn, self).__init__()
        self.dense_up = nn.Linear(128, 512)
        self.dense_down = nn.Linear(512, 128)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, something):
        fc = self.dense_up(something)
        fc = self.relu(fc)
        fc = self.dense_down(fc)
        return fc

class q_k_v(nn.Module):
    def __init__(self):
        super(q_k_v, self).__init__()
        emb_size = 128
        hidden_size = 512
        self.dense = nn.Linear(emb_size, hidden_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, lookup):
        a = self.dense(lookup)
        a = self.relu(a) # [seq, batch, dim]
        return a

class encoder(nn.Module):
    def __init__(self, vocab_list, emb_weight):
        super(encoder, self).__init__()
        self.embedding = embedding(vocab_list, emb_weight)

        emb_size = 128
        self.dense_to_128 = nn.Linear(config.embedding_dim, emb_size)
        self.q = q_k_v()
        self.k = q_k_v()
        self.v = q_k_v()
        self.relu = nn.ReLU(inplace=True)

        self.multi_head = multihead_att()
        self.add_norm = add_norm()
        self.positionwise_fcn = positionwise_fcn()

    def forward(self, sentence):
        lookup = self.embedding(sentence) # 300 dim # [batch, seq, dim] [256, 21, 300]
        lookup = self.dense_to_128(lookup)
        lookup = self.relu(lookup) # [256, 21, 128]
 
        q = self.q(lookup) # [256, 21, 512]
        k = self.k(lookup)
        v = self.v(lookup)

        m_h = self.multi_head(q, k, v) # [256, 21, 128]
        
        something = self.add_norm(lookup, m_h) # [256, 21, 128]

        p_fcn = self.positionwise_fcn(something) # [256, 21, 128]

        output = self.add_norm(something, p_fcn) # [256, 21, 128]

        return output
        return 0

class classifier(nn.Module):
    def __init__(self, vocab_list, emb_weight):
        super(classifier, self).__init__()
        self.encoder_pre = encoder(vocab_list, emb_weight)
        self.encoder_hyp = encoder(vocab_list, emb_weight)
        self.dense_1 = nn.Linear(128, 1)
        self.dense_3 = nn.Linear(128, 3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, premise, hypothesis):
        premise = self.encoder_pre(premise) # [256, 21, 128]
        hypothesis =self.encoder_hyp(hypothesis) # [256, 21, 128]

        premise = premise.view(config.batch, 128, -1)  # [256, 128, 21]
        similarity = premise @ hypothesis # [256, 128, 128]

        similarity = self.dense_1(similarity) # [256, 128, 1]
        similarity = similarity.view(config.batch, -1) # [256, 128]
        similarity = self.relu(similarity)
        
        classes = self.dense_3(similarity) # [256, 3]
        return classes