# -*- coding: utf-8 -*-
import pdb
import torch
import warnings
from Fair.train import GPU
import torch.nn as nn
import numpy as np
import math
import copy
from torch.autograd import Variable
import torch.nn.functional as F

warnings.filterwarnings('ignore')

device = torch.device("cuda:%d" % GPU if torch.cuda.is_available() and GPU >= 0 else "cpu")

torch.backends.cudnn.deterministic = True


class SingleAttention(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='add', time_aware=False):
        super(SingleAttention, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim
        self.time_aware = time_aware

        self.attn = None

        if attention_type == 'add':
            if self.time_aware == True:
                # self.Wx = nn.Parameter(torch.randn(attention_input_dim+1, attention_hidden_dim))
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
                self.Wtime_aware = nn.Parameter(torch.randn(1, attention_hidden_dim))
                nn.init.kaiming_uniform_(self.Wtime_aware, a=math.sqrt(5))
            else:
                self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.bh = nn.Parameter(torch.zeros(attention_hidden_dim, ))
            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1, ))

            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == 'mul':
            self.Wa = nn.Parameter(torch.randn(attention_input_dim, attention_input_dim))
            self.ba = nn.Parameter(torch.zeros(1, ))

            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))
        elif attention_type == 'concat':
            if self.time_aware == True:
                self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim + 1, attention_hidden_dim))
            else:
                self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim, attention_hidden_dim))

            self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
            self.ba = nn.Parameter(torch.zeros(1, ))

            nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        elif attention_type == 'new':
            self.Wt = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))
            self.Wx = nn.Parameter(torch.randn(attention_input_dim, attention_hidden_dim))

            self.rate = nn.Parameter(torch.ones(1))
            nn.init.kaiming_uniform_(self.Wx, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.Wt, a=math.sqrt(5))

        else:
            raise RuntimeError('Wrong attention type.')

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, demo=None):

        batch_size, time_step, input_dim = input.size()  # batch_size * time_step * hidden_dim(i)

        time_decays = torch.tensor(range(time_step - 1, -1, -1), dtype=torch.float32).unsqueeze(-1).unsqueeze(0).to(
            device)  # 1*t*1
        b_time_decays = time_decays.repeat(batch_size, 1, 1)  # b t 1

        if self.attention_type == 'add':  # B*T*I  @ H*I
            q = torch.matmul(input[:, -1, :], self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            if self.time_aware == True:
                # k_input = torch.cat((input, time), dim=-1)
                k = torch.matmul(input, self.Wx)  # b t h
                # k = torch.reshape(k, (batch_size, 1, time_step, self.attention_hidden_dim)) #B*1*T*H
                time_hidden = torch.matmul(b_time_decays, self.Wtime_aware)  # b t h
            else:
                k = torch.matmul(input, self.Wx)  # b t h
                # k = torch.reshape(k, (batch_size, 1, time_step, self.attention_hidden_dim)) #B*1*T*H
            h = q + k + self.bh  # b t h
            if self.time_aware == True:
                h += time_hidden
            h = self.tanh(h)  # B*T*H
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t
        elif self.attention_type == 'mul':
            e = torch.matmul(input[:, -1, :], self.Wa)  # b i
            e = torch.matmul(e.unsqueeze(1), input.permute(0, 2, 1)).squeeze() + self.ba  # b t
        elif self.attention_type == 'concat':
            q = input[:, -1, :].unsqueeze(1).repeat(1, time_step, 1)  # b t i
            k = input
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            if self.time_aware == True:
                c = torch.cat((c, b_time_decays), dim=-1)  # B*T*2I+1
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == 'new':

            q = torch.matmul(input[:, -1, :], self.Wt)  # b h
            q = torch.reshape(q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            k = torch.matmul(input, self.Wx)  # b t h
            dot_product = torch.matmul(q, k.transpose(1, 2)).squeeze()  # b t
            denominator = self.rate * torch.log(2.71828 + (1 - self.sigmoid(dot_product)) * (b_time_decays.squeeze()))
            e = self.tanh(dot_product / denominator)  # b * t

        # s = torch.sum(e, dim=-1, keepdim=True)
        # mask = subsequent_mask(time_step).to(device) # 1 t t 下三角
        # scores = e.masked_fill(mask == 0, -1e9)# b t t 下三角
        a = self.softmax(e)  # B*T
        self.attn = a
        v = torch.matmul(a.unsqueeze(1), input).squeeze()  # B*I

        return v, a


class FinalAttentionQKV(nn.Module):
    def __init__(self, attention_input_dim, attention_hidden_dim, attention_type='add', dropout=None):
        super(FinalAttentionQKV, self).__init__()

        self.attention_type = attention_type
        self.attention_hidden_dim = attention_hidden_dim
        self.attention_input_dim = attention_input_dim

        self.W_q = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_k = nn.Linear(attention_input_dim, attention_hidden_dim)
        self.W_v = nn.Linear(attention_input_dim, attention_hidden_dim)

        self.W_out = nn.Linear(attention_hidden_dim, 1)

        self.b_in = nn.Parameter(torch.zeros(1, ))
        self.b_out = nn.Parameter(torch.zeros(1, ))

        nn.init.kaiming_uniform_(self.W_q.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_k.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_v.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_out.weight, a=math.sqrt(5))

        self.Wh = nn.Parameter(torch.randn(2 * attention_input_dim, attention_hidden_dim))
        self.Wa = nn.Parameter(torch.randn(attention_hidden_dim, 1))
        self.ba = nn.Parameter(torch.zeros(1, ))

        nn.init.kaiming_uniform_(self.Wh, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Wa, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):

        batch_size, time_step, input_dim = input.size()  # batch_size * input_dim + 1 * hidden_dim(i)
        input_q = self.W_q(input[:, -1, :])  # b h
        input_k = self.W_k(input)  # b t h
        input_v = self.W_v(input)  # b t h

        if self.attention_type == 'add':  # B*T*I  @ H*I

            q = torch.reshape(input_q, (batch_size, 1, self.attention_hidden_dim))  # B*1*H
            h = q + input_k + self.b_in  # b t h
            h = self.tanh(h)  # B*T*H
            e = self.W_out(h)  # b t 1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        elif self.attention_type == 'mul':
            q = torch.reshape(input_q, (batch_size, self.attention_hidden_dim, 1))  # B*h 1
            e = torch.matmul(input_k, q).squeeze()  # b t

        elif self.attention_type == 'concat':
            q = input_q.unsqueeze(1).repeat(1, time_step, 1)  # b t h
            k = input_k
            c = torch.cat((q, k), dim=-1)  # B*T*2I
            h = torch.matmul(c, self.Wh)
            h = self.tanh(h)
            e = torch.matmul(h, self.Wa) + self.ba  # B*T*1
            e = torch.reshape(e, (batch_size, time_step))  # b t

        a = self.softmax(e)  # B*T
        if self.dropout is not None:
            a = self.dropout(a)
        v = torch.matmul(a.unsqueeze(1), input_v).squeeze()  # B*I

        return v, a


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
        device)
    return torch.index_select(a, dim, order_index).to(device)


class PositionwiseFeedForward(nn.Module):  # new added

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):  # new added / not use anymore

    def __init__(self, d_model, dropout, max_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0  # 下三角矩阵


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)  # b h t d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)  # b h t t
    if mask is not None:  # 1 1 t t
        scores = scores.masked_fill(mask == 0, -1e9)  # b h t t 下三角
    p_attn = F.softmax(scores, dim=-1)  # b h t t
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn  # b h t v (d_k)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.):

        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, self.d_k * self.h), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # 1 1 t t

        nbatches = query.size(0)  # b
        input_dim = query.size(1)  # i+1
        feature_dim = query.size(-1)  # i+1

        # input size -> # batch_size * d_input * hidden_dim

        # d_model => h * d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]  # b num_head d_input d_k

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)  # b num_head d_input d_v (d_k)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)  # batch_size * d_input * hidden_dim

        return self.final_linear(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def cov(m, y=None):
    if y is not None:
        m = torch.cat((m, y), dim=0)
    m_exp = torch.mean(m, dim=1)
    x = m - m_exp[:, None]
    cov = 1 / (x.size(1) - 1) * x.mm(x.t())
    return cov


class SublayerConnection(nn.Module):

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value)


class ConCare(nn.Module):
    def __init__(self, icd_size, pro_size, hidden_dim, MHD_num_head, drop=0.1):
        super(ConCare, self).__init__()
        self.input_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.d_model = hidden_dim
        self.MHD_num_head = MHD_num_head
        self.d_ff = hidden_dim
        self.output_dim = icd_size

        self.dia_embedding = nn.Linear(icd_size, hidden_dim)
        self.pro_embedding = nn.Linear(pro_size, hidden_dim)

        # layers
        self.PositionalEncoding = PositionalEncoding(self.d_model, dropout=0, max_len=400)

        self.GRUs = clones(nn.GRU(1, self.hidden_dim, batch_first=True), self.input_dim)
        self.LastStepAttentions = clones(
            SingleAttention(self.hidden_dim, 8, attention_type='concat', time_aware=True,), self.input_dim)

        self.FinalAttentionQKV = FinalAttentionQKV(self.hidden_dim, self.hidden_dim, attention_type='mul',
                                                   dropout=drop)

        self.MultiHeadedAttention = MultiHeadedAttention(self.MHD_num_head, self.d_model, dropout=drop)
        self.SublayerConnection = SublayerConnection(self.d_model, dropout=drop)

        self.PositionwiseFeedForward = PositionwiseFeedForward(self.d_model, self.d_ff, dropout=drop)

        self.output = nn.Linear(self.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(p=drop)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input_data, choice):
        # input shape [batch_size, timestep, feature_dim]
        if choice == 'dia':
            embed_data = self.dia_embedding(input_data)
        else:
            embed_data = self.pro_embedding(input_data)

        time_output = torch.zeros(input_data.shape[0], input_data.shape[1], self.output_dim).to(device)

        for j in range(input_data.size(1)):
            input_data = embed_data[:, :j+1]
            batch_size = input_data.size(0)
            time_step = input_data.size(1)
            feature_dim = input_data.size(2)
            assert (feature_dim == self.input_dim)  # input Tensor : 256 * 48 * 76
            assert (self.d_model % self.MHD_num_head == 0)

            GRU_embeded_input = self.GRUs[0](input_data[:, :, 0].unsqueeze(-1),
                                             Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(
                                                 device))[
                0]  # b t h
            Attention_embeded_input, self.gru_atten = self.LastStepAttentions[0](GRU_embeded_input)
            Attention_embeded_input = Attention_embeded_input.unsqueeze(1)  # b 1 h
            self.gru_atten = self.gru_atten.unsqueeze(1)  # b 1 h
            for i in range(feature_dim - 1):
                embeded_input = self.GRUs[i + 1](input_data[:, :, i + 1].unsqueeze(-1),
                                                 Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(
                                                     device))[0]  # b 1 h
                embeded_input, atten = self.LastStepAttentions[i + 1](embeded_input)
                embeded_input = embeded_input.unsqueeze(1)  # b 1 h
                atten = atten.unsqueeze(1)  # b 1 h

                Attention_embeded_input = torch.cat((Attention_embeded_input, embeded_input), 1)  # b i h
                self.gru_atten = torch.cat((self.gru_atten, atten), 1)  # b i h

            posi_input = self.dropout(Attention_embeded_input)  # batch_size * d_input+1 * hidden_dim

            contexts = self.SublayerConnection(posi_input,
                                               lambda x: self.MultiHeadedAttention(posi_input, posi_input, posi_input,
                                                                                   None))

            contexts = self.SublayerConnection(contexts, lambda x: self.PositionwiseFeedForward(contexts))
            weighted_contexts = self.FinalAttentionQKV(contexts)[0]
            output = self.output(weighted_contexts)
            # output = self.sigmoid(output)
            time_output[:, j] = output

        return time_output
