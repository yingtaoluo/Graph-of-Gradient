# -*- coding: utf-8 -*-
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
import warnings

warnings.filterwarnings('ignore')


def normalize_scale(x):
    return x/(x.mean()+1e-8)


def min_max_scale(x):
    # Scale to [0, 1] range
    min_val = x.min()
    max_val = x.max()
    return (x - min_val) / (max_val - min_val + 1e-8)


class Adversary(nn.Module):
    def __init__(self, icd_size, pro_size, length, hidden_size=16, dropout=0.1, batch_first=True):
        super().__init__()
        self.dia_embedding = nn.Linear(icd_size, hidden_size)
        self.pro_embedding = nn.Linear(pro_size, hidden_size)
        self.rnn_model = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, dropout=dropout,
                                 bidirectional=False, batch_first=batch_first)
        self.output_func = SimpleGCN(hidden_size*length, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, edges):
        x1 = self.dia_embedding(x1)
        x2 = self.pro_embedding(x2)
        # embed_data (N, T, H)
        rnn_output1, _ = self.rnn_model(x1)  # (N, T, H)
        rnn_output2, _ = self.rnn_model(x2)  # (N, T, H)
        rnn_output = (rnn_output1 + rnn_output2).reshape(x1.shape[0], -1)  # (N, T*H)
        output = self.output_func(rnn_output, edges)  # (N, 1)
        # output = self.sigmoid(output)
        scaled_out = normalize_scale(output)

        return scaled_out


class SimpleGCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        # A single GCN layer
        x = self.conv1(x, edge_index)
        return F.leaky_relu_(x)+1

