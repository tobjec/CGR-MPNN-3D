import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric as tg
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool

from sklearn.metrics import mean_absolute_error, mean_squared_error

class GNN(nn.Module):
    def __init__(self, num_node_features, num_edge_features, depth=3, hidden_size=300, p_dropout=0.02):
        super(GNN, self).__init__()

        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = p_dropout

        self.edge_init = nn.Linear(num_node_features + num_edge_features, self.hidden_size)
        self.convs = torch.nn.ModuleList()
        for _ in range(self.depth):
            self.convs.append(DMPNNConv(self.hidden_size))
        self.edge_to_node = nn.Linear(num_node_features + self.hidden_size, self.hidden_size)
        self.pool = global_add_pool
        self.ffn = nn.Linear(self.hidden_size, 1)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        # initial edge features
        row, col = edge_index
        h_0 = F.relu(self.edge_init(torch.cat([x[row], edge_attr], dim=1)))
        h = h_0

        # convolutions
        for l in range(self.depth):
            _, h = self.convs[l](edge_index, h)
            h += h_0
            h = F.dropout(F.relu(h), self.dropout, training=self.training)

        # dmpnn edge -> node aggregation
        s, _ = self.convs[l](edge_index, h) #only use for summing
        q  = torch.cat([x,s], dim=1)
        h = F.relu(self.edge_to_node(q))
        return self.ffn(self.pool(h, batch)).squeeze(-1)

class DMPNNConv(MessagePassing):
    def __init__(self, hidden_size):
        super(DMPNNConv, self).__init__(aggr='add')
        self.lin = nn.Linear(hidden_size, hidden_size)

    def forward(self, edge_index, edge_attr):
        row, col = edge_index
        a_message = self.propagate(edge_index, x=None, edge_attr=edge_attr)
        rev_message = torch.flip(edge_attr.view(edge_attr.size(0) // 2, 2, -1), dims=[1]).view(edge_attr.size(0), -1)

        return a_message, self.lin(a_message[row] - rev_message)

    def message(self, edge_attr):
        return edge_attr
