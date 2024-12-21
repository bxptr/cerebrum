import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class SynapticGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels=1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.relu = nn.ReLU()
        self.conv2 = GCNConv(hidden_dim, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x

