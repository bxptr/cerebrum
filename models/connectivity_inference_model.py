import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class ConnectivityInferenceGNN(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_nodes, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.relu1 = nn.ReLU()
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.conv3 = GCNConv(hidden_dim, hidden_dim // 2)
        self.relu3 = nn.ReLU()
        self.conv4 = GCNConv(hidden_dim // 2, hidden_dim // 4)
        self.relu4 = nn.ReLU()
        self.out_lin = nn.Linear(hidden_dim // 4, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.num_nodes = num_nodes

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index, edge_weight)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.conv4(x, edge_index, edge_weight)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.out_lin(x)  # [num_nodes, 1]
        
        # Expand to [num_nodes, num_nodes]
        adjacency_pred = torch.relu(torch.matmul(x, x.T))
        
        # Ensure symmetry
        adjacency_pred = (adjacency_pred + adjacency_pred.T) / 2
        return adjacency_pred

