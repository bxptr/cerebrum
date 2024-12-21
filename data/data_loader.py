import torch
from torch_geometric.utils import from_networkx
import os
import pandas as pd
import numpy as np

def load_graph_into_pyg(G, device):
    data = from_networkx(G)
    if 'weight' in data:
        data.edge_weight = data.weight
    else:
        data.edge_weight = torch.ones(data.edge_index.shape[1], device=device)
    data.num_nodes = G.number_of_nodes()
    data = data.to(device)
    return data

def load_real_connectivity_data(path, num_nodes):
    if not os.path.exists(path):
        print(f"Warning: Real connectivity data not found at {path}. Using default synthetic data.")
        return None
    try:
        mat = pd.read_csv(path, header=None).values
        if mat.shape[0] != num_nodes or mat.shape[1] != num_nodes:
            print("Warning: Real connectivity data size does not match simulated network size. Using default.")
            return None
        if np.isnan(mat).any() or np.isinf(mat).any():
            print("Warning: Real connectivity data contains NaN or Inf. Using default.")
            return None
        return torch.tensor(mat, dtype=torch.float32)
    except Exception as e:
        print(f"Error loading real connectivity data: {e}. Using default.")
        return None

def load_real_recording_data(path, num_nodes):
    if not os.path.exists(path):
        print(f"Warning: Real neural recordings not found at {path}. Using synthetic simulation data only.")
        return None
    try:
        data = pd.read_csv(path, header=None).values
        # Expecting something like [time, nodes]
        if data.shape[1] != num_nodes:
            print("Warning: Real recording data size does not match network size. Using synthetic data.")
            return None
        if np.isnan(data).any() or np.isinf(data).any():
            print("Warning: Real recording data contains NaN or Inf. Using synthetic data.")
            return None
        return data
    except Exception as e:
        print(f"Error loading real recording data: {e}. Using synthetic data.")
        return None

