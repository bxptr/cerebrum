import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm

def train_connectivity_inference(model, data, true_adj, epochs=100, lr=0.001, device='cpu'):
    model.to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    loss_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred_adj = model(data.x, data.edge_index, data.edge_weight)
        loss = F.mse_loss(pred_adj, true_adj)
        
        if torch.isnan(loss):
            print(f"Training halted at epoch {epoch} due to NaN loss.")
            break

        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 10 == 0:
            print(f"[Connectivity Inference] Epoch {epoch}, Loss: {loss.item():.4f}")

    return model

