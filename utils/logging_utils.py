import os
import torch

def save_checkpoint(path, model_state, info_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state": model_state, "info": info_dict}, path)

def load_checkpoint(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint["info"]

