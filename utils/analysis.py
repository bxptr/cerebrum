import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.metrics import compute_firing_rate, compute_synchronization
import copy

def analyze_results(V_record, time_array):
    fr = compute_firing_rate(V_record)
    sync = compute_synchronization(V_record)
    if np.isnan(fr).any() or np.isnan(sync):
        print("Warning: Analysis encountered NaN values.")
    else:
        print(f"Average firing rate across neurons: {fr.mean():.3f} spikes/time-unit")
        print(f"Synchronization measure (std of mean potential): {sync:.3f}")

        # Plot firing rate distribution
        plt.figure(figsize=(10,4))
        plt.hist(fr, bins=20, alpha=0.7)
        plt.xlabel("Firing Rate")
        plt.ylabel("Count")
        plt.title("Firing Rate Distribution")
        plt.tight_layout()
        plt.show()

def compute_functional_connectivity(V_record):
    # Compute functional connectivity as correlation matrix of node potentials
    # V_record shape: [time, nodes]
    if not np.isfinite(V_record).all():
        print("Warning: Functional connectivity computation received NaN values.")
        return np.nan
    # Normalize V_record to have zero mean
    V_norm = V_record - V_record.mean(axis=0)
    # Check for zero variance
    stddev = V_norm.std(axis=0)
    if np.any(stddev == 0):
        print("Warning: Some neurons have zero variance in membrane potentials. Adjusting to prevent NaN correlation.")
        stddev[stddev == 0] = 1e-8  # Small value to prevent division by zero
        V_norm = V_norm / stddev
    else:
        V_norm = V_norm / stddev
    # Compute correlation matrix
    corr_matrix = np.corrcoef(V_norm, rowvar=False)
    return corr_matrix

def sensitivity_analysis(model_params, simulate_func, base_metric, param_variation=0.1):
    """
    model_params: dictionary of parameters to vary
    simulate_func: function that runs a short simulation and returns a performance metric (e.g. firing rate mean)
    base_metric: baseline metric with current params
    param_variation: fractional change in parameter values for sensitivity.

    We'll vary each parameter by ±param_variation and see metric changes.
    """
    sensitivity_results = {}
    for p in model_params:
        orig_val = model_params[p]
        deltas = []
        for sign in [-1, 1]:
            new_val = orig_val * (1 + sign * param_variation)
            # Make a copy of model_params
            temp_params = copy.deepcopy(model_params)
            temp_params[p] = float(new_val)  # Ensure it's a float
            try:
                met = simulate_func(temp_params)
                if np.isnan(met):
                    print(f"Warning: Simulation metric is NaN for parameter {p} with value {new_val}")
                deltas.append((met - base_metric) / base_metric)
            except Exception as e:
                print(f"Error during sensitivity simulation for parameter {p} with value {new_val}: {e}")
                deltas.append(np.nan)
        # Calculate mean absolute delta, ignoring NaNs
        deltas = [d for d in deltas if not np.isnan(d)]
        if deltas:
            sensitivity_results[p] = np.mean(np.abs(deltas))
        else:
            sensitivity_results[p] = np.nan
    return sensitivity_results

def compute_uncertainty(model, data, samples=10, device='cpu'):
    """
    Compute uncertainty by making multiple forward passes (with dropout) 
    and measuring variance of the predicted adjacency.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for _ in range(samples):
            pred_adj = model(data.x, data.edge_index, data.edge_weight)
            preds.append(pred_adj.cpu().numpy())
    preds = np.array(preds)  # [samples, num_nodes, num_nodes]
    mean_pred = preds.mean(axis=0)
    std_pred = preds.std(axis=0)
    return mean_pred, std_pred

