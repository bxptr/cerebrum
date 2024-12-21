import torch
import numpy as np
from tqdm import tqdm
from config import CONFIG
from data.generate_network import generate_graph
from data.data_loader import load_graph_into_pyg, load_real_connectivity_data, load_real_recording_data
from models.gnn_model import SynapticGNN
from models.combined_model import CombinedHHGNN
from models.connectivity_inference_model import ConnectivityInferenceGNN
from models.neuron_editor import NeuronEditor
from utils.visualization import plot_potentials, plot_adjacency
from utils.analysis import analyze_results, compute_functional_connectivity, sensitivity_analysis, compute_uncertainty
from utils.training import train_connectivity_inference
import matplotlib.pyplot as plt

def quick_simulation_for_metric(hh_params, I_ext, data, dt, steps, save_interval, device):
    # Run a short simulation to get a metric (average firing rate or mean potential)
    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]

    stp_params = CONFIG["stp_params"]
    simulation_gnn = SynapticGNN(in_channels=1, hidden_dim=CONFIG["gnn_hidden_dim"]).to(device)
    combined_model = CombinedHHGNN(hh_params, simulation_gnn, stp_params, num_edges, 
                                   refractory_period=CONFIG["refractory_period"], 
                                   dt=dt, 
                                   synaptic_delay_steps=CONFIG["synaptic_delay_steps"]).to(device)
    combined_model.initialize_state(num_nodes)

    V = -65.0 * torch.ones(num_nodes, device=device)
    m = 0.05 * torch.ones(num_nodes, device=device)
    h = 0.6 * torch.ones(num_nodes, device=device)
    n = 0.32 * torch.ones(num_nodes, device=device)
    mCa = 0.0 * torch.ones(num_nodes, device=device)

    prev_spikes = (V > 0).float()

    V_rec = []
    for step in range(steps):
        try:
            V, m, h, n, mCa, W_eff = combined_model(V, m, h, n, mCa, I_ext, data, dt, prev_spikes)
        except ValueError as e:
            print(f"Simulation halted due to error: {e}")
            break
        if not torch.isfinite(V).all():
            print("Simulation encountered NaN values. Halting simulation.")
            break
        if step % save_interval == 0:
            V_rec.append(V.detach().cpu().numpy())
        prev_spikes = (V > 0).float()

    if len(V_rec) == 0:
        return np.nan
    V_rec = np.array(V_rec)
    return V_rec.mean()

def parameter_optimization(hh_params, data, device, dt):
    best_metric = None
    best_params = None
    steps = 100  # shorter simulation for speed

    for I_ext_val in CONFIG["parameter_search_space"]["external_current"]:
        for gCa_val in CONFIG["parameter_search_space"]["gCa"]:
            temp_hh_params = hh_params.copy()
            temp_hh_params["gCa"] = gCa_val
            I_ext = I_ext_val * torch.ones(data.num_nodes, device=device)
            metric = quick_simulation_for_metric(temp_hh_params, I_ext, data, dt, steps, CONFIG["save_interval"], device)
            if np.isnan(metric):
                continue
            # Define target metric
            target = -50.0  # placeholder
            score = abs(metric - target)
            if (best_metric is None) or (score < best_metric):
                best_metric = score
                best_params = (I_ext_val, gCa_val)

    if best_params is not None:
        I_ext_val, gCa_val = best_params
        CONFIG["external_current"] = I_ext_val
        hh_params["gCa"] = gCa_val
        print(f"Parameter optimization complete. Using external_current={I_ext_val}, gCa={gCa_val}")
    else:
        print("Parameter optimization failed to find a valid parameter set.")

def main():
    device = CONFIG["device"]
    dt = CONFIG["time_step"]
    total_time = CONFIG["total_time"]
    steps = int(total_time / dt)
    CONFIG["synaptic_delay_steps"] = int(CONFIG["synaptic_delay"] / dt)

    hh_params = {
        "gNa": CONFIG["conductance"]["gNa"],
        "gK": CONFIG["conductance"]["gK"],
        "gL": CONFIG["conductance"]["gL"],
        "gCa": CONFIG["conductance"]["gCa"],
        "ENa": CONFIG["reversal_potentials"]["ENa"],
        "EK": CONFIG["reversal_potentials"]["EK"],
        "EL": CONFIG["reversal_potentials"]["EL"],
        "ECa": CONFIG["reversal_potentials"]["ECa"],
        "Cm": CONFIG["Cm"]
    }

    stp_params = CONFIG["stp_params"]
    time_array = np.arange(0, total_time, dt)

    # Attempt to load real data
    real_connectivity = load_real_connectivity_data(CONFIG["real_connectivity_data_path"], CONFIG["network_size"])
    real_recordings = load_real_recording_data(CONFIG["real_recording_data_path"], CONFIG["network_size"])

    if real_connectivity is not None and real_recordings is not None:
        target_mean = real_recordings.mean()
        adjustment = (target_mean + 65) * 0.1
        CONFIG["external_current"] += adjustment
        print(f"Adjusted external current from real data to {CONFIG['external_current']:.2f}")

    # Parameter optimization if enabled
    if CONFIG["use_parameter_optimization"]:
        G_opt = generate_graph(CONFIG["network_size"], CONFIG["graph_type"], CONFIG["graph_p"])
        data_opt = load_graph_into_pyg(G_opt, device)
        try:
            parameter_optimization(hh_params, data_opt, device, dt)
        except Exception as e:
            print(f"Parameter optimization failed: {e}")

    # Apply disease model if active
    neuron_editor = NeuronEditor(disease_name=CONFIG["active_disease"], disease_models=CONFIG["disease_models"])
    hh_params = neuron_editor.apply_disease_modifications(hh_params)

    # Experiment loop
    for graph_type in CONFIG["experiments"]["topologies"]:
        for run_id in range(CONFIG["experiments"]["runs_per_topology"]):
            print(f"\nRunning experiment with {graph_type}, run {run_id+1}...")
            G = generate_graph(CONFIG["network_size"], graph_type, CONFIG["graph_p"])
            data = load_graph_into_pyg(G, device)
            num_nodes = data.num_nodes
            num_edges = data.edge_index.shape[1]

            V = -65.0 * torch.ones(num_nodes, device=device)
            m = 0.05 * torch.ones(num_nodes, device=device)
            h = 0.6 * torch.ones(num_nodes, device=device)
            n = 0.32 * torch.ones(num_nodes, device=device)
            mCa = 0.0 * torch.ones(num_nodes, device=device)

            I_ext = CONFIG["external_current"] * torch.ones(num_nodes, device=device)
            simulation_gnn = SynapticGNN(in_channels=1, hidden_dim=CONFIG["gnn_hidden_dim"]).to(device)
            combined_model = CombinedHHGNN(hh_params, simulation_gnn, stp_params, num_edges, 
                                           refractory_period=CONFIG["refractory_period"],
                                           dt=dt, 
                                           synaptic_delay_steps=CONFIG["synaptic_delay_steps"]).to(device)
            combined_model.initialize_state(num_nodes)

            V_record = []
            prev_spikes = (V > 0).float()

            for step_idx in tqdm(range(steps), desc="Simulating"):
                try:
                    V, m, h, n, mCa, W_eff = combined_model(V, m, h, n, mCa, I_ext, data, dt, prev_spikes)
                except ValueError as e:
                    print(f"Simulation halted due to error: {e}")
                    break

                if not torch.isfinite(V).all():
                    print("Simulation encountered NaN values. Halting simulation.")
                    break

                if step_idx % CONFIG["save_interval"] == 0:
                    V_record.append(V.detach().cpu().numpy())

                prev_spikes = (V > 0).float()

            if len(V_record) == 0:
                print("No valid recordings were saved for this run.")
                continue

            V_record = np.array(V_record)
            analyze_results(V_record, time_array[::CONFIG["save_interval"]])

            plot_potentials(time_array[::CONFIG["save_interval"]], V_record, sample_neurons=[0,1,2])

            true_adj = torch.zeros(num_nodes, num_nodes, device=device)
            for (u, v) in G.edges():
                w = G[u][v]['weight']
                true_adj[u, v] = w
                true_adj[v, u] = w

            if torch.max(true_adj) > 0:
                true_adj = true_adj / torch.max(true_adj)
            else:
                print("Warning: True adjacency matrix has all zero values.")

            mean_potential = V_record.mean(axis=0)
            std_potential = V_record.std(axis=0)
            spike_counts = (V_record > 0).sum(axis=0)

            mean_potential_norm = (mean_potential - mean_potential.mean()) / (mean_potential.std() + 1e-8)
            std_potential_norm = (std_potential - std_potential.mean()) / (std_potential.std() + 1e-8)
            spike_counts_norm = (spike_counts - spike_counts.mean()) / (spike_counts.std() + 1e-8)

            features = np.stack([mean_potential_norm, std_potential_norm, spike_counts_norm], axis=1)
            node_features = torch.tensor(features, dtype=torch.float32, device=device)
            data.x = node_features

            print(f"Node Features - Shape: {node_features.shape}")
            print(f"Node Features - Mean: {node_features.mean().item():.4f}, Std: {node_features.std().item():.4f}")

            inference_model = ConnectivityInferenceGNN(in_channels=3, hidden_dim=32, num_nodes=num_nodes, dropout=CONFIG["inference_model_dropout"]).to(device)
            if CONFIG["training"]:
                try:
                    inference_model = train_connectivity_inference(inference_model, data, true_adj,
                                                                    epochs=CONFIG["inference_epochs"],
                                                                    lr=CONFIG["inference_lr"],
                                                                    device=device)
                except Exception as e:
                    print(f"Connectivity inference training failed: {e}")
                    continue

                inference_model.eval()
                with torch.no_grad():
                    pred_adj = inference_model(data.x, data.edge_index, data.edge_weight)
                    mse = ((pred_adj - true_adj)**2).mean().item()
                    pred_np = pred_adj.cpu().numpy()
                    true_np = true_adj.cpu().numpy()
                    if np.std(pred_np.flatten()) == 0 or np.std(true_np.flatten()) == 0:
                        corr = 0.0
                        print("Warning: Zero standard deviation in adjacency matrices. Setting correlation to 0.")
                    else:
                        corr = np.corrcoef(pred_np.flatten(), true_np.flatten())[0,1]
                print(f"Connectivity inference MSE: {mse:.4f}, Correlation: {corr:.4f}")

                if CONFIG["compute_uncertainty"]:
                    mean_pred, std_pred = compute_uncertainty(inference_model, data, samples=CONFIG["uncertainty_samples"], device=device)
                    print(f"Uncertainty in inference (mean std): {std_pred.mean():.4f}")

                plot_adjacency(true_adj.cpu().numpy(), "True Adjacency")
                plot_adjacency(pred_adj.cpu().numpy(), "Predicted Adjacency")

                torch.save(inference_model.state_dict(), CONFIG["weights_dir"] + f"/model_{graph_type}.pt")

            func_conn = compute_functional_connectivity(V_record)
            if not np.isnan(func_conn).all():
                plt.figure(figsize=(6,6))
                plt.imshow(func_conn, cmap='hot', interpolation='nearest')
                plt.colorbar()
                plt.title("Functional Connectivity (Correlation)")
                plt.tight_layout()
                plt.show()
            else:
                print("Functional connectivity computation resulted in NaN values.")

            base_metric = V_record.mean()
            def simulate(p):
                return quick_simulation_for_metric(p, CONFIG["external_current"]*torch.ones(num_nodes, device=device), data, dt, 100, CONFIG["save_interval"], device)
            sens_results = sensitivity_analysis(hh_params, simulate, base_metric, param_variation=0.1)
            print("Sensitivity analysis results:", sens_results)

if __name__ == "__main__":
    main()

