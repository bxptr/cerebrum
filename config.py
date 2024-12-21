import torch

# Enhanced configuration for advanced experiments including refractory and synaptic delays, parameter optimization, and real data integration.

CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Simulation parameters
    "network_size": 200,
    "time_step": 0.01,
    "total_time": 100.0,
    "save_interval": 10,
    "external_current": 8.0,

    # Graph configuration
    "graph_type": "scale_free",
    "graph_p": 0.05,

    # Hodgkin-Huxley parameters including Ca current (biologically plausible)
    "conductance": {
        "gNa": 120.0,
        "gK": 36.0,
        "gL": 0.3,
        "gCa": 1.0
    },
    "reversal_potentials": {
        "ENa": 50.0,
        "EK": -77.0,
        "EL": -54.4,
        "ECa": 120.0
    },
    "Cm": 1.0,

    # Synaptic plasticity parameters
    "stp_params": {
        "U": 0.2,
        "D": 200.0,
        "F": 150.0,
        "initial_u": 0.2
    },

    # GNN architecture
    "gnn_hidden_dim": 64,

    # Connectivity inference training parameters
    "training": True,
    "inference_epochs": 1000,
    "inference_lr": 0.001,
    "inference_batch_size": 1,
    "weights_dir": "results/weights",

    # Experiment settings
    "experiments": {
        "topologies": ["scale_free"], # erdos_renyi, small_world, scale_free 
        "runs_per_topology": 1
    },

    # New additions
    # Refractory and synaptic delays
    "refractory_period": 2.0,  # ms of refractory period after spike
    "synaptic_delay": 1.0,     # ms delay in synaptic transmission
    "synaptic_delay_steps": 0, # Will be computed based on dt in run_experiments

    # Paths for real data (placeholders)
    "real_connectivity_data_path": "data/real_connectivity.csv",
    "real_recording_data_path": "data/real_recordings.csv",

    # Parameter optimization toggles
    "use_parameter_optimization": True,
    "parameter_search_space": {
        # We vary external_current and gCa slightly for demonstration
        "external_current": [7.5, 8.0, 8.5],
        "gCa": [0.8, 1.0, 1.2]
    },

    # Uncertainty computation toggles
    "compute_uncertainty": True,
    "inference_model_dropout": 0.1,  # Use dropout for uncertainty estimation
    "uncertainty_samples": 10,

    "save_plots": True,
    "plot_dir": "results/plots",

    # Disease models configuration
    "active_disease": "Epilepsy",  # Options: None, "Parkinson's", "Epilepsy"
    "disease_models": {
        "Parkinson's": {
            "gNa_scale": 1.0,
            "gK_scale": 0.8,
            "gL_scale": 1.0,
            "gCa_scale": 1.5
        },
        "Epilepsy": {
            "gNa_scale": 1.2,
            "gK_scale": 1.0,
            "gL_scale": 0.95,
            "gCa_scale": 1.2
        }
    }
}

