import matplotlib.pyplot as plt
import numpy as np

def plot_potentials(time_array, V_traces, sample_neurons=[0]):
    plt.figure(figsize=(10,5))
    for neuron in sample_neurons:
        if neuron >= V_traces.shape[1]:
            print(f"Warning: Neuron index {neuron} out of range for V_traces with shape {V_traces.shape}")
            continue
        plt.plot(time_array, V_traces[:, neuron], label=f"Neuron {neuron}")
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend()
    plt.title('Membrane Potential Traces')
    plt.tight_layout()
    plt.show()

def plot_adjacency(matrix, title="Adjacency Matrix"):
    if not np.isfinite(matrix).all():
        print(f"Warning: Adjacency matrix contains NaN or Inf values. Plot may be inaccurate.")
    plt.figure(figsize=(6,6))
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

