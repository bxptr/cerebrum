import os
import requests
import gzip
import shutil
import networkx as nx
import numpy as np
import pandas as pd

def download_file(url, dest_path):
    """Downloads a file from a URL to a destination path."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Downloaded {url} to {dest_path}")
    else:
        raise Exception(f"Failed to download {url}")

def process_connectome(adjacency_list_path, output_csv_path, desired_size=200):
    """
    Processes an adjacency list (with headers and weights) to create an adjacency matrix CSV.
    Trims or pads the network to the desired size.
    """
    # Read the adjacency list CSV with pandas
    df = pd.read_csv(adjacency_list_path)
    
    # Ensure columns are as expected
    if {'Source', 'Target', 'Weight'}.issubset(df.columns):
        edge_list = df[['Source', 'Target', 'Weight']].values
    else:
        raise ValueError("The input file must contain 'Source', 'Target', and 'Weight' columns.")
    
    # Create the graph using NetworkX
    G = nx.DiGraph()  # Directed graph (use nx.Graph() for undirected)
    for source, target, weight in edge_list:
        G.add_edge(source.strip(), target.strip(), weight=float(weight))
    
    # Ensure the graph has at least the desired number of nodes
    all_nodes = list(G.nodes())
    while len(all_nodes) < desired_size:
        G.add_node(f"Node_{len(all_nodes)}")
        all_nodes.append(f"Node_{len(all_nodes)}")
    
    # Select a fixed number of nodes
    current_size = G.number_of_nodes()
    if current_size > desired_size:
        nodes_to_remove = list(G.nodes())[desired_size:]
        G.remove_nodes_from(nodes_to_remove)
        print(f"Trimmed network from {current_size} to {desired_size} nodes.")
    elif current_size < desired_size:
        # Add isolated nodes
        for i in range(desired_size - current_size):
            G.add_node(current_size + i)
        print(f"Padded network to {desired_size} nodes with isolated nodes.")

    # Generate adjacency matrix
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='sorted')
    print("Relabeled nodes to sequential integers starting from 0.")
    adj_matrix = nx.to_numpy_array(G, nodelist=range(desired_size), weight='weight')
    
    # Save to CSV
    np.savetxt(output_csv_path, adj_matrix, delimiter=',')
    print(f"Saved adjacency matrix to {output_csv_path}")


def generate_synthetic_recordings(output_csv_path, num_nodes=200, total_time=300.0, dt=0.01):
    """
    Generates synthetic neural recordings and saves them as a CSV.
    Each row corresponds to a time point, and each column to a neuron.
    """
    steps = int(total_time / dt)
    time = np.linspace(0, total_time, steps)
    
    # Generate synthetic membrane potentials with random spikes
    V_record = -65 + 5 * np.random.randn(steps, num_nodes)  # Baseline with noise
    spike_prob = 0.01  # Probability of spike at each time step
    
    for t in range(steps):
        spikes = np.random.rand(num_nodes) < spike_prob
        V_record[t, spikes] = 30.0  # Spike value
    
    # Save to CSV
    np.savetxt(output_csv_path, V_record, delimiter=',')
    print(f"Saved synthetic neural recordings to {output_csv_path}")

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Define URLs and paths
    connectome_url = "https://raw.githubusercontent.com/openworm/c302/refs/heads/master/c302/data/herm_full_edgelist.csv"
    adjacency_list_path = "data/C_elegans_synapses.csv"
    connectivity_csv_path = "data/real_connectivity.csv"
    recordings_csv_path = "data/real_recordings.csv"
    
    # Download the C. elegans connectome adjacency list
    if not os.path.exists(adjacency_list_path):
        print("Downloading C. elegans connectome data...")
        download_file(connectome_url, adjacency_list_path)
    else:
        print(f"Connectome file already exists at {adjacency_list_path}")
    
    # Process the connectome to create adjacency matrix CSV
    process_connectome(adjacency_list_path, connectivity_csv_path, desired_size=200)
    
    # Generate synthetic neural recordings
    generate_synthetic_recordings(recordings_csv_path, num_nodes=200, total_time=300.0, dt=0.01)

if __name__ == "__main__":
    main()

