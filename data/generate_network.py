import networkx as nx
import numpy as np

def generate_graph(n=200, graph_type="erdos_renyi", p=0.05):
    if graph_type == "erdos_renyi":
        G = nx.erdos_renyi_graph(n, p)
    elif graph_type == "small_world":
        # Watts–Strogatz small-world network
        k = max(1, int(0.1 * n))  # Ensure k >=1
        G = nx.watts_strogatz_graph(n, k, 0.1)
    elif graph_type == "scale_free":
        G = nx.scale_free_graph(n).to_undirected()
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Remove self-loops and parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.Graph(G)  # Convert to simple graph

    # Assign random initial weights (synaptic strengths) between 0.1 and 1.0
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.1, 1.0)

    return G

