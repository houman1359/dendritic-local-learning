"""
Plotting utilities and helper functions.

This module provides common plotting utilities used across different visualization modules.
"""

import logging

import matplotlib.pyplot as plt
import torch

# Optional networkx import for network utilities
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


def visualize_graph(G, mode="info"):
    """
    Visualize a graph structure (e.g., dendritic tree).

    Args:
        G: NetworkX graph object
        mode: Visualization mode ("info", "simple", etc.)
    """
    if not NETWORKX_AVAILABLE:
        logging.warning("NetworkX is not available. Cannot visualize dendritic tree.")
        return

    if G is None:
        logging.warning("No graph provided for visualization.")
        return

    plt.figure(figsize=(12, 8))

    # Create layout
    pos = nx.spring_layout(G, k=1, iterations=50)

    # Draw different node types with different colors
    layer_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "layer"]
    branch_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "branch"]

    # Draw nodes
    if layer_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=layer_nodes,
            node_color="lightblue",
            node_size=1000,
            label="Layers",
        )

    if branch_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=branch_nodes,
            node_color="lightcoral",
            node_size=500,
            label="Branches",
        )

    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.6)

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title("Dendritic Tree Structure")
    plt.legend()
    plt.axis("off")
    plt.tight_layout()


# ============================================================================
# Data Extraction Functions (moved from analysis.py)
# ============================================================================


def dendritic_activations_todict(
    dendrinet,
    exc_input_list,
    inh_input_list,
):
    """
    Extract DendriNet activations to dictionary format.

    Args:
        dendrinet: DendriNet model
        exc_input_list: list of excitatory inputs
        inh_input_list: list of inhibitory inputs

    Returns:
        Dictionary containing activations organized by soma and branch
    """
    layer_sizes = dendrinet.layer_sizes
    n_soma = dendrinet.n_soma
    n_branch_layers = dendrinet.n_branch_layers

    plot_dict = {}
    for n in range(n_soma):
        plot_dict[f"soma{n}"] = {}

    output_list = [None] * len(exc_input_list)
    for i in range(n_branch_layers + 1):
        branches_per_soma = int(layer_sizes[i] / n_soma)

        chunk_list = []

        for j in range(len(exc_input_list)):
            output_list[j] = dendrinet.branch_layers[i](
                exc_input_list[j], inh_input_list[j], output_list[j]
            )
            chunk_list.append(output_list[j].chunk(n_soma, dim=-1))

        level = (
            "soma layer"
            if (i == n_branch_layers)
            else f"branch layer {n_branch_layers - i}"
        )

        for n in range(n_soma):
            plot_dict[f"soma{n}"][level] = {}
            for b in range(branches_per_soma):
                activation_list = []
                for k in range(len(exc_input_list)):
                    chunk = chunk_list[k][n][:, b]
                    activation_list.append(chunk)

                plot_dict[f"soma{n}"][level][f"branch{b}"] = {}
                plot_dict[f"soma{n}"][level][f"branch{b}"][
                    "activation list"
                ] = activation_list

    return plot_dict


def einet_activations_todict(einet, input_list):
    """
    Extract EINet activations to dictionary format.

    Args:
        einet: EINet model
        input_list: list of input tensors

    Returns:
        Dictionary containing activations organized by layer and cell type
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    einet = einet.to(device)

    exc_input_list = [inputs.to(device) for inputs in input_list]
    inh_input_list = [None] * len(input_list)

    plot_dict = {}

    for i in range(len(einet.layers)):
        inh_dendrinet = einet.layers[i].inhibitory_cells
        exc_dendrinet = einet.layers[i].excitatory_cells

        plot_dict[f"eilayer{i+1}"] = {}

        plot_dict[f"eilayer{i+1}"]["inh cells"] = dendritic_activations_todict(
            dendrinet=inh_dendrinet,
            exc_input_list=exc_input_list,
            inh_input_list=inh_input_list,
        )

        inh_temp = []
        for j in range(len(input_list)):
            inh_temp.append(inh_dendrinet(exc_input_list[j], inh_input_list[j]))
        inh_input_list = inh_temp

        plot_dict[f"eilayer{i+1}"]["exc cells"] = dendritic_activations_todict(
            dendrinet=exc_dendrinet,
            exc_input_list=exc_input_list,
            inh_input_list=inh_input_list,
        )

        exc_temp = []
        for j in range(len(input_list)):
            exc_temp.append(exc_dendrinet(exc_input_list[j], inh_input_list[j]))
        exc_input_list = exc_temp

    return plot_dict
