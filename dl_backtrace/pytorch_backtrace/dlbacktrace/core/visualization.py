# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/visualization.py

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import graphviz
from networkx.drawing.nx_pydot import graphviz_layout


def visualize_graph(graph, save_path="graph.png"):
    """📊 Visualize forward execution graph with dynamic scaling"""
    num_nodes = len(graph.nodes)

    # -- Dynamic sizing for large graphs --
    fig_width = max(40, max(12, num_nodes // 10))
    fig_height = max(30, max(10, num_nodes // 15))
    node_size = min(50, 5000 // (num_nodes + 1))
    font_size = min(3, 20 - (num_nodes // 50))
    edge_width = min(0.5, 3 - (num_nodes / 200))
    arrow_size = min(3, 15 - (num_nodes // 100))

    plt.figure(figsize=(fig_width, fig_height))

    try:
        pos = graphviz_layout(graph, prog='dot')
    except:
        pos = nx.spring_layout(graph, k=5 / (num_nodes ** 0.5))

    nx.draw(
        graph, pos, with_labels=True,
        node_size=node_size, edgecolors="black", node_color='lightblue',
        font_size=font_size, arrowsize=arrow_size, width=edge_width, arrowstyle='-|>'
    )

    plt.title(f"Graph Visualization ({num_nodes} nodes)", fontsize=16)
    plt.savefig(save_path, format="png", dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Graph saved as {save_path} ✅")


def visualize_relevance(graph, all_wt, output_path="backtrace_graph", top_k=None, relevance_threshold=None):
    """🎯 Visualize relevance backtrace using Graphviz"""
    relevance_data = {}

    # --- Extract relevance stats from all_wt ---
    for node_name, rel in all_wt.items():
        node_key = node_name.replace("/", " ").replace(":", " ")
        if isinstance(rel, (list, tuple)):
            flat = [float(r.sum()) for r in rel if hasattr(r, "sum")]
            stats = (float(sum(flat) / len(flat)), max(flat), min(flat)) if flat else (0.0, 0.0, 0.0)
        elif hasattr(rel, "sum"):
            stats = (float(rel.mean()), float(rel.max()), float(rel.min()))
        else:
            try:
                val = float(rel)
                stats = (val, val, val)
            except:
                stats = (0.0, 0.0, 0.0)
        relevance_data[node_key] = stats

    # --- Filter based on top_k or threshold ---
    flat_scores = {k: v[0] for k, v in relevance_data.items()}

    force_include = {
        node.replace("/", " ").replace(":", " ")
        for node in graph.nodes
        if graph.nodes[node].get("layer_type") in ("Placeholder", "Model_Input")
    }

    if top_k:
        top_keys = sorted(flat_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        top_node_names = {k for k, _ in top_keys} | force_include
    elif relevance_threshold:
        top_node_names = {k for k, v in flat_scores.items() if abs(v) >= relevance_threshold} | force_include
    else:
        top_node_names = set(relevance_data.keys()) | force_include

    # --- Color map for node types ---
    color_map = {
        "MLP_Layer": "lightblue",
        "DL_Layer": "lightgreen",
        "Activation": "orange",
        "Normalization": "pink",
        "Mathematical_Operation": "yellow",
        "Vector_Operation": "gray",
        "Indexing_Operation": "lightgray",
        "ATen_Operation": "violet",
        "NLP_Embedding": "lightsalmon",
        "Attention": "gold",
        "Output": "red",
        "Placeholder": "white",
        "Model_Input": "lightcyan"
    }

    g = graphviz.Digraph("DLBacktrace", format="png", graph_attr={"rankdir": "LR", "splines": "spline"})

    # --- Add nodes with relevance ---
    for node in graph.nodes:
        name = node.replace("/", " ").replace(":", " ")
        if name not in top_node_names:
            continue
        rel = relevance_data.get(name, (0.0, 0.0, 0.0))
        fill = color_map.get(graph.nodes[node].get("layer_type", "Unknown"), "white")
        g.node(
            name,
            label=f"{name}\nMean: {rel[0]:.3f}\nMax: {rel[1]:.3f}\nMin: {rel[2]:.3f}",
            style="filled",
            fillcolor=fill,
            fontname="Helvetica",
            fontsize="10"
        )

    # --- Add edges ---
    for node in graph.nodes:
        name = node.replace("/", " ").replace(":", " ")
        if name not in top_node_names:
            continue
        for parent in graph.nodes[node].get("parents", []):
            parent_fmt = parent.replace("/", " ").replace(":", " ")
            if parent_fmt in top_node_names:
                g.edge(parent_fmt, name)

    g.render(output_path, format="svg", cleanup=True)
    print(f"📊 DLBacktrace Graph saved at → {output_path}.svg")
