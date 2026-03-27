# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/visualization.py

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import graphviz
from networkx.drawing.nx_pydot import graphviz_layout
from collections import defaultdict
from IPython.display import display, SVG, Image as IPyImage
from typing import Optional, Sequence

# Semantically meaningful layer categories for compact visualization
# These match the ATEN_LAYER_MAP categories in graph_builder.py
SEMANTIC_LAYER_TYPES: tuple[str, ...] = (
    "MLP_Layer",      # Linear/FC layers (linear, addmm)
    "DL_Layer",       # Conv layers (conv2d, max_pool2d, etc.)
    "Activation",     # ReLU, GELU, SiLU, etc.
    "Normalization",  # BatchNorm, LayerNorm, GroupNorm
    "Attention",      # Self/Cross attention (scaled_dot_product_attention)
    "Output",         # Final output node (layer_type)
    "Placeholder",    # Input nodes (layer_type) - CRITICAL for graph connectivity
    "Model_Input",    # Legacy input type (layer_type)
    "NLP_Embedding",  # Embedding layers (embedding, embedding_bag)
)

# Default types to always force-include (for graph connectivity)
# Note: Placeholder excluded to keep compact graphs clean
DEFAULT_FORCE_INCLUDE_TYPES: tuple[str, ...] = ("Model_Input", "Output")

# Prefixes for parameter/bias nodes to exclude from compact graphs
EXCLUDED_NODE_PREFIXES: tuple[str, ...] = ("p_model_", "b_model_")


def _is_excluded_node(node_name: str) -> bool:
    """Check if node should be excluded (parameter/bias weights)."""
    return any(node_name.startswith(prefix) for prefix in EXCLUDED_NODE_PREFIXES)


def _get_node_category(node_attrs: dict) -> str:
    """Get the semantic category for a node by checking both layer_type and layer_name.
    
    The graph stores:
    - layer_type: 'ATen_Operation', 'Placeholder', 'Output', 'Operation', etc.
    - layer_name: 'DL_Layer', 'MLP_Layer', 'Activation', 'Normalization', etc.
    
    For filtering, we need to check layer_name first (for ATen ops), then layer_type.
    """
    layer_name = node_attrs.get("layer_name", "")
    layer_type = node_attrs.get("layer_type", "Unknown")
    
    # For ATen operations, layer_name contains the semantic category
    if layer_name in SEMANTIC_LAYER_TYPES:
        return layer_name
    
    # For Output, Model_Input, etc., layer_type is the category
    # Note: Placeholder excluded to keep compact graphs clean
    if layer_type in ("Output", "Model_Input"):
        return layer_type
    
    # Return layer_type as fallback
    return layer_type


def visualize_graph(graph, save_path="graph.png", *, show=True, dpi=600):
    """📊 Visualize forward execution graph with dynamic scaling (shows inline + saves)"""
    num_nodes = len(graph.nodes)

    # -- Dynamic sizing for large graphs --
    fig_width = max(40, max(12, num_nodes // 10))
    fig_height = max(30, max(10, num_nodes // 15))
    node_size = min(50, 5000 // (num_nodes + 1))
    font_size = max(3, 20 - (num_nodes // 50))
    edge_width = max(0.2, 3 - (num_nodes / 200))
    arrow_size = max(3, 15 - (num_nodes // 100))

    plt.figure(figsize=(fig_width, fig_height))

    try:
        pos = graphviz_layout(graph, prog='dot')
    except Exception:
        pos = nx.spring_layout(graph, k=5 / (num_nodes ** 0.5))

    nx.draw(
        graph, pos, with_labels=True,
        node_size=node_size, edgecolors="black", node_color='lightblue',
        font_size=font_size, arrowsize=arrow_size, width=edge_width, arrowstyle='-|>'
    )

    plt.title(f"Graph Visualization ({num_nodes} nodes)", fontsize=16)
    plt.savefig(save_path, format="png", dpi=dpi, bbox_inches="tight")

    # --- show inline in Colab/Jupyter ---
    if show:
        plt.show()

    plt.close()
    print(f"Graph saved as {save_path} ✅")


def visualize_relevance(graph, all_wt, output_path="backtrace_graph",
                        *, top_k=None, relevance_threshold=None,
                        layer_types: Optional[Sequence[str]] = None,
                        rankdir: str = "LR",
                        show=True, inline_format="svg"):
    """🎯 Visualize relevance backtrace using Graphviz (shows inline + saves)
    
    Parameters
    ----------
    graph : networkx.DiGraph
        The computation graph with layer_type attributes on nodes
    all_wt : dict
        Relevance weights for each node
    output_path : str
        Output file path (without extension)
    top_k : int, optional
        Show only top-k nodes by relevance
    relevance_threshold : float, optional
        Show nodes with |relevance| >= threshold
    layer_types : list[str], optional
        Filter to only these layer types. If None, shows all nodes.
        Use SEMANTIC_LAYER_TYPES for a compact paper-ready graph.
    rankdir : str
        Graph direction: "LR" (left-to-right, wide), "TB" (top-to-bottom, tall),
        "RL" (right-to-left), "BT" (bottom-to-top). Default "LR".
        Use "TB" for LaTeX/paper-friendly vertical layout.
    show : bool
        Whether to display inline in Jupyter/Colab
    inline_format : str
        Format for inline display ("svg" or "png")
    """
    relevance_data = {}

    # --- Extract relevance stats from all_wt ---
    # Mean: sum of all entries (for batch=1) or average of sums across batches
    # Max/Min: max/min of batch sums (for batched) or max/min element (for single)
    for node_name, rel in all_wt.items():
        node_key = node_name.replace("/", " ").replace(":", " ")
        if isinstance(rel, (list, tuple)):
            # Batched data: list of tensors
            batch_sums = [float(r.sum()) for r in rel if hasattr(r, "sum")]
            if batch_sums:
                mean_val = sum(batch_sums) / len(batch_sums)
                stats = (mean_val, max(batch_sums), min(batch_sums))
            else:
                stats = (0.0, 0.0, 0.0)
        elif hasattr(rel, "sum"):
            # Single tensor (batch size = 1)
            # Mean = sum of all entries in relevance vector
            stats = (float(rel.sum()), float(rel.max()), float(rel.min()))
        else:
            try:
                val = float(rel)
                stats = (val, val, val)
            except Exception:
                stats = (0.0, 0.0, 0.0)
        relevance_data[node_key] = stats

    # --- Filter based on top_k, threshold, or layer_types ---
    flat_scores = {k: v[0] for k, v in relevance_data.items()}
    total_nodes = len(graph.nodes)

    force_include = {
        node.replace("/", " ").replace(":", " ")
        for node in graph.nodes
        if _get_node_category(graph.nodes[node]) in DEFAULT_FORCE_INCLUDE_TYPES
        and not _is_excluded_node(node)
    }

    if layer_types is not None:
        # Layer-type filtering mode - use _get_node_category for proper semantic matching
        layer_types_set = set(layer_types)
        top_node_names = {
            node.replace("/", " ").replace(":", " ")
            for node in graph.nodes
            if _get_node_category(graph.nodes[node]) in layer_types_set
            and not _is_excluded_node(node)
        } | force_include
        print(f"📊 Layer-type filtering: {total_nodes} nodes → {len(top_node_names)} nodes (filter: {list(layer_types_set)[:5]}{'...' if len(layer_types_set) > 5 else ''})")
    elif top_k:
        top_keys = sorted(flat_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        top_node_names = {k for k, _ in top_keys if not _is_excluded_node(k)} | force_include
        print(f"📊 Top-k filtering: {total_nodes} nodes → {len(top_node_names)} nodes (top_k={top_k})")
    elif relevance_threshold is not None:
        top_node_names = {k for k, v in flat_scores.items() if abs(v) >= relevance_threshold and not _is_excluded_node(k)} | force_include
        print(f"📊 Threshold filtering: {total_nodes} nodes → {len(top_node_names)} nodes (threshold={relevance_threshold})")
    else:
        top_node_names = {k for k in relevance_data.keys() if not _is_excluded_node(k)} | force_include
        print(f"📊 No filtering: {total_nodes} nodes")

    # --- Build raw->normalized name mapping for ancestor lookup ---
    raw_to_norm = {node: node.replace("/", " ").replace(":", " ") for node in graph.nodes}
    norm_to_raw = {v: k for k, v in raw_to_norm.items()}
    
    # --- Helper to find transitive ancestors in filtered set ---
    def find_filtered_ancestors(node_raw, visited=None):
        """BFS to find all ancestors that are in the filtered set."""
        if visited is None:
            visited = set()
        ancestors = set()
        parents = graph.nodes[node_raw].get("parents", [])
        for parent_raw in parents:
            if parent_raw in visited:
                continue
            visited.add(parent_raw)
            parent_norm = raw_to_norm.get(parent_raw, parent_raw.replace("/", " ").replace(":", " "))
            if parent_norm in top_node_names:
                ancestors.add(parent_norm)
            elif parent_raw in graph.nodes:
                # Recursively search this parent's ancestors
                ancestors.update(find_filtered_ancestors(parent_raw, visited))
        return ancestors

    # --- Color map for node types (uses layer_name for ATen ops, layer_type for others) ---
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

    g = graphviz.Digraph(
        "DLBacktrace",
        format="svg",
        graph_attr={"rankdir": rankdir, "splines": "spline"},
        node_attr={"fontname": "Helvetica", "fontsize": "10"}
    )

    # --- Add nodes with relevance ---
    for node in graph.nodes:
        name = node.replace("/", " ").replace(":", " ")
        if name not in top_node_names:
            continue
        rel = relevance_data.get(name, (0.0, 0.0, 0.0))
        # Use layer_name first (for semantic category), then layer_type as fallback
        node_category = _get_node_category(graph.nodes[node])
        fill = color_map.get(node_category, color_map.get(graph.nodes[node].get("layer_type", "Unknown"), "white"))
        g.node(
            name,
            label=f"{name}\nMean: {rel[0]:.3f}\nMax: {rel[1]:.3f}\nMin: {rel[2]:.3f}",
            style="filled",
            fillcolor=fill,
        )

    # --- Add edges (with transitive connections when layer_types filtering) ---
    added_edges = set()
    for node in graph.nodes:
        name = node.replace("/", " ").replace(":", " ")
        if name not in top_node_names:
            continue
        
        if layer_types is not None:
            # Use transitive ancestor search for filtered graphs
            ancestors = find_filtered_ancestors(node)
            for ancestor in ancestors:
                edge = (ancestor, name)
                if edge not in added_edges:
                    added_edges.add(edge)
                    g.edge(ancestor, name)
        else:
            # Original direct parent logic
            for parent in graph.nodes[node].get("parents", []):
                parent_fmt = parent.replace("/", " ").replace(":", " ")
                if parent_fmt in top_node_names:
                    edge = (parent_fmt, name)
                    if edge not in added_edges:
                        added_edges.add(edge)
                        g.edge(parent_fmt, name)

    out = g.render(output_path, format="svg", cleanup=True)

    # --- ALSO show inline in Colab/Jupyter ---
    if show:
        if inline_format.lower() == "svg":
            svg_bytes = g.pipe(format="svg")
            display(SVG(svg_bytes))
        else:
            png_bytes = g.pipe(format="png")
            display(IPyImage(data=png_bytes))

    print(f"📊 DLBacktrace Graph saved at → {output_path}.svg")
    return g, out


# ─────────────────────────────────────────
# helper: collapse 1-parent 1-child nodes
# ─────────────────────────────────────────
class SimpleGraph:
    def __init__(self, nodes_dict):
        self.nodes = nodes_dict


def simplify_graph_by_collapsing_degree2(
    graph,
    *,
    protect_types=("Placeholder", "Model_Input", "Output", "Attention"),
    max_passes=10,
):
    nodes_attr = {n: dict(graph.nodes[n]) for n in graph.nodes}
    parents = {n: list(nodes_attr[n].get("parents", [])) for n in nodes_attr}

    def build_children(ps):
        ch = defaultdict(list)
        for child, ps_ in ps.items():
            for p in ps_:
                ch[p].append(child)
        return ch

    children = build_children(parents)
    collapsed_into = defaultdict(set)

    def is_protected(n):
        lt = nodes_attr.get(n, {}).get("layer_type", "Unknown")
        return lt in protect_types

    changed = True
    passes = 0
    while changed and passes < max_passes:
        changed = False
        passes += 1

        to_collapse = []
        for n in list(nodes_attr.keys()):
            if n not in nodes_attr:
                continue
            if is_protected(n):
                continue
            ps = parents.get(n, [])
            cs = children.get(n, [])
            if len(ps) == 1 and len(cs) == 1:
                p, c = ps[0], cs[0]
                if p != c and p in nodes_attr and c in nodes_attr:
                    to_collapse.append((n, p, c))

        if not to_collapse:
            break

        for n, p, c in to_collapse:
            if n not in nodes_attr or p not in nodes_attr or c not in nodes_attr:
                continue

            # rewire child
            if n in parents.get(c, []):
                parents[c].remove(n)
            if p not in parents[c]:
                parents[c].append(p)

            # rewire parent
            if n in children.get(p, []):
                children[p].remove(n)
            if c not in children.get(p, []):
                children[p].append(c)

            # remove n everywhere
            for gp in parents.get(n, []):
                if n in children.get(gp, []):
                    children[gp].remove(n)
            for gc in children.get(n, []):
                if n in parents.get(gc, []):
                    parents[gc].remove(n)

            parents.pop(n, None)
            children.pop(n, None)

            collapsed_into[c].add(n)
            nodes_attr.pop(n, None)

            changed = True

        children = build_children(parents)

    simplified_nodes = {}
    for n in nodes_attr:
        simplified_nodes[n] = {
            "parents": list(parents.get(n, [])),
            "layer_type": nodes_attr[n].get("layer_type", "Unknown"),
            "collapsed_count": len(collapsed_into.get(n, set())),
        }

    return SimpleGraph(simplified_nodes), collapsed_into


def visualize_relevance_fast(
    graph,
    all_wt,
    output_path="backtrace_collapsed_fast",
    *,
    collapsed_map=None,
    max_parents_per_node=None,
    engine_auto_threshold=1200,
    disable_concentrate_for_sfdp=True,
    layer_types: Optional[Sequence[str]] = None,
    rankdir: str = "LR",
    show=True,
    inline_format="svg",
):
    """Fast visualization for large/collapsed graphs.
    
    Parameters
    ----------
    layer_types : list[str], optional
        Filter to only these layer types. If None, shows all nodes.
    rankdir : str
        Graph direction: "LR" (left-to-right), "TB" (top-to-bottom).
        Default "LR". Use "TB" for LaTeX-friendly vertical layout.
    """
    def _norm(s):
        return s.replace("/", " ").replace(":", " ")

    # present nodes - keep all for transitive edge computation
    all_raw = list(graph.nodes.keys())
    total_nodes = len(all_raw)
    
    # Determine filtered set using _get_node_category for proper semantic matching
    if layer_types is not None:
        layer_types_set = set(layer_types) | set(DEFAULT_FORCE_INCLUDE_TYPES)
        present_raw = [
            raw for raw in all_raw
            if _get_node_category(graph.nodes[raw]) in layer_types_set
            and not _is_excluded_node(raw)
        ]
        print(f"📊 Layer-type filtering (fast): {total_nodes} nodes → {len(present_raw)} nodes (filter: {list(layer_types)[:5]}{'...' if len(layer_types) > 5 else ''})")
    else:
        present_raw = [raw for raw in all_raw if not _is_excluded_node(raw)]
        print(f"📊 No filtering (fast): {total_nodes} nodes → {len(present_raw)} nodes (excluded p_model_*/b_model_*)")
    
    norm_by_raw = {raw: _norm(raw) for raw in all_raw}  # All nodes for lookup
    present_norm = {_norm(raw) for raw in present_raw}  # Filtered set
    
    # Helper to find transitive ancestors in filtered set
    def find_filtered_ancestors(node_raw, visited=None):
        """BFS to find all ancestors that are in the filtered set."""
        if visited is None:
            visited = set()
        ancestors = set()
        parents = graph.nodes[node_raw].get("parents", []) or []
        for parent_raw in parents:
            if parent_raw in visited:
                continue
            visited.add(parent_raw)
            parent_norm = norm_by_raw.get(parent_raw, _norm(parent_raw))
            if parent_norm in present_norm:
                ancestors.add(parent_norm)
            elif parent_raw in graph.nodes:
                # Recursively search this parent's ancestors
                ancestors.update(find_filtered_ancestors(parent_raw, visited))
        return ancestors

    # relevance only for present
    # Mean: sum of all entries (for batch=1) or average of sums across batches
    # Max: maximum individual value in the relevance tensor
    # Min: minimum individual value in the relevance tensor
    rel_map = {}
    for k, v in all_wt.items():
        nk = _norm(k)
        if nk not in present_norm:
            continue
        if isinstance(v, (list, tuple)):
            # Batched data
            batch_sums = [float(t.sum()) for t in v if hasattr(t, "sum")]
            if batch_sums:
                mean_val = sum(batch_sums) / len(batch_sums)
                rel_map[nk] = (mean_val, max(batch_sums), min(batch_sums))
            else:
                rel_map[nk] = (0.0, 0.0, 0.0)
        elif hasattr(v, "sum"):
            # Single tensor (batch size = 1): Mean = sum of all entries
            rel_map[nk] = (float(v.sum()), float(v.max()), float(v.min()))
        else:
            try:
                x = float(v)
                rel_map[nk] = (x, x, x)
            except Exception:
                rel_map[nk] = (0.0, 0.0, 0.0)

    # defaults
    for nk in present_norm:
        rel_map.setdefault(nk, (0.0, 0.0, 0.0))

    # aggregate collapsed
    if collapsed_map:
        for kept_raw, removed_raws in collapsed_map.items():
            kept_norm = _norm(kept_raw)
            km, kx, kn = rel_map.get(kept_norm, (0.0, 0.0, 0.0))
            agg_m, agg_x, agg_n = km, kx, kn
            for rm_raw in removed_raws:
                rm_norm = _norm(rm_raw)
                m, x, n = rel_map.get(rm_norm, (0.0, 0.0, 0.0))
                agg_m += m
                agg_x = max(agg_x, x)
                agg_n = min(agg_n, n)
            rel_map[kept_norm] = (agg_m, agg_x, agg_n)

    num_nodes = len(present_raw)
    engine = "dot" if num_nodes < engine_auto_threshold else "sfdp"

    graph_attr = {
        "overlap": "false",
        "nodesep": "0.25",
        "ranksep": "0.35",
        "ratio": "compress",
        "margin": "0.05",
        "outputorder": "edgesfirst",
    }
    if engine == "dot":
        graph_attr["rankdir"] = rankdir
        graph_attr["splines"] = "spline"
        graph_attr["concentrate"] = "true"
    else:
        if not disable_concentrate_for_sfdp:
            graph_attr["concentrate"] = "true"

    g = graphviz.Digraph(
        "DLBacktraceFast",
        format="svg",
        engine=engine,
        graph_attr=graph_attr,
        node_attr={"fontname": "Helvetica", "fontsize": "9"},
        edge_attr={"arrowsize": "0.5", "penwidth": "0.7"},
    )

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
        "Model_Input": "lightcyan",
    }

    def _short(s, n=48):
        return s if len(s) <= n else s[:n-1] + "…"

    # nodes
    for raw in present_raw:
        nk = norm_by_raw[raw]
        mean, mx, mn = rel_map.get(nk, (0.0, 0.0, 0.0))
        # Use _get_node_category for proper semantic coloring
        node_category = _get_node_category(graph.nodes[raw])
        fill = color_map.get(node_category, color_map.get(graph.nodes[raw].get("layer_type", "Unknown"), "white"))
        collapsed = graph.nodes[raw].get("collapsed_count", 0)
        collapsed_line = f"\n[collapsed {collapsed}]" if collapsed else ""

        g.node(
            nk,
            label=(
                f"{_short(nk)}\n"
                f"Mean: {mean:.3f}\n"
                f"Max: {mx:.3f}\n"
                f"Min: {mn:.3f}"
                f"{collapsed_line}"
            ),
            style="filled",
            fillcolor=fill,
        )

    # edges - use transitive ancestors when layer_types filtering
    added = set()
    for raw in present_raw:
        child = norm_by_raw[raw]
        
        if layer_types is not None:
            # Use transitive ancestor search for filtered graphs
            ancestors = find_filtered_ancestors(raw)
            for ancestor in ancestors:
                e = (ancestor, child)
                if e not in added:
                    added.add(e)
                    g.edge(ancestor, child)
        else:
            # Original direct parent logic
            parents = graph.nodes[raw].get("parents", []) or []
            if max_parents_per_node is not None and len(parents) > max_parents_per_node:
                parents = sorted(
                    parents,
                    key=lambda p: abs(rel_map.get(_norm(p), (0.0, 0.0, 0.0))[0]),
                    reverse=True,
                )[:max_parents_per_node]

            for p_raw in parents:
                pn = norm_by_raw.get(p_raw, _norm(p_raw))
                e = (pn, child)
                if e not in added:
                    added.add(e)
                    g.edge(pn, child)

    out = g.render(output_path, cleanup=True)

    # --- Also show inline in Colab/Jupyter ---
    if show:
        if inline_format.lower() == "svg":
            svg_bytes = g.pipe(format="svg")
            display(SVG(svg_bytes))
        else:
            png_bytes = g.pipe(format="png")
            display(IPyImage(data=png_bytes))

    print(f"✅ Fast graph saved → {out} (nodes={num_nodes}, engine={engine})")
    return g, out


def visualize_relevance_auto(
    graph,
    all_wt,
    output_path="backtrace_graph",
    *,
    node_threshold=500,
    engine_auto_threshold=1500,
    fast_output_path="backtrace_collapsed_fast",
    layer_types: Optional[Sequence[str]] = None,
    rankdir: str = "LR",
    show=True,
    inline_format="svg",
):
    """Auto-choose pretty vs fast visualization; always show inline and save.
    
    Parameters
    ----------
    layer_types : list[str], optional
        Filter to only these layer types. If specified, layer_types filtering
        takes precedence over automatic collapsing for large graphs.
        Use SEMANTIC_LAYER_TYPES for a compact paper-ready graph.
    rankdir : str
        Graph direction: "LR" (left-to-right, wide), "TB" (top-to-bottom, tall).
        Default "LR". Use "TB" for LaTeX/paper-friendly vertical layout.
    """
    # If layer_types specified, count only matching nodes for threshold decision
    if layer_types is not None:
        layer_types_set = set(layer_types) | set(DEFAULT_FORCE_INCLUDE_TYPES)
        filtered_count = sum(
            1 for n in graph.nodes 
            if _get_node_category(graph.nodes[n]) in layer_types_set
            and not _is_excluded_node(n)
        )
        num_nodes = filtered_count
        print(f"num_nodes after layer_types filter: {num_nodes} (from {len(graph.nodes)} total)")
    else:
        num_nodes = sum(1 for n in graph.nodes if not _is_excluded_node(n))
        print(f"num_nodes: {num_nodes} (from {len(graph.nodes)} total, excluded p_model_*/b_model_*)")

    if num_nodes < node_threshold:
        # small graph → original pretty version
        visualize_relevance(
            graph,
            all_wt,
            output_path=output_path,
            layer_types=layer_types,
            rankdir=rankdir,
            show=show,
            inline_format=inline_format,
        )
    else:
        # big graph → collapse then fast
        print(f"big graph → collapsing it ...")
        simp_graph, collapsed_map = simplify_graph_by_collapsing_degree2(
            graph,
            protect_types=("Placeholder", "Model_Input", "Output", "Attention"),
        )
        print(f"Calculate relevance using `visualize_relevance_fast(...)`")
        visualize_relevance_fast(
            simp_graph,
            all_wt,
            output_path=fast_output_path,
            collapsed_map=collapsed_map,
            max_parents_per_node=2,
            engine_auto_threshold=engine_auto_threshold,
            layer_types=layer_types,
            rankdir=rankdir,
            show=show,
            inline_format=inline_format,
        )


def visualize_relevance_paginated(
    graph,
    all_wt,
    output_path="backtrace_graph",
    *,
    max_nodes_per_page: int = 30,
    layer_types: Optional[Sequence[str]] = None,
    show=True,
    inline_format="svg",
):
    """🎯 Visualize relevance backtrace as multiple sub-graphs (pages) for long DAGs.
    
    Splits the graph into topologically-ordered pages, each containing at most
    `max_nodes_per_page` nodes. Each page is saved as a separate file and
    displayed inline sequentially.
    
    Parameters
    ----------
    graph : networkx.DiGraph
        The computation graph with layer_type attributes on nodes
    all_wt : dict
        Relevance weights for each node
    output_path : str
        Base output file path (without extension). Pages will be named
        {output_path}_page1.svg, {output_path}_page2.svg, etc.
    max_nodes_per_page : int
        Maximum number of nodes per page/sub-graph (default: 30)
    layer_types : list[str], optional
        Filter to only these layer types. Use SEMANTIC_LAYER_TYPES for compact graphs.
    show : bool
        Whether to display inline in Jupyter/Colab
    inline_format : str
        Format for inline display ("svg" or "png")
    
    Returns
    -------
    list[tuple[graphviz.Digraph, str]]
        List of (graph, output_path) tuples for each page
    """
    # --- Extract relevance stats ---
    def _norm(s):
        return s.replace("/", " ").replace(":", " ")
    
    relevance_data = {}
    for node_name, rel in all_wt.items():
        node_key = _norm(node_name)
        if isinstance(rel, (list, tuple)):
            batch_sums = [float(r.sum()) for r in rel if hasattr(r, "sum")]
            if batch_sums:
                mean_val = sum(batch_sums) / len(batch_sums)
                relevance_data[node_key] = (mean_val, max(batch_sums), min(batch_sums))
            else:
                relevance_data[node_key] = (0.0, 0.0, 0.0)
        elif hasattr(rel, "sum"):
            relevance_data[node_key] = (float(rel.sum()), float(rel.max()), float(rel.min()))
        else:
            try:
                val = float(rel)
                relevance_data[node_key] = (val, val, val)
            except Exception:
                relevance_data[node_key] = (0.0, 0.0, 0.0)
    
    # --- Filter nodes ---
    total_nodes = len(graph.nodes)
    
    if layer_types is not None:
        layer_types_set = set(layer_types) | set(DEFAULT_FORCE_INCLUDE_TYPES)
        filtered_nodes = [
            node for node in graph.nodes
            if _get_node_category(graph.nodes[node]) in layer_types_set
            and not _is_excluded_node(node)
        ]
    else:
        filtered_nodes = [node for node in graph.nodes if not _is_excluded_node(node)]
    
    filtered_norm = {_norm(n) for n in filtered_nodes}
    print(f"📊 Paginated: {total_nodes} nodes → {len(filtered_nodes)} nodes")
    
    # --- Build parent mapping for filtered nodes ---
    raw_to_norm = {node: _norm(node) for node in graph.nodes}
    
    def find_filtered_ancestors(node_raw, visited=None):
        """BFS to find ancestors in filtered set (for transitive edges)."""
        if visited is None:
            visited = set()
        ancestors = set()
        parents = graph.nodes[node_raw].get("parents", []) or []
        for parent_raw in parents:
            if parent_raw in visited:
                continue
            visited.add(parent_raw)
            parent_norm = raw_to_norm.get(parent_raw, _norm(parent_raw))
            if parent_norm in filtered_norm:
                ancestors.add(parent_norm)
            elif parent_raw in graph.nodes:
                ancestors.update(find_filtered_ancestors(parent_raw, visited))
        return ancestors
    
    # --- Topological sort of filtered nodes ---
    # Build edges for filtered subgraph
    filtered_edges = {}  # node_norm -> set of parent_norms (in filtered set)
    for node in filtered_nodes:
        node_norm = _norm(node)
        if layer_types is not None:
            filtered_edges[node_norm] = find_filtered_ancestors(node)
        else:
            parents = graph.nodes[node].get("parents", []) or []
            filtered_edges[node_norm] = {_norm(p) for p in parents if _norm(p) in filtered_norm}
    
    # Kahn's algorithm for topological sort
    in_degree = {n: 0 for n in filtered_norm}
    for node, parents in filtered_edges.items():
        for p in parents:
            if p in in_degree:
                in_degree[node] = in_degree.get(node, 0) + 1
    
    # Start with nodes that have no filtered parents (in_degree == 0)
    queue = [n for n, d in in_degree.items() if d == 0]
    topo_order = []
    
    while queue:
        node = queue.pop(0)
        topo_order.append(node)
        # Find children of this node
        for child, parents in filtered_edges.items():
            if node in parents:
                in_degree[child] -= 1
                if in_degree[child] == 0 and child not in topo_order:
                    queue.append(child)
    
    # Add any remaining nodes (handles cycles gracefully)
    for n in filtered_norm:
        if n not in topo_order:
            topo_order.append(n)
    
    print(f"📊 Topological order: {len(topo_order)} nodes")
    
    # --- Split into pages ---
    pages = []
    for i in range(0, len(topo_order), max_nodes_per_page):
        pages.append(topo_order[i:i + max_nodes_per_page])
    
    print(f"📊 Split into {len(pages)} pages (max {max_nodes_per_page} nodes/page)")
    
    # --- Color scale setup ---
    all_means = [relevance_data.get(n, (0.0, 0.0, 0.0))[0] for n in topo_order]
    max_abs = max(abs(v) for v in all_means) if all_means else 1.0
    if max_abs == 0:
        max_abs = 1.0
    
    def get_color(mean_val):
        norm = mean_val / max_abs
        if norm >= 0:
            r = int(255 * (1 - norm))
            return f"#{r:02x}ff{r:02x}"  # green
        else:
            g = int(255 * (1 + norm))
            return f"#ff{g:02x}{g:02x}"  # red
    
    # --- Render each page ---
    results = []
    for page_idx, page_nodes in enumerate(pages):
        page_num = page_idx + 1
        page_node_set = set(page_nodes)
        
        # Include connector nodes from previous page for context
        connector_nodes = set()
        if page_idx > 0:
            prev_page_nodes = set(pages[page_idx - 1])
            for node in page_nodes:
                for parent in filtered_edges.get(node, set()):
                    if parent in prev_page_nodes:
                        connector_nodes.add(parent)
        
        g = graphviz.Digraph(
            name=f"DLBacktrace_Page{page_num}",
            format="svg",
            graph_attr={
                "rankdir": "TB",  # Top-to-bottom for vertical flow
                "label": f"DLBacktrace Graph - Page {page_num}/{len(pages)}",
                "labelloc": "t",
                "fontsize": "14",
                "nodesep": "0.3",
                "ranksep": "0.5",
            },
            node_attr={
                "shape": "box",
                "style": "filled,rounded",
                "fontsize": "10",
            },
            edge_attr={
                "fontsize": "8",
            },
        )
        
        # Add connector nodes (grayed out, from previous page)
        for node in connector_nodes:
            stats = relevance_data.get(node, (0.0, 0.0, 0.0))
            label = f"{node}\n(from prev page)"
            g.node(node, label=label, fillcolor="lightgray", style="filled,rounded,dashed")
        
        # Add page nodes
        for node in page_nodes:
            stats = relevance_data.get(node, (0.0, 0.0, 0.0))
            mean, mx, mn = stats
            label = f"{node}\nMean={mean:.4f}\nMax={mx:.4f} Min={mn:.4f}"
            color = get_color(mean)
            g.node(node, label=label, fillcolor=color)
        
        # Add edges within page and from connectors
        added_edges = set()
        for node in page_nodes:
            for parent in filtered_edges.get(node, set()):
                if parent in page_node_set or parent in connector_nodes:
                    edge = (parent, node)
                    if edge not in added_edges:
                        added_edges.add(edge)
                        # Get edge weight (child's mean relevance)
                        child_mean = relevance_data.get(node, (0.0, 0.0, 0.0))[0]
                        g.edge(parent, node, label=f"{child_mean:.3f}")
        
        # Render
        page_output = f"{output_path}_page{page_num}"
        out = g.render(page_output, cleanup=True)
        results.append((g, out))
        
        # Show inline
        if show:
            print(f"\n{'='*50}")
            print(f"📄 Page {page_num}/{len(pages)} ({len(page_nodes)} nodes)")
            print(f"{'='*50}")
            if inline_format.lower() == "svg":
                svg_bytes = g.pipe(format="svg")
                display(SVG(svg_bytes))
            else:
                png_bytes = g.pipe(format="png")
                display(IPyImage(data=png_bytes))
        
        print(f"✅ Page {page_num} saved → {out}")
    
    print(f"\n📊 Total: {len(pages)} pages saved with base path '{output_path}_pageN.svg'")
    return results
