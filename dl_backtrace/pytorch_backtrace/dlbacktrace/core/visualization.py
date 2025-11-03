# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/visualization.py

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import graphviz
from networkx.drawing.nx_pydot import graphviz_layout
from collections import defaultdict
from IPython.display import display, SVG, Image as IPyImage


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
                        show=True, inline_format="svg"):
    """🎯 Visualize relevance backtrace using Graphviz (shows inline + saves)"""
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
            except Exception:
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
    elif relevance_threshold is not None:
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

    g = graphviz.Digraph(
        "DLBacktrace",
        format="svg",
        graph_attr={"rankdir": "LR", "splines": "spline"},
        node_attr={"fontname": "Helvetica", "fontsize": "10"}
    )

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
    show=True,
    inline_format="svg",
):
    def _norm(s):
        return s.replace("/", " ").replace(":", " ")

    # present nodes
    present_raw = list(graph.nodes.keys())
    norm_by_raw = {raw: _norm(raw) for raw in present_raw}
    present_norm = set(norm_by_raw.values())

    # relevance only for present
    rel_map = {}
    for k, v in all_wt.items():
        nk = _norm(k)
        if nk not in present_norm:
            continue
        if isinstance(v, (list, tuple)):
            flat = [float(t.sum()) for t in v if hasattr(t, "sum")]
            if flat:
                mean = float(sum(flat) / len(flat))
                rel_map[nk] = (mean, max(flat), min(flat))
            else:
                rel_map[nk] = (0.0, 0.0, 0.0)
        elif hasattr(v, "sum"):
            rel_map[nk] = (float(v.mean()), float(v.max()), float(v.min()))
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
        graph_attr["rankdir"] = "LR"
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
        lt = graph.nodes[raw].get("layer_type", "Unknown")
        fill = color_map.get(lt, "white")
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

    # edges
    added = set()
    for raw in present_raw:
        child = norm_by_raw[raw]
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
            if e in added:
                continue
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
    show=True,
    inline_format="svg",
):
    """Auto-choose pretty vs fast; always show inline and save."""
    num_nodes = len(graph.nodes)
    print(f"num_nodes: {num_nodes}")

    if num_nodes < node_threshold:
        # small graph → original pretty version
        visualize_relevance(
            graph,
            all_wt,
            output_path=output_path,
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
            show=show,
            inline_format=inline_format,
        )
