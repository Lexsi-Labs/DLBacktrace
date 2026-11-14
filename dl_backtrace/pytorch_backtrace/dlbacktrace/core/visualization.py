# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/visualization.py

import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import graphviz
from networkx.drawing.nx_pydot import graphviz_layout
import re
from collections import defaultdict
from IPython.display import display, SVG, Image as IPyImage
from typing import Optional, Sequence

# Layer categories kept by the compact graph (`compact=True`). They match the categories that
# graph_builder.py stores in each node's `layer_name` (ATen operations) or `layer_type`.
SEMANTIC_LAYER_TYPES: tuple[str, ...] = (
    "MLP_Layer",      # linear, addmm
    "DL_Layer",       # conv, pooling, lstm
    "Activation",     # relu, gelu, silu, softmax, ...
    "Normalization",  # batch_norm, layer_norm, dropout
    "Attention",      # scaled_dot_product_attention
    "Output",
    "Placeholder",    # the model inputs, and the parameters unless show_parameters=False
    "Model_Input",
    "NLP_Embedding",  # embedding, embedding_bag
)

# Always kept when a layer-type filter is used.
DEFAULT_FORCE_INCLUDE_TYPES: tuple[str, ...] = ("Model_Input", "Output")

# Placeholders with these prefixes are parameters (p_), buffers (b_) and lifted constants (c_)
# of the exported program (see trace_utils.extract_placeholders). They carry no relevance;
# `show_parameters=False` leaves them out of the graphs that are drawn.
EXCLUDED_NODE_PREFIXES: tuple[str, ...] = ("p_", "b_", "c_")

_RANKDIRS = ("LR", "TB", "RL", "BT")

COLOR_MAP = {
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


def _fmt_rel(x):
    """Format small relevance values without rounding them to visual zero."""
    try:
        x = float(x)
    except Exception:
        return "0.000"
    if x == 0.0:
        return "0.000"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.3e}"
    return f"{x:.3f}"


def _norm_node_name(s):
    return str(s).replace("/", " ").replace(":", " ")


def _to_float(x, default=0.0):
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "item"):
            x = x.item()
        return float(x)
    except Exception:
        return default


def _check_stat(stat):
    if stat not in ("sum", "mean"):
        raise ValueError("relevance_stat must be 'sum' or 'mean'")
    return stat


def _check_rankdir(rankdir):
    rankdir = str(rankdir).upper()
    if rankdir not in _RANKDIRS:
        raise ValueError(f"rankdir must be one of {_RANKDIRS}")
    return rankdir


def _relevance_stats(rel, stat="sum"):
    """Return (value, max, min, score) for a relevance object.

    `value` is the total relevance of the node (`stat="sum"`, the default) or its mean
    (`stat="mean"`). Relevance is propagated as a total (the output relevance is split among
    the inputs of each layer), so the sum can be compared between nodes; the mean divides it by
    the tensor size, so the same total reads differently on tensors of different sizes. For a
    node that hands relevance to several inputs (a list) the sum is the total over all of them,
    and max / min are over the per-input sums.
    """
    _check_stat(stat)
    if isinstance(rel, (list, tuple)):
        vals = []
        for item in rel:
            if hasattr(item, "sum"):
                vals.append(_to_float(item.sum()))
        if not vals:
            return 0.0, 0.0, 0.0, 0.0
        total = float(sum(vals))
        mean = total / len(vals)
        mx = max(vals)
        mn = min(vals)
        return (total if stat == "sum" else mean), mx, mn, max(abs(mean), abs(mx), abs(mn))

    if hasattr(rel, "sum"):
        mean = _to_float(rel.mean())
        mx = _to_float(rel.max())
        mn = _to_float(rel.min())
        total = _to_float(rel.sum())
        return (total if stat == "sum" else mean), mx, mn, max(abs(mean), abs(mx), abs(mn), abs(total))

    val = _to_float(rel)
    return val, val, val, abs(val)


def _relevance_maps(all_wt, stat="sum"):
    stats = {}
    scores = {}
    for node_name, rel in (all_wt or {}).items():
        node_key = _norm_node_name(node_name)
        value, mx, mn, score = _relevance_stats(rel, stat)
        stats[node_key] = (value, mx, mn)
        scores[node_key] = score
    return stats, scores


def _is_excluded_node(node_name, attrs=None):
    """True for a parameter, buffer or constant placeholder (name only, if no attributes)."""
    if attrs is not None and attrs.get("layer_type", "Placeholder") != "Placeholder":
        return False
    return str(node_name).startswith(EXCLUDED_NODE_PREFIXES)


def _get_node_category(attrs):
    """The semantic category of a node, e.g. "MLP_Layer".

    graph_builder stores layer_type ("ATen_Operation", "Placeholder", "Output", ...) and, for
    ATen operations, the category in layer_name ("MLP_Layer", "Activation", ...). The category
    is layer_name when it is one of the colour-map categories, otherwise layer_type. `layer_types`
    matches either.
    """
    layer_name = attrs.get("layer_name", "")
    if layer_name in COLOR_MAP:
        return layer_name
    return attrs.get("layer_type", "Unknown")


def _node_kinds(attrs):
    """Every name a node can be selected by in `layer_types`: its category and its layer_type."""
    return {_get_node_category(attrs), attrs.get("layer_type", "Unknown"), attrs.get("layer_name", "")}


def _node_fill(attrs):
    return COLOR_MAP.get(_get_node_category(attrs), COLOR_MAP.get(attrs.get("layer_type", "Unknown"), "white"))


def _normalise_layer_types(layer_types):
    if layer_types is None:
        return None
    if isinstance(layer_types, str):
        return [layer_types]
    return list(layer_types)


def _compact_graph(graph, layer_types=None, show_parameters=True):
    """A `networkx.DiGraph` with only the nodes of `layer_types` (plus the output) and, if
    `show_parameters` is False, without parameter placeholders. With neither, the graph itself.

    Every kept node is linked to its nearest kept ancestors, so the nodes that were dropped
    in between (reshapes, casts, ...) do not break the graph into pieces. Node attributes
    kept: layer_type, layer_name, collapsed_count, and `parents` (the new ones).
    """
    layer_types = _normalise_layer_types(layer_types)
    if layer_types is None and show_parameters:
        return graph
    wanted = None if layer_types is None else set(layer_types) | set(DEFAULT_FORCE_INCLUDE_TYPES)
    attrs = {n: graph.nodes[n] for n in graph.nodes}

    def keep(n):
        if not show_parameters and _is_excluded_node(n, attrs[n]):
            return False
        return wanted is None or bool(_node_kinds(attrs[n]) & wanted)

    def parents_of(n):
        ps = attrs[n].get("parents") or []
        if not ps and hasattr(graph, "predecessors"):
            ps = list(graph.predecessors(n))
        return [p for p in ps if p in attrs]

    dag = nx.DiGraph()
    dag.add_nodes_from(attrs)
    dag.add_edges_from((p, n) for n in attrs for p in parents_of(n))

    nearest = {}  # node -> nearest kept ancestors (the node itself when it is kept)
    for n in nx.topological_sort(dag):
        if keep(n):
            nearest[n] = [n]
        else:
            found = []
            for p in parents_of(n):
                for a in nearest[p]:
                    if a not in found:
                        found.append(a)
            nearest[n] = found

    out = nx.DiGraph()
    for n in attrs:
        if not keep(n):
            continue
        parents = []
        for p in parents_of(n):
            for a in nearest[p]:
                if a not in parents:
                    parents.append(a)
        keep_attrs = {k: attrs[n][k] for k in ("layer_type", "layer_name", "collapsed_count") if k in attrs[n]}
        out.add_node(n, parents=parents, **keep_attrs)
    for n in out.nodes:
        out.add_edges_from((p, n) for p in out.nodes[n]["parents"])

    if wanted is not None:
        unknown = sorted(t for t in layer_types if t not in SEMANTIC_LAYER_TYPES
                         and not any(t in _node_kinds(a) for a in attrs.values()))
        msg = f"Layer-type filtering: {len(attrs)} nodes -> {len(out)} nodes"
        print(msg + (f" (no node of type {unknown} in this graph)" if unknown else ""))
    return out


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
                        relevance_stat="sum",
                        show_parameters=True,
                        show=True, inline_format="svg"):
    """🎯 Visualize relevance backtrace using Graphviz (shows inline + saves)

    layer_types : list[str], optional
        Draw only the nodes of these layer types (e.g. "MLP_Layer", "Attention"; see
        `SEMANTIC_LAYER_TYPES`), linked through the nodes that were left out. The output is
        always kept (and a `Model_Input`, if the graph has one); include "Placeholder" to keep
        the model inputs.
    rankdir : {"LR", "TB", "RL", "BT"}
        Direction of the graph. "TB" gives a tall graph that fits a page.
    relevance_stat : {"sum", "mean"}
        What the first line of each label shows: the node's total relevance (default) or its
        mean. See `_relevance_stats`.

    show_parameters : bool
        Draw the parameter, buffer and constant placeholders (`p_*`, `b_*`, `c_*`). They carry
        no relevance; `False` leaves them out. Default True.
    """
    rankdir = _check_rankdir(rankdir)
    graph = _compact_graph(graph, layer_types, show_parameters)
    relevance_data, _ = _relevance_maps(all_wt, relevance_stat)

    # --- Filter based on top_k or threshold ---
    flat_scores = {k: v[0] for k, v in relevance_data.items()}
    drawn = {_norm_node_name(n) for n in graph.nodes}

    force_include = {
        _norm_node_name(node)
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
    top_node_names &= drawn

    g = graphviz.Digraph(
        "DLBacktrace",
        format="svg",
        graph_attr={"rankdir": rankdir, "splines": "spline"},
        node_attr={"fontname": "Helvetica", "fontsize": "10"}
    )

    # --- Add nodes with relevance ---
    stat_name = relevance_stat.capitalize()
    for node in graph.nodes:
        name = _norm_node_name(node)
        if name not in top_node_names:
            continue
        rel = relevance_data.get(name, (0.0, 0.0, 0.0))
        g.node(
            name,
            label=(
                f"{name}\n"
                f"{stat_name}: {_fmt_rel(rel[0])}\n"
                f"Max: {_fmt_rel(rel[1])}\n"
                f"Min: {_fmt_rel(rel[2])}"
            ),
            style="filled",
            fillcolor=_node_fill(graph.nodes[node]),
        )

    # --- Add edges ---
    for node in graph.nodes:
        name = _norm_node_name(node)
        if name not in top_node_names:
            continue
        for parent in graph.nodes[node].get("parents", []):
            parent_fmt = _norm_node_name(parent)
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

        collapsed_this_pass = 0
        while True:
            candidate = None
            children = build_children(parents)
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
                        candidate = (n, p, c)
                        break

            if candidate is None:
                break

            n, p, c = candidate
            if n not in nodes_attr:
                break

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
            collapsed_into[c].update(collapsed_into.pop(n, set()))
            nodes_attr.pop(n, None)

            changed = True
            collapsed_this_pass += 1

        if collapsed_this_pass == 0:
            break

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
    relevance_stat="sum",
    show_parameters=True,
    show=True,
    inline_format="svg",
):
    """Render a large graph quickly.

    `layer_types`, `rankdir`, `relevance_stat` and `show_parameters` are as in
    `visualize_relevance`; `rankdir` applies to the `dot` engine only.
    """
    rankdir = _check_rankdir(rankdir)
    graph = _compact_graph(graph, layer_types, show_parameters)
    _norm = _norm_node_name

    # present nodes
    present_raw = list(graph.nodes.keys())
    norm_by_raw = {raw: _norm(raw) for raw in present_raw}
    present_norm = set(norm_by_raw.values())

    # Keep relevance for all original nodes. The collapsed renderer needs
    # relevance from nodes that were removed by simplify_graph_by_collapsing_degree2.
    all_rel_map, _ = _relevance_maps(all_wt, relevance_stat)

    rel_map = {nk: all_rel_map.get(nk, (0.0, 0.0, 0.0)) for nk in present_norm}

    # aggregate collapsed
    if collapsed_map:
        for kept_raw, removed_raws in collapsed_map.items():
            kept_norm = _norm(kept_raw)
            km, kx, kn = rel_map.get(kept_norm, (0.0, 0.0, 0.0))
            agg_m, agg_x, agg_n = km, kx, kn
            for rm_raw in removed_raws:
                rm_norm = _norm(rm_raw)
                m, x, n = all_rel_map.get(rm_norm, (0.0, 0.0, 0.0))
                if relevance_stat == "mean":  # a collapsed chain keeps the same total, so sums are not added
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
        "outputorder": "nodesfirst",
    }
    if engine == "dot":
        graph_attr["rankdir"] = rankdir
        graph_attr["splines"] = "spline"
        graph_attr["concentrate"] = "true"
    else:
        graph_attr["splines"] = "true"
        if not disable_concentrate_for_sfdp:
            graph_attr["concentrate"] = "true"

    g = graphviz.Digraph(
        "DLBacktraceFast",
        format="svg",
        engine=engine,
        graph_attr=graph_attr,
        node_attr={"fontname": "Helvetica", "fontsize": "9"},
        edge_attr={"arrowsize": "0.7", "penwidth": "1.4", "color": "#111827"},
    )

    def _short(s, n=48):
        return s if len(s) <= n else s[:n-1] + "…"

    # nodes
    for raw in present_raw:
        nk = norm_by_raw[raw]
        mean, mx, mn = rel_map.get(nk, (0.0, 0.0, 0.0))
        fill = _node_fill(graph.nodes[raw])
        collapsed = graph.nodes[raw].get("collapsed_count", 0)
        collapsed_line = f"\n[collapsed {collapsed}]" if collapsed else ""

        g.node(
            nk,
            label=(
                f"{_short(nk)}\n"
                f"{relevance_stat.capitalize()}: {_fmt_rel(mean)}\n"
                f"Max: {_fmt_rel(mx)}\n"
                f"Min: {_fmt_rel(mn)}"
                f"{collapsed_line}"
            ),
            style="filled",
            fillcolor=fill,
        )

    collapsed_owner = {}
    if collapsed_map:
        for kept_raw, removed_raws in collapsed_map.items():
            for removed_raw in removed_raws:
                collapsed_owner[removed_raw] = kept_raw

    def _visible_raw(raw):
        seen = set()
        while raw in collapsed_owner and raw not in seen:
            seen.add(raw)
            raw = collapsed_owner[raw]
        return raw if raw in norm_by_raw else None

    # edges
    added = set()
    for raw in present_raw:
        child = norm_by_raw[raw]
        parents = graph.nodes[raw].get("parents", []) or []
        if not parents and hasattr(graph, "predecessors"):
            try:
                parents = list(graph.predecessors(raw))
            except Exception:
                parents = []
        if max_parents_per_node is not None and len(parents) > max_parents_per_node:
            parents = sorted(
                parents,
                key=lambda p: abs(rel_map.get(_norm(p), (0.0, 0.0, 0.0))[0]),
                reverse=True,
            )[:max_parents_per_node]

        for p_raw in parents:
            visible_parent = _visible_raw(p_raw)
            if visible_parent is None:
                continue
            pn = norm_by_raw[visible_parent]
            if pn == child:
                continue
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

    print(f"✅ Fast graph saved → {out} (nodes={num_nodes}, edges={len(added)}, engine={engine})")
    return g, out


def visualize_relevance_subgraph(
    graph,
    all_wt,
    output_path="backtrace_relevance_subgraph",
    *,
    top_k=120,
    max_nodes=220,
    layer_types: Optional[Sequence[str]] = None,
    rankdir: str = "LR",
    relevance_stat="sum",
    show_parameters=True,
    show=True,
    inline_format="svg",
):
    """Render a bounded relevance-guided subgraph using direct NetworkX edges.

    With `layer_types` the subgraph is chosen among the nodes of those types only (linked
    through the nodes that were left out); `rankdir` and `relevance_stat` are as in
    `visualize_relevance`, and so is `show_parameters`.
    """
    rankdir = _check_rankdir(rankdir)
    full_nodes = len(graph.nodes)
    graph = _compact_graph(graph, layer_types, show_parameters)
    stats_by_norm, scores_by_norm = _relevance_maps(all_wt, relevance_stat)
    raw_by_norm = {_norm_node_name(raw): raw for raw in graph.nodes}
    norm_by_raw = {raw: _norm_node_name(raw) for raw in graph.nodes}

    output_nodes = [
        raw for raw in graph.nodes
        if graph.nodes[raw].get("layer_type") == "Output"
    ]
    if not output_nodes:
        output_nodes = [raw for raw in graph.nodes if graph.out_degree(raw) == 0]

    ranked = []
    for raw, norm in norm_by_raw.items():
        score = scores_by_norm.get(norm, 0.0)
        if score > 0:
            ranked.append((raw, score))
    ranked.sort(key=lambda item: item[1], reverse=True)

    selected = set(output_nodes[:3])
    top_nodes = [raw for raw, _ in ranked[:top_k]]

    def best_path_to_output(raw):
        best = None
        for out in output_nodes[:5]:
            try:
                path = nx.shortest_path(graph, raw, out)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            if best is None or len(path) < len(best):
                best = path
        return best

    for raw in top_nodes:
        if len(selected) >= max_nodes:
            break
        path = best_path_to_output(raw)
        if path:
            if len(selected | set(path)) <= max_nodes:
                selected.update(path)
            else:
                selected.add(raw)
        else:
            selected.add(raw)

    # If paths were unavailable or too sparse, add direct neighbors of top nodes.
    for raw in top_nodes:
        if len(selected) >= max_nodes:
            break
        if raw not in selected:
            continue
        neighbors = list(graph.predecessors(raw))[:2] + list(graph.successors(raw))[:2]
        for n in neighbors:
            if len(selected) >= max_nodes:
                break
            selected.add(n)

    subgraph = graph.subgraph(selected).copy()
    if subgraph.number_of_edges() == 0 and top_nodes:
        selected = set()
        for raw in top_nodes:
            if len(selected) >= max_nodes:
                break
            selected.add(raw)
            for n in list(graph.predecessors(raw))[:2] + list(graph.successors(raw))[:2]:
                if len(selected) >= max_nodes:
                    break
                selected.add(n)
        subgraph = graph.subgraph(selected).copy()

    def _short(s, n=42):
        return s if len(s) <= n else s[:n - 1] + "…"

    g = graphviz.Digraph(
        "DLBacktraceRelevanceSubgraph",
        format="svg",
        engine="dot",
        graph_attr={
            "rankdir": rankdir,
            "splines": "spline",
            "concentrate": "true",
            "nodesep": "0.22",
            "ranksep": "0.42",
            "margin": "0.05",
            "outputorder": "nodesfirst",
        },
        node_attr={"fontname": "Helvetica", "fontsize": "9", "shape": "box", "style": "rounded,filled"},
        edge_attr={"arrowsize": "0.65", "penwidth": "1.2", "color": "#111827"},
    )

    for raw in subgraph.nodes:
        norm = norm_by_raw.get(raw, _norm_node_name(raw))
        mean, mx, mn = stats_by_norm.get(norm, (0.0, 0.0, 0.0))
        g.node(
            norm,
            label=(
                f"{_short(norm)}\n"
                f"{_get_node_category(graph.nodes[raw])}\n"
                f"{relevance_stat.capitalize()}: {_fmt_rel(mean)}\n"
                f"Max: {_fmt_rel(mx)}"
            ),
            fillcolor=_node_fill(graph.nodes[raw]),
        )

    for src, dst in subgraph.edges:
        g.edge(norm_by_raw.get(src, _norm_node_name(src)), norm_by_raw.get(dst, _norm_node_name(dst)))

    out = g.render(output_path, cleanup=True)

    if show:
        if inline_format.lower() == "svg":
            display(SVG(g.pipe(format="svg")))
        else:
            display(IPyImage(data=g.pipe(format="png")))

    top_score = ranked[0][1] if ranked else 0.0
    print(
        "✅ Relevance subgraph saved → "
        f"{out} (full_nodes={full_nodes}, "
        f"rendered_nodes={subgraph.number_of_nodes()}, "
        f"rendered_edges={subgraph.number_of_edges()}, "
        f"top_score={_fmt_rel(top_score)})"
    )
    return g, out


def visualize_relevance_auto(
    graph,
    all_wt,
    output_path="backtrace_graph",
    *,
    node_threshold=500,
    engine_auto_threshold=1500,
    fast_output_path="backtrace_collapsed_fast",
    graph_mode="top_k",
    graph_top_k=120,
    graph_max_nodes=220,
    layer_types: Optional[Sequence[str]] = None,
    rankdir: str = "LR",
    relevance_stat="sum",
    show_parameters=True,
    show=True,
    inline_format="svg",
):
    """Render either a relevance-guided top-k subgraph or the entire graph.

    `layer_types` (see `SEMANTIC_LAYER_TYPES`) restricts either mode to the nodes of those
    types, linked through the nodes that were left out; the node count that picks the pretty
    or the fast renderer for `graph_mode="entire"` is taken after the filter. `rankdir`
    ("LR", "TB", "RL", "BT") is the direction of the graph.
    """
    graph_mode = (graph_mode or "top_k").strip().lower()
    print(f"num_nodes: {len(graph.nodes)}")

    if graph_mode in {"top_k", "topk", "subgraph", "relevance"}:
        print("rendering relevance-guided top-k subgraph ...")
        return visualize_relevance_subgraph(
            graph,
            all_wt,
            output_path=fast_output_path,
            top_k=graph_top_k,
            max_nodes=graph_max_nodes,
            layer_types=layer_types,
            rankdir=rankdir,
            relevance_stat=relevance_stat,
            show_parameters=show_parameters,
            show=show,
            inline_format=inline_format,
        )

    if graph_mode not in {"entire", "full", "all"}:
        raise ValueError("graph_mode must be one of {'top_k', 'entire'}")

    graph = _compact_graph(graph, layer_types, show_parameters)
    num_nodes = len(graph.nodes)

    if num_nodes < node_threshold:
        print("rendering entire graph ...")
        return visualize_relevance(
            graph,
            all_wt,
            output_path=output_path,
            rankdir=rankdir,
            relevance_stat=relevance_stat,
            show_parameters=show_parameters,
            show=show,
            inline_format=inline_format,
        )
    else:
        print("big graph → rendering entire graph; this can be slow ...")
        return visualize_relevance_fast(
            graph,
            all_wt,
            output_path=fast_output_path,
            collapsed_map=None,
            max_parents_per_node=None,
            engine_auto_threshold=engine_auto_threshold,
            rankdir=rankdir,
            relevance_stat=relevance_stat,
            show_parameters=show_parameters,
            show=show,
            inline_format=inline_format,
        )


# ─────────────────────────────────────────
# paginated graphs: a long DAG as several pages
# ─────────────────────────────────────────
def _paginate(graph, max_nodes_per_page):
    """Split the nodes of a DAG into pages of at most `max_nodes_per_page`, in execution order."""
    if max_nodes_per_page < 1:
        raise ValueError("max_nodes_per_page must be at least 1")
    position = {n: i for i, n in enumerate(graph.nodes)}
    order = list(nx.lexicographical_topological_sort(graph, key=position.get))
    return [order[i:i + max_nodes_per_page] for i in range(0, len(order), max_nodes_per_page)]


def _relevance_fill(value, max_abs):
    """Green for positive relevance, red for negative, white for zero; darker is larger."""
    t = max(-1.0, min(1.0, value / max_abs))
    c = int(255 * (1 - abs(t)))
    return f"#{c:02x}ff{c:02x}" if t >= 0 else f"#ff{c:02x}{c:02x}"


def _page_graph(graph, pages, index, stats, max_abs, rankdir, stat_name):
    """The Graphviz graph of one page. Parents on earlier pages appear as dashed grey stubs."""
    members = set(pages[index])
    g = graphviz.Digraph(
        name=f"DLBacktrace_Page{index + 1}",
        format="svg",
        graph_attr={
            "rankdir": rankdir,
            "label": f"DLBacktrace Graph - Page {index + 1}/{len(pages)}",
            "labelloc": "t",
            "fontsize": "14",
            "nodesep": "0.3",
            "ranksep": "0.5",
        },
        node_attr={"shape": "box", "style": "filled,rounded", "fontname": "Helvetica", "fontsize": "10"},
    )
    stubs = []
    for node in pages[index]:
        for parent in graph.nodes[node]["parents"]:
            if parent not in members and parent not in stubs:
                stubs.append(parent)
    for parent in stubs:
        g.node(_norm_node_name(parent), label=f"{_norm_node_name(parent)}\n(earlier page)",
               fillcolor="lightgray", style="filled,rounded,dashed")
    for node in pages[index]:
        name = _norm_node_name(node)
        value, mx, mn = stats.get(name, (0.0, 0.0, 0.0))
        g.node(name, label=(f"{name}\n{stat_name}: {_fmt_rel(value)}\n"
                            f"Max: {_fmt_rel(mx)} Min: {_fmt_rel(mn)}"),
               fillcolor=_relevance_fill(value, max_abs))
    for node in pages[index]:
        for parent in graph.nodes[node]["parents"]:
            g.edge(_norm_node_name(parent), _norm_node_name(node))
    return g


def _svg_size(svg):
    """(width, height) in px of a Graphviz SVG (it states them in pt)."""
    m = re.search(r'<svg[^>]*?\swidth="([\d.]+)pt"[^>]*?\sheight="([\d.]+)pt"', svg)
    return float(m.group(1)) * 4 / 3, float(m.group(2)) * 4 / 3


def _combine_svgs(svgs, pages_per_row, gap=24, title=None):
    """Lay the page SVGs out in a grid, `pages_per_row` to a row, and return one SVG string."""
    sizes = [_svg_size(svg) for svg in svgs]
    rows = [list(range(i, min(i + pages_per_row, len(svgs)))) for i in range(0, len(svgs), pages_per_row)]
    col_w = [max(sizes[r[c]][0] for r in rows if c < len(r)) for c in range(pages_per_row) if any(c < len(r) for r in rows)]
    top = 30 if title else 0
    width = sum(col_w) + gap * (len(col_w) + 1)
    height = top + sum(max(sizes[i][1] for i in r) + gap for r in rows) + gap
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'width="{width:.0f}px" height="{height:.0f}px">',
             f'<rect width="100%" height="100%" fill="white"/>']
    if title:
        parts.append(f'<text x="{gap}" y="22" font-family="Helvetica" font-size="14">{title}</text>')
    y = top + gap
    for r in rows:
        x = gap
        for c, i in enumerate(r):
            body = re.sub(r"^.*?(?=<svg)", "", svgs[i], count=1, flags=re.S)  # drop the XML header
            parts.append(body.replace("<svg", f'<svg x="{x:.0f}" y="{y:.0f}"', 1))
            x += col_w[c] + gap
        y += max(sizes[i][1] for i in r) + gap
    parts.append("</svg>")
    return "\n".join(parts)


def visualize_relevance_paginated(
    graph,
    all_wt,
    output_path="backtrace_graph",
    *,
    max_nodes_per_page: int = 30,
    layer_types: Optional[Sequence[str]] = None,
    rankdir: str = "TB",
    pages_per_row: int = 1,
    relevance_stat="sum",
    show_parameters=True,
    show=True,
    inline_format="svg",
):
    """Draw a long graph as several pages of at most `max_nodes_per_page` nodes each.

    Nodes are taken in execution order. Each page is saved as `{output_path}_page{N}.svg`;
    a parent that is on an earlier page appears on the page as a dashed grey stub, so no
    edge is lost. With `pages_per_row > 1` the pages are also laid out in a grid, that many
    to a row, in `{output_path}_combined.svg` (SVG only; `inline_format="png"` shows the
    pages one by one instead of the grid). Node colour shows the sign and size of the
    relevance (green positive, red negative, darker is larger).

    `layer_types`, `rankdir`, `relevance_stat` and `show_parameters` are as in
    `visualize_relevance`.

    Returns a list of `(graphviz.Digraph, saved path)`, one per page, and the combined SVG
    path as a last entry when it was made.
    """
    rankdir = _check_rankdir(rankdir)
    if pages_per_row < 1:
        raise ValueError("pages_per_row must be at least 1")
    compact = _compact_graph(graph, layer_types, show_parameters)
    stats, _ = _relevance_maps(all_wt, relevance_stat)
    pages = _paginate(compact, max_nodes_per_page)
    print(f"📊 Paginated: {len(graph.nodes)} nodes → {len(compact)} nodes in {len(pages)} pages "
          f"(max {max_nodes_per_page} per page)")

    values = [abs(stats.get(_norm_node_name(n), (0.0,))[0]) for page in pages for n in page]
    max_abs = max(values, default=0.0) or 1.0
    stat_name = relevance_stat.capitalize()

    def inline(g):
        if inline_format.lower() == "svg":
            display(SVG(g.pipe(format="svg")))
        else:
            display(IPyImage(data=g.pipe(format="png")))

    grid = pages_per_row > 1 and len(pages) > 1
    results = []
    for i in range(len(pages)):
        g = _page_graph(compact, pages, i, stats, max_abs, rankdir, stat_name)
        out = g.render(f"{output_path}_page{i + 1}", cleanup=True)
        results.append((g, out))
        print(f"✅ Page {i + 1}/{len(pages)} saved → {out} ({len(pages[i])} nodes)")
        if show and not (grid and inline_format.lower() == "svg"):
            inline(g)

    if grid:
        svg = _combine_svgs([g.pipe(format="svg").decode("utf-8") for g, _ in results], pages_per_row,
                            title=f"DLBacktrace Graph - {len(pages)} pages, {pages_per_row} per row")
        combined = f"{output_path}_combined.svg"
        with open(combined, "w", encoding="utf-8") as f:
            f.write(svg)
        results.append((None, combined))
        print(f"✅ Combined grid saved → {combined}")
        if show and inline_format.lower() == "svg":
            display(SVG(svg))
    return results
