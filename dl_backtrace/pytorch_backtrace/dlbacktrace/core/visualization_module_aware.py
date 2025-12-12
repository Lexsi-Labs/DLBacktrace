import graphviz
from IPython.display import display, SVG, Image as IPyImage


def attach_module_metadata(graph, fx_node_to_module):
    """
    Attach module path/class metadata to graph nodes.

    This function is purely additive and does not
    modify graph structure.
    """
    if not fx_node_to_module:
        return graph

    for node in graph.nodes:
        if node in fx_node_to_module:
            module_path, module_class = fx_node_to_module[node]
            graph.nodes[node]["module_path"] = module_path
            graph.nodes[node]["module_class"] = module_class

    return graph


def visualize_relevance_with_module_labels(
    graph,
    all_wt,
    fx_node_to_module,
    output_path="backtrace_graph_modules",
    *,
    show=True,
    inline_format="svg",
):
    """
    Full replacement visualization that preserves the original graph
    but injects module-path information into node labels.
    """
    # 1. Shallow copy the graph to avoid modifying the original
    gcopy = graph.copy()

    # 2. Attach module metadata
    attach_module_metadata(gcopy, fx_node_to_module)

    # 3. Extract relevance stats
    relevance_data = {}
    for node_name, rel in all_wt.items():
        node_key = node_name.replace("/", " ").replace(":", " ")
        if hasattr(rel, "mean"):
            stats = (float(rel.mean()), float(rel.max()), float(rel.min()))
        else:
            try:
                val = float(rel)
                stats = (val, val, val)
            except:
                stats = (0.0, 0.0, 0.0)
        relevance_data[node_key] = stats

    # 4. Build Graphviz digraph (similar to visualize_relevance, but modified)
    g = graphviz.Digraph(
        "DLBacktraceModules",
        format="svg",
        graph_attr={"rankdir": "LR", "splines": "spline"},
        node_attr={"fontname": "Helvetica", "fontsize": "10"}
    )

    # Color map reused from original
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

    # 5. Add nodes (THIS IS WHERE MODULE LABELS ARE ADDED)
    for node in gcopy.nodes:
        name = node.replace("/", " ").replace(":", " ")
        rel = relevance_data.get(name, (0.0, 0.0, 0.0))

        # Pull module info
        module_path = gcopy.nodes[node].get("module_path")
        module_label = f"\n[{module_path}]" if module_path else ""

        fill = color_map.get(gcopy.nodes[node].get("layer_type", "Unknown"), "white")

        label = (
            f"{name}{module_label}\n"
            f"Mean: {rel[0]:.3f}\n"
            f"Max: {rel[1]:.3f}\n"
            f"Min: {rel[2]:.3f}"
        )

        g.node(name, label=label, style="filled", fillcolor=fill)

    # 6. Add edges
    for node in gcopy.nodes:
        child = node.replace("/", " ").replace(":", " ")
        for parent in gcopy.nodes[node].get("parents", []):
            parent_norm = parent.replace("/", " ").replace(":", " ")
            g.edge(parent_norm, child)

    # 7. Render & show
    out = g.render(output_path, cleanup=True)

    if show:
        svg_bytes = g.pipe(format="svg") if inline_format == "svg" else g.pipe(format="png")
        display(SVG(svg_bytes) if inline_format == "svg" else IPyImage(data=svg_bytes))

    print(f"📌 Module-aware relevance graph saved → {output_path}.svg")
    return g, out
