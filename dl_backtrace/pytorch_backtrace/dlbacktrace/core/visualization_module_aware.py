from .visualization import visualize_relevance
from copy import deepcopy


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


def visualize_relevance_with_modules(
    graph,
    all_wt,
    fx_node_to_module,
    output_path="backtrace_graph_modules",
    *,
    show=True,
    inline_format="svg",
):
    """
    Module-aware relevance visualization.

    Uses the existing visualize_relevance logic,
    but extends node labels with module information.
    """
    # Work on a copy to avoid side effects
    graph = deepcopy(graph)

    # Attach metadata
    attach_module_metadata(graph, fx_node_to_module)

    # Monkey-free label extension via node attributes
    for n in graph.nodes:
        mod = graph.nodes[n].get("module_path")
        if mod:
            graph.nodes[n]["_module_suffix"] = f"\n[{mod}]"
        else:
            graph.nodes[n]["_module_suffix"] = ""

    # Call existing visualization
    return visualize_relevance(
        graph,
        all_wt,
        output_path=output_path,
        show=show,
        inline_format=inline_format,
    )
