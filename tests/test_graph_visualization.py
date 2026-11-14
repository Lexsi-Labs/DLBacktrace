"""Graph visualisation: layer-type filtering, compact graphs, pages, rankdir and the label.

Most tests check the Graphviz source and the files written, with `render` / `pipe` stubbed, so
they do not need the `dot` program. The tests at the end render for real and are skipped when
`dot` is not installed.
"""
import re
import shutil
import xml.etree.ElementTree as ET

import networkx as nx
import numpy as np
import pytest
import torch

graphviz = pytest.importorskip("graphviz")

from dl_backtrace.pytorch_backtrace import DLBacktrace
from dl_backtrace.pytorch_backtrace.dlbacktrace import SEMANTIC_LAYER_TYPES
from dl_backtrace.pytorch_backtrace.dlbacktrace.core import visualization as viz

HAS_DOT = shutil.which("dot") is not None
needs_dot = pytest.mark.skipif(not HAS_DOT, reason="Graphviz `dot` is not installed")

FAKE_SVG = '<?xml version="1.0"?>\n<svg width="{w}pt" height="{h}pt" viewBox="0 0 {w} {h}"><g/></svg>'


@pytest.fixture
def no_dot(monkeypatch):
    """Stub the two Graphviz calls that need `dot`; the source of each graph is still built."""
    monkeypatch.setattr(graphviz.Digraph, "render",
                        lambda self, filename=None, **kw: f"{filename}.{self.format}")
    monkeypatch.setattr(graphviz.Digraph, "pipe", lambda self, **kw: FAKE_SVG.format(w=300, h=150).encode())
    monkeypatch.setattr(viz, "display", lambda *a, **k: None)


def _graph():
    """embedding -> [view -> linear] -> add (residual) -> silu -> linear_1 -> output, plus parameters."""
    nodes = {
        "input_ids": ("Placeholder", "Placeholder", []),
        "attention_mask": ("Placeholder", "Placeholder", []),
        "p_model_w": ("Placeholder", "Placeholder", []),
        "b_buf": ("Placeholder", "Placeholder", []),
        "c_lifted_tensor_0": ("Placeholder", "Placeholder", []),
        "embedding": ("ATen_Operation", "NLP_Embedding", ["input_ids", "p_model_w"]),
        "view": ("ATen_Operation", "Vector_Operation", ["embedding"]),
        "linear": ("ATen_Operation", "MLP_Layer", ["view", "p_model_w"]),
        "add": ("ATen_Operation", "Mathematical_Operation", ["embedding", "linear", "c_lifted_tensor_0"]),
        "silu": ("ATen_Operation", "Activation", ["add", "attention_mask"]),
        "linear_1": ("ATen_Operation", "MLP_Layer", ["silu"]),
        "output": ("Output", "Output", ["linear_1"]),
    }
    g = nx.DiGraph()
    for name, (layer_type, layer_name, parents) in nodes.items():
        g.add_node(name, layer_type=layer_type, layer_name=layer_name, parents=list(parents))
    for name, (_, _, parents) in nodes.items():
        g.add_edges_from((p, name) for p in parents)
    return g


def _relevance(graph, size=8):
    """A relevance of `size` ones for every node: its sum is `size`, its mean 1."""
    return {n: torch.ones(1, size // 2, 2) for n in graph.nodes}


def _edges(source):
    return set(re.findall(r'^\s*"?([\w ]+?)"? -> "?([\w ]+?)"?(?:\s|\[|$)', source, flags=re.M))


def _node(source, name):
    """The attribute text of one node statement."""
    return re.search(rf'^\s*"?{re.escape(name)}"? \[(.*?)\]$', source, flags=re.M | re.S).group(1)


def _names(source):
    return set(re.findall(r'^\s*"?([\w ]+?)"?\s+\[label=', source, flags=re.M))


# ---------------------------------------------------------------- categories and compaction
def test_category_prefers_layer_name_and_falls_back_to_layer_type():
    assert viz._get_node_category({"layer_type": "ATen_Operation", "layer_name": "MLP_Layer"}) == "MLP_Layer"
    assert viz._get_node_category({"layer_type": "Output", "layer_name": "Output"}) == "Output"
    assert viz._get_node_category({"layer_type": "ATen_Operation", "layer_name": "Unknown_Operation"}) == "ATen_Operation"
    assert viz._get_node_category({"layer_type": "Operation"}) == "Operation"


def test_parameter_placeholders_are_excluded_but_not_other_nodes():
    assert viz._is_excluded_node("p_model_w", {"layer_type": "Placeholder"})
    assert viz._is_excluded_node("b_buf", {"layer_type": "Placeholder"})
    assert viz._is_excluded_node("c_lifted_tensor_0", {"layer_type": "Placeholder"})
    assert not viz._is_excluded_node("input_ids", {"layer_type": "Placeholder"})
    assert not viz._is_excluded_node("p_proj", {"layer_type": "ATen_Operation"})


def test_compact_graph_without_filter_is_the_graph_itself():
    g = _graph()
    assert viz._compact_graph(g) is g


def test_compact_graph_can_drop_parameters():
    c = viz._compact_graph(_graph(), show_parameters=False)
    assert set(c.nodes) == set(_graph().nodes) - {"p_model_w", "b_buf", "c_lifted_tensor_0"}
    assert set(c.predecessors("linear")) == {"view"}


def test_compact_graph_keeps_selected_types_and_links_through_dropped_nodes(capsys):
    c = viz._compact_graph(_graph(), SEMANTIC_LAYER_TYPES, show_parameters=False)
    # view (Vector_Operation) and add (Mathematical_Operation) are dropped; the inputs, the
    # output and the semantic layers stay
    assert set(c.nodes) == {"input_ids", "attention_mask", "embedding", "linear", "silu", "linear_1", "output"}
    assert set(c.edges) == {
        ("input_ids", "embedding"),
        ("embedding", "linear"),     # through view
        ("embedding", "silu"),       # through add (residual)
        ("linear", "silu"),          # through add
        ("attention_mask", "silu"),
        ("silu", "linear_1"),
        ("linear_1", "output"),
    }
    assert c.nodes["silu"]["parents"] == ["embedding", "linear", "attention_mask"]
    assert "12 nodes -> 7 nodes" in capsys.readouterr().out


def test_compact_graph_keeps_parameter_placeholders_by_default(capsys):
    c = viz._compact_graph(_graph(), SEMANTIC_LAYER_TYPES)   # "Placeholder" is one of the types
    assert {"p_model_w", "b_buf", "c_lifted_tensor_0"} <= set(c.nodes)
    assert ("p_model_w", "embedding") in c.edges
    assert not {"p_model_w", "b_buf"} & set(viz._compact_graph(_graph(), ["MLP_Layer"]).nodes)  # not a requested type


def test_compact_graph_single_type_still_keeps_the_output(capsys):
    c = viz._compact_graph(_graph(), ["MLP_Layer"])
    assert set(c.nodes) == {"linear", "linear_1", "output"}
    assert set(c.edges) == {("linear", "linear_1"), ("linear_1", "output")}


def test_compact_graph_deduplicates_edges_through_two_paths():
    g = nx.DiGraph()
    for n, lt, ln, ps in [("a", "ATen_Operation", "MLP_Layer", []), ("x", "ATen_Operation", "Vector_Operation", ["a"]),
                          ("y", "ATen_Operation", "Vector_Operation", ["a"]), ("b", "ATen_Operation", "MLP_Layer", ["x", "y"])]:
        g.add_node(n, layer_type=lt, layer_name=ln, parents=ps)
        g.add_edges_from((p, n) for p in ps)
    c = viz._compact_graph(g, ["MLP_Layer"])
    assert list(c.edges) == [("a", "b")] and c.nodes["b"]["parents"] == ["a"]


def test_compact_graph_long_chain_does_not_recurse(capsys):
    g = nx.DiGraph()
    g.add_node("n0", layer_type="Placeholder", layer_name="Placeholder", parents=[])
    for i in range(1, 5000):
        g.add_node(f"n{i}", layer_type="ATen_Operation", layer_name="Vector_Operation", parents=[f"n{i-1}"])
        g.add_edge(f"n{i-1}", f"n{i}")
    g.add_node("out", layer_type="Output", layer_name="Output", parents=["n4999"])
    g.add_edge("n4999", "out")
    assert list(viz._compact_graph(g, ["Placeholder"]).edges) == [("n0", "out")]


def test_a_layer_type_that_matches_nothing_is_reported(capsys):
    viz._compact_graph(_graph(), ["MLPLayer"])
    assert "no node of type ['MLPLayer']" in capsys.readouterr().out


def test_layer_types_can_be_a_single_string_or_a_layer_type(capsys):
    assert "silu" in viz._compact_graph(_graph(), "Activation").nodes
    # "ATen_Operation" is every ATen operation, whatever its category
    assert {"view", "add", "embedding"} <= set(viz._compact_graph(_graph(), ["ATen_Operation"]).nodes)


# ---------------------------------------------------------------- the relevance label
def test_label_shows_the_sum_by_default_and_the_mean_on_request():
    value, mx, mn, _ = viz._relevance_stats(torch.ones(1, 4, 2))
    assert (value, mx, mn) == (8.0, 1.0, 1.0)
    assert viz._relevance_stats(torch.ones(1, 4, 2), "mean")[0] == 1.0
    # a node that hands relevance to several inputs: total over all of them
    assert viz._relevance_stats([torch.ones(3), torch.ones(5) * 2])[0] == 13.0
    assert viz._relevance_stats([torch.ones(3), torch.ones(5) * 2], "mean")[0] == 6.5
    with pytest.raises(ValueError):
        viz._relevance_stats(torch.ones(2), "median")


def test_sum_does_not_depend_on_the_tensor_size_but_the_mean_does():
    """The same total (100) on tensors of different sizes: equal sums, different means."""
    small, large = torch.full((1, 2, 2), 25.0), torch.full((1, 2, 50), 1.0)
    assert viz._relevance_stats(small)[0] == viz._relevance_stats(large)[0] == 100.0
    assert viz._relevance_stats(small, "mean")[0] != viz._relevance_stats(large, "mean")[0]


def test_entire_graph_label_uses_sum_and_stat_mean_restores_the_mean(no_dot):
    g = _graph()
    sum_g, _ = viz.visualize_relevance(g, _relevance(g), "x", show=False)
    assert "Sum: 8.000" in sum_g.source and "Mean:" not in sum_g.source
    mean_g, _ = viz.visualize_relevance(g, _relevance(g), "x", show=False, relevance_stat="mean")
    assert "Mean: 1.000" in mean_g.source and "Sum:" not in mean_g.source


# ---------------------------------------------------------------- entire graph, rankdir, filter
def test_entire_graph_shows_parameters_by_default_and_every_edge(no_dot):
    g = _graph()
    dot, _ = viz.visualize_relevance(g, _relevance(g), "x", show=False)
    assert _names(dot.source) == set(g.nodes)
    assert _edges(dot.source) == set(g.edges)


@pytest.mark.parametrize("fn", ["visualize_relevance", "visualize_relevance_fast", "visualize_relevance_subgraph",
                                "visualize_relevance_paginated", "visualize_relevance_auto"])
def test_show_parameters_false_hides_parameter_placeholders_in_every_renderer(no_dot, fn, tmp_path):
    g = _graph()
    r = _relevance(g)
    kw = dict(graph_mode="entire") if fn == "visualize_relevance_auto" else {}
    shown = getattr(viz, fn)(g, r, str(tmp_path / "x"), show=False, **kw)
    hidden = getattr(viz, fn)(g, r, str(tmp_path / "x"), show=False, show_parameters=False, **kw)
    src = lambda res: res[0][0].source if isinstance(res, list) else res[0].source
    assert {"p_model_w", "b_buf", "c_lifted_tensor_0"} <= _names(src(shown))
    assert not _names(src(hidden)) & {"p_model_w", "b_buf", "c_lifted_tensor_0"}
    assert {"input_ids", "attention_mask", "embedding", "output"} <= _names(src(hidden))
    assert not {e for e in _edges(src(hidden)) if "p_model_w" in e or "b_buf" in e}


@pytest.mark.parametrize("rankdir", ["LR", "TB", "RL", "BT"])
def test_rankdir_reaches_every_renderer(no_dot, rankdir):
    g = _graph()
    r = _relevance(g)
    for dot in (viz.visualize_relevance(g, r, "x", rankdir=rankdir, show=False)[0],
                viz.visualize_relevance_fast(g, r, "x", rankdir=rankdir, show=False)[0],
                viz.visualize_relevance_subgraph(g, r, "x", rankdir=rankdir, show=False)[0],
                viz.visualize_relevance_paginated(g, r, "x", rankdir=rankdir, show=False)[0][0]):
        assert f"rankdir={rankdir}" in dot.source


def test_rankdir_default_is_lr_and_pages_default_to_tb(no_dot):
    g = _graph()
    r = _relevance(g)
    assert "rankdir=LR" in viz.visualize_relevance(g, r, "x", show=False)[0].source
    assert "rankdir=TB" in viz.visualize_relevance_paginated(g, r, "x", show=False)[0][0].source


def test_bad_rankdir_is_rejected(no_dot):
    g = _graph()
    with pytest.raises(ValueError, match="rankdir"):
        viz.visualize_relevance(g, _relevance(g), "x", rankdir="sideways", show=False)


@pytest.mark.parametrize("fn", ["visualize_relevance", "visualize_relevance_fast", "visualize_relevance_subgraph"])
def test_layer_types_filter_each_renderer(no_dot, fn):
    g = _graph()
    dot, _ = getattr(viz, fn)(g, _relevance(g), "x", layer_types=SEMANTIC_LAYER_TYPES, show_parameters=False,
                              show=False)
    names = _names(dot.source)
    assert {"embedding", "linear", "silu", "linear_1", "output"} <= names
    assert not names & {"view", "add", "p_model_w", "b_buf", "c_lifted_tensor_0"}
    assert ("embedding", "silu") in _edges(dot.source)  # linked through the dropped add


def test_colour_follows_the_category_not_aten_operation(no_dot):
    g = _graph()
    dot, _ = viz.visualize_relevance(g, _relevance(g), "x", show=False)
    assert "fillcolor=lightblue" in _node(dot.source, "linear")       # MLP_Layer
    assert "fillcolor=orange" in _node(dot.source, "silu")            # Activation
    assert "fillcolor=red" in _node(dot.source, "output")


def test_auto_picks_the_renderer_after_the_filter(no_dot, capsys):
    g = _graph()
    r = _relevance(g)
    # 12 nodes: the entire graph has 12, the compact one 7 once the parameters are hidden, 10 with them
    viz.visualize_relevance_auto(g, r, "x", graph_mode="entire", node_threshold=8, show=False)
    assert "big graph" in capsys.readouterr().out
    viz.visualize_relevance_auto(g, r, "x", graph_mode="entire", node_threshold=8,
                                 layer_types=SEMANTIC_LAYER_TYPES, show_parameters=False, show=False)
    assert "rendering entire graph" in capsys.readouterr().out
    with pytest.raises(ValueError, match="graph_mode"):
        viz.visualize_relevance_auto(g, r, "x", graph_mode="partial", show=False)


def test_top_k_graph_mode_still_draws_the_relevant_path(no_dot):
    g = _graph()
    r = _relevance(g)
    r["embedding"] = torch.ones(1, 4, 2) * 50   # most relevant
    dot, _ = viz.visualize_relevance_auto(g, r, "x", graph_mode="top_k", graph_top_k=1, show=False)
    names = _names(dot.source)
    assert {"embedding", "output"} <= names
    assert ("silu", "linear_1") in _edges(dot.source)  # the path from the top node to the output
    hidden, _ = viz.visualize_relevance_auto(g, r, "x", graph_mode="top_k", graph_top_k=5, show_parameters=False, show=False)
    shown, _ = viz.visualize_relevance_auto(g, r, "x", graph_mode="top_k", graph_top_k=5, show=False)
    assert not _names(hidden.source) & {"p_model_w", "b_buf", "c_lifted_tensor_0"}
    assert _names(shown.source) & {"p_model_w", "b_buf", "c_lifted_tensor_0"}


# ---------------------------------------------------------------- pages
def _long_graph(n=23):
    g = nx.DiGraph()
    g.add_node("n0", layer_type="Placeholder", layer_name="Placeholder", parents=[])
    for i in range(1, n):
        g.add_node(f"n{i}", layer_type="ATen_Operation", layer_name="MLP_Layer",
                   parents=[f"n{i-1}"] + ([f"n{i-7}"] if i >= 8 else []))
        g.add_edges_from((p, f"n{i}") for p in g.nodes[f"n{i}"]["parents"])
    return g


def test_pages_hold_at_most_the_limit_in_execution_order():
    g = _long_graph()
    pages = viz._paginate(g, 10)
    assert [len(p) for p in pages] == [10, 10, 3]
    assert [n for p in pages for n in p] == [f"n{i}" for i in range(23)]
    with pytest.raises(ValueError):
        viz._paginate(g, 0)


def test_every_node_is_on_one_page_and_no_edge_is_lost(no_dot, tmp_path):
    g = _long_graph()
    results = viz.visualize_relevance_paginated(g, _relevance(g), str(tmp_path / "p"),
                                                max_nodes_per_page=10, show=False)
    assert [out for _, out in results] == [str(tmp_path / f"p_page{i}.svg") for i in (1, 2, 3)]
    edges = set()
    for dot, _ in results:
        edges |= _edges(dot.source)
    assert edges == set(g.edges)  # also the n(i-7) -> n(i) edges that skip over a page


def test_parents_from_earlier_pages_are_dashed_stubs(no_dot, tmp_path):
    g = _long_graph()
    results = viz.visualize_relevance_paginated(g, _relevance(g), str(tmp_path / "p"), max_nodes_per_page=10, show=False)
    first, second, _ = (dot.source for dot, _ in results)
    assert "earlier page" not in first
    stub = _node(second, "n9")
    assert "(earlier page)" in stub and "dashed" in stub
    assert "Page 2/3" in second


def test_node_colour_shows_sign_and_size(no_dot, tmp_path):
    g = _long_graph(4)
    r = {n: torch.ones(2) for n in g.nodes}
    r["n1"] = torch.ones(2) * 2     # largest
    r["n2"] = -torch.ones(2)        # negative
    r["n3"] = torch.zeros(2)
    (dot, _), = viz.visualize_relevance_paginated(g, r, str(tmp_path / "p"), show=False)
    assert 'fillcolor="#00ff00"' in dot.source       # largest positive: full green
    assert 'fillcolor="#ff7f7f"' in _node(dot.source, "n2")          # negative, half the largest: light red
    assert viz._relevance_fill(0.0, 4.0) == "#ffffff"


def test_pages_per_row_makes_a_grid_file(no_dot, tmp_path):
    g = _long_graph(40)   # 4 pages of 10
    results = viz.visualize_relevance_paginated(g, _relevance(g), str(tmp_path / "p"), max_nodes_per_page=10,
                                                pages_per_row=3, show=False)
    assert len(results) == 5 and results[-1][1] == str(tmp_path / "p_combined.svg")
    root = ET.parse(results[-1][1]).getroot()
    nested = [e for e in root if e.tag.endswith("svg")]
    assert len(nested) == 4
    xs = [float(e.get("x")) for e in nested]
    ys = [float(e.get("y")) for e in nested]
    assert xs[0] == xs[3] and xs[0] < xs[1] < xs[2]       # page 4 starts the second row under page 1
    assert ys[0] == ys[1] == ys[2] < ys[3]


def test_a_single_page_makes_no_grid(no_dot, tmp_path):
    g = _long_graph(5)
    results = viz.visualize_relevance_paginated(g, _relevance(g), str(tmp_path / "p"), pages_per_row=3, show=False)
    assert len(results) == 1 and not (tmp_path / "p_combined.svg").exists()


def test_combine_svgs_places_cells_without_overlap():
    svgs = [FAKE_SVG.format(w=w, h=h) for w, h in [(300, 150), (150, 300), (225, 75)]]
    root = ET.fromstring(viz._combine_svgs(svgs, 2, gap=10))
    cells = [e for e in root if e.tag.endswith("svg")]
    boxes = [(float(c.get("x")), float(c.get("y")), *viz._svg_size(svgs[i])) for i, c in enumerate(cells)]
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        for x2, y2, w2, h2 in boxes[i + 1:]:
            assert x1 + w1 <= x2 + 1 or x2 + w2 <= x1 + 1 or y1 + h1 <= y2 + 1 or y2 + h2 <= y1 + 1
    assert float(root.get("width").removesuffix("px")) >= max(b[0] + b[2] for b in boxes)
    assert float(root.get("height").removesuffix("px")) >= max(b[1] + b[3] for b in boxes)


def test_paginated_argument_checks(no_dot, tmp_path):
    g = _long_graph(5)
    with pytest.raises(ValueError):
        viz.visualize_relevance_paginated(g, _relevance(g), str(tmp_path / "p"), pages_per_row=0, show=False)


# ---------------------------------------------------------------- through the DLBacktrace method
class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(6, 8)
        self.b = torch.nn.Linear(8, 3)

    def forward(self, x):
        return self.b(torch.relu(self.a(x)))


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    x = torch.randn(1, 6)
    dlb = DLBacktrace(_Tiny().eval(), (x,), device="cpu", verbose=False)
    dlb.predict(x)
    dlb.evaluation(mode="default", multiplier=100.0, task="multi-class classification")
    return dlb


def test_method_compact_keeps_the_semantic_layers_of_a_real_graph(tiny, no_dot, tmp_path):
    dot, _ = tiny.visualize_dlbacktrace(output_path=str(tmp_path / "g"), compact=True, graph_mode="entire", show=False)
    names = _names(dot.source)
    assert any(n.startswith("linear") for n in names) and any(n.startswith("relu") for n in names)
    assert "output" in names and any(n.startswith("p_") for n in names)   # parameters are shown by default
    dot, _ = tiny.visualize_dlbacktrace(output_path=str(tmp_path / "g"), compact=True, graph_mode="entire",
                                        show_parameters=False, show=False)
    names = _names(dot.source)
    assert any(n.startswith("linear") for n in names) and not any(n.startswith(("p_", "b_", "c_")) for n in names)
    assert "rankdir=LR" in dot.source and "Sum:" in dot.source


def test_method_layer_types_beats_compact_and_filters(tiny, no_dot, tmp_path):
    dot, _ = tiny.visualize_dlbacktrace(output_path=str(tmp_path / "g"), compact=True, layer_types=["Activation"],
                                        graph_mode="entire", show=False)
    names = _names(dot.source)
    assert any(n.startswith("relu") for n in names) and not any(n.startswith("linear") for n in names)


def test_method_paginated_with_default_and_explicit_rankdir(tiny, no_dot, tmp_path):
    res = tiny.visualize_dlbacktrace(output_path=str(tmp_path / "g"), paginated=True, max_nodes_per_page=3, show=False)
    assert len(res) >= 2 and "rankdir=TB" in res[0][0].source
    res = tiny.visualize_dlbacktrace(output_path=str(tmp_path / "g"), paginated=True, rankdir="LR", show=False)
    assert "rankdir=LR" in res[0][0].source


@pytest.mark.parametrize("kwargs", [dict(), dict(graph_mode="entire"), dict(compact=True, graph_mode="entire"),
                                    dict(paginated=True, max_nodes_per_page=4)])
def test_method_show_parameters_flag_on_a_real_graph(tiny, no_dot, tmp_path, kwargs):
    def names(**kw):
        res = tiny.visualize_dlbacktrace(output_path=str(tmp_path / "g"), show=False, **kwargs, **kw)
        dots = [d for d, _ in res] if isinstance(res, list) else [res[0]]
        return set().union(*(_names(d.source) for d in dots))
    assert any(n.startswith("p_") for n in names())
    assert not any(n.startswith(("p_", "b_", "c_")) for n in names(show_parameters=False))


def test_method_default_is_still_the_top_k_subgraph(tiny, no_dot, tmp_path):
    dot, out = tiny.visualize_dlbacktrace(output_path=str(tmp_path / "g"), show=False)
    assert "DLBacktraceRelevanceSubgraph" in dot.source and out.endswith(".svg")


# ---------------------------------------------------------------- real renders (need `dot`)
@needs_dot
@pytest.mark.parametrize("kwargs", [
    dict(),
    dict(graph_mode="entire"),
    dict(compact=True, graph_mode="entire"),
    dict(compact=True, rankdir="TB"),
    dict(layer_types=["MLP_Layer"], graph_mode="entire", rankdir="BT"),
    dict(graph_mode="entire", relevance_stat="mean"),
])
def test_real_render_is_valid_svg_with_every_node(tiny, tmp_path, kwargs):
    dot, out = tiny.visualize_dlbacktrace(output_path=str(tmp_path / "g"), show=False, **kwargs)
    svg = ET.parse(out).getroot()
    drawn = [e for e in svg.iter() if e.get("class") == "node"]
    assert len(drawn) == len(_names(dot.source)) > 2


@needs_dot
def test_real_pages_and_grid_render_as_valid_svg(tiny, tmp_path):
    res = tiny.visualize_dlbacktrace(output_path=str(tmp_path / "g"), paginated=True, max_nodes_per_page=4,
                                     pages_per_row=2, show=False)
    pages, combined = res[:-1], res[-1][1]
    assert len(pages) >= 2
    total = 0
    for dot, out in pages:
        root = ET.parse(out).getroot()
        total += len([e for e in root.iter() if e.get("class") == "node"])
    grid = ET.parse(combined).getroot()
    assert len([e for e in grid if e.tag.endswith("svg")]) == len(pages)
    assert len([e for e in grid.iter() if e.get("class") == "node"]) == total
