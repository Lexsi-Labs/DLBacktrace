"""Two-step generation: forward_pass() generates, relevance_pass() explains afterwards.

`run_task("forward-only-generation")` is the same generation without any relevance.
The two-step result must equal what one-step `run_task("generation")` computes for
the same tokens, and neither generation-only entry point may run the relevance pass.
"""
import re

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from dl_backtrace.pytorch_backtrace import DLBacktrace

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]
ARCHS = ["llama", "llama-kv-cache", "gpt2-wrapped"]  # "wrapped": a module holding the LM in `.model`, as the CLI does
STEPS = 3


class _Tok:
    """No tokenizer is needed for forward-only generation; the one-step task only decodes."""

    def decode(self, ids, **_):
        return " ".join(str(int(i)) for i in ids)


class _Wrapper(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.model = lm

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits


def _model(arch, device):
    torch.manual_seed(0)
    if arch.startswith("llama"):
        cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=4, use_cache=arch.endswith("cache"))
        model = transformers.LlamaForCausalLM._from_config(cfg, attn_implementation="eager")
    else:
        cfg = transformers.GPT2Config(vocab_size=64, n_embd=16, n_layer=2, n_head=4, n_positions=64, use_cache=False)
        model = transformers.GPT2LMHeadModel._from_config(cfg, attn_implementation="eager")
    return model.eval().to(device)


def _setup(arch, device, prompt_len=4):
    model = _model(arch, device)
    sample = torch.randint(1, 64, (1, 6), device=device)
    traced = _Wrapper(model).eval() if arch.endswith("wrapped") else model
    dlb = DLBacktrace(traced, (sample, torch.ones_like(sample)), device=device, verbose=False)  # no dynamic_shapes
    ids = torch.randint(1, 64, (1, prompt_len), device=device)
    return model, dlb, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _greedy(model, inputs, n):
    with torch.no_grad():
        return model.generate(**inputs, max_new_tokens=n, do_sample=False, use_cache=False).cpu()


class _NoRelevance:
    """Fails the test if the relevance pass runs."""

    def __init__(self, dlb, monkeypatch):
        def boom(*a, **k):
            raise AssertionError("relevance was computed")

        monkeypatch.setattr(dlb, "evaluation", boom)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arch", ARCHS)
def test_forward_only_task_generates_without_relevance(arch, device, monkeypatch):
    model, dlb, inputs = _setup(arch, device)
    _NoRelevance(dlb, monkeypatch)
    out = dlb.run_task(task="forward-only-generation", inputs=inputs, max_new_tokens=STEPS)  # no tokenizer
    ref = _greedy(model, inputs, STEPS)
    assert out["task"] == "forward-only-generation"
    assert torch.equal(out["complete_sequence"].cpu(), ref)
    assert out["generated_token_ids"] == ref[0, -STEPS:].tolist()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arch", ARCHS)
def test_forward_pass_generates_without_relevance(arch, device, monkeypatch):
    model, dlb, inputs = _setup(arch, device)
    _NoRelevance(dlb, monkeypatch)
    out = dlb.forward_pass(inputs, max_new_tokens=STEPS)
    ref = _greedy(model, inputs, STEPS)
    assert out["num_steps"] == STEPS
    assert torch.equal(out["complete_sequence"].cpu(), ref)
    assert out["generated_token_ids"] == ref[0, -STEPS:].tolist()


def _trace_value(entry, key):
    """The one-step trace stores a list-valued node as `name[0]`, `name[1]`, ..."""
    if key in entry:
        return [entry[key]]
    parts = []
    while f"{key}[{len(parts)}]" in entry:
        parts.append(entry[f"{key}[{len(parts)}]"])
    return parts


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arch", ARCHS)
def test_two_step_relevance_equals_one_step(arch, device):
    model, dlb, inputs = _setup(arch, device)
    one = dlb.run_task(task="generation", inputs=inputs, tokenizer=_Tok(), max_new_tokens=STEPS,
                       return_relevance=True, relevance_compress_dtype="float32")
    dlb2 = DLBacktrace(dlb.model, (inputs["input_ids"], inputs["attention_mask"]), device=device, verbose=False)
    gen = dlb2.forward_pass(inputs, max_new_tokens=STEPS)
    assert torch.equal(gen["complete_sequence"].cpu(), one["generated_ids"].cpu())
    two = dlb2.relevance_pass()

    assert [r["step_index"] for r in two] == list(range(STEPS))
    assert [r["token_id"] for r in two] == gen["generated_token_ids"]
    checked = 0
    for step, entry in zip(two, one["relevance_trace"]):
        for key in {re.sub(r"\[\d+\]$", "", k) for k in entry}:
            want = [torch.as_tensor(np.asarray(t)).float().cpu() for t in _trace_value(entry, key)]
            got = step["relevance"][key]
            got = got if isinstance(got, (list, tuple)) else [got]
            got = [torch.as_tensor(np.asarray(t.cpu() if torch.is_tensor(t) else t)).float() for t in got]
            assert len(got) == len(want), key
            for g, w in zip(got, want):
                assert g.shape == w.shape, key
                assert torch.allclose(g, w, rtol=1e-4, atol=1e-5 * max(1.0, float(w.abs().max()))), key
                checked += 1
    assert checked > 50
    assert float(torch.as_tensor(np.asarray(two[0]["relevance"]["input_ids"][0] if isinstance(
        two[0]["relevance"]["input_ids"], list) else two[0]["relevance"]["input_ids"])).abs().sum()) > 0


def test_relevance_pass_selected_steps_and_sampling(monkeypatch):
    model, dlb, inputs = _setup("llama", "cpu")
    torch.manual_seed(1)
    gen = dlb.forward_pass(inputs, max_new_tokens=4, temperature=0.8, top_k=10)
    assert gen["num_steps"] == 4
    picked = dlb.relevance_pass(token_indices=[3, 0])
    assert [(r["step_index"], r["token_id"]) for r in picked] == [(3, gen["generated_token_ids"][3]),
                                                                  (0, gen["generated_token_ids"][0])]
    # Explaining the last sampled token equals one explicit forward on the prefix before it.
    seq = gen["complete_sequence"]
    dlb.predict(seq[:, :-1], torch.ones_like(seq[:, :-1]))
    direct = dlb.evaluation(task="generation", target_token_ids=[gen["generated_token_ids"][3]])
    assert np.allclose(np.asarray(picked[0]["relevance"]["input_ids"]), np.asarray(direct["input_ids"]))


def test_relevance_pass_needs_a_forward_pass_and_valid_indices():
    _, dlb, inputs = _setup("llama", "cpu")
    with pytest.raises(RuntimeError, match="forward_pass"):
        dlb.relevance_pass()
    dlb.forward_pass(inputs, max_new_tokens=2)
    for bad in ([2], [-1]):
        with pytest.raises(IndexError):
            dlb.relevance_pass(token_indices=bad)
    dlb.clear_traces()
    with pytest.raises(RuntimeError, match="forward_pass"):
        dlb.relevance_pass()


def test_forward_pass_stops_at_eos():
    model, dlb, inputs = _setup("llama", "cpu")
    first = _greedy(model, inputs, 1)[0, -1].item()
    out = dlb.forward_pass(inputs, max_new_tokens=5, eos_token_id=first)
    assert out["generated_token_ids"] == [first] and out["num_steps"] == 1
    assert len(dlb.relevance_pass()) == 1


def test_relevance_results_survive_clearing_the_object():
    _, dlb, inputs = _setup("llama", "cpu")
    dlb.forward_pass(inputs, max_new_tokens=1)
    (res,) = dlb.relevance_pass()
    n = len(res["relevance"])
    dlb.clear_traces()
    assert len(res["relevance"]) == n > 0


@pytest.mark.parametrize("arch", ["llama", "gpt2-wrapped"])
@pytest.mark.parametrize("explain", ["none", [2]])
def test_unexplained_steps_match_greedy(arch, explain):
    """Steps that skip relevance use the native KV-cache forward. For a causal LM passed
    to DLBacktrace directly ("llama") it ran the base model and took hidden states for
    logits (IndexError, or wrong tokens), because it always looked at `model.model`;
    a wrapper that holds the LM in `.model` ("gpt2-wrapped") has to keep working."""
    model, dlb, inputs = _setup(arch, "cpu")
    out = dlb.run_task(task="generation", inputs=inputs, tokenizer=_Tok(), max_new_tokens=STEPS,
                       return_relevance=explain != "none", explain_tokens=explain)
    assert torch.equal(out["generated_ids"].cpu(), _greedy(model, inputs, STEPS))


def test_store_node_io_false_keeps_nothing_and_snapshot_options_are_accepted():
    _, dlb, inputs = _setup("llama", "cpu")
    out = dlb.forward_pass(inputs, max_new_tokens=1, store_node_io=False, move_snapshots_to_cpu=False)
    assert out["num_steps"] == 1
    with pytest.raises(RuntimeError, match="forward_pass"):
        dlb.relevance_pass()


def test_beam_search_without_and_with_relevance(monkeypatch):
    """The beam path read `_should_run_dlb`, which was only defined for the non-beam path
    (UnboundLocalError with return_relevance=True)."""
    model, dlb, inputs = _setup("llama", "cpu")
    out = dlb.run_task(task="generation", inputs=inputs, tokenizer=_Tok(), max_new_tokens=2, num_beams=2,
                       return_relevance=True)
    assert len(out["relevance_trace"]) == 2
    _NoRelevance(dlb, monkeypatch)
    gen = dlb.forward_pass(inputs, max_new_tokens=2, num_beams=2)
    assert gen["num_steps"] == 2


def test_visualize_dlbacktrace_passes_show_and_inline_format(monkeypatch):
    from dl_backtrace.pytorch_backtrace.dlbacktrace import dlbacktrace as mod

    _, dlb, _ = _setup("llama", "cpu")
    seen = {}
    dlb.all_wt = {}
    monkeypatch.setattr(mod, "visualize_relevance_auto", lambda *a, **k: seen.update(k))
    dlb.visualize_dlbacktrace(show=False, inline_format="png")
    assert seen["show"] is False and seen["inline_format"] == "png"
    dlb.visualize_dlbacktrace()
    assert seen["show"] is True and seen["inline_format"] == "svg"
