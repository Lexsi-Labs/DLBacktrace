"""Generation runs the traced graph on a longer sequence at every step.

Before the fix a graph traced without dynamic_shapes (as in the guides) was fixed
to the sample length, so every step after the first failed in `view`, and the
server's Dim(min=1, max=prompt_len) made torch.export reject one-token prompts.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")

from dl_backtrace.pytorch_backtrace import DLBacktrace
from dl_backtrace.pytorch_backtrace.dlbacktrace.dlbacktrace import sequence_dynamic_shapes


class _Tok:
    """Just enough tokenizer for generation: the sampler and server only decode."""

    def __call__(self, texts, **_):
        ids = torch.tensor([[5]])  # a one-token prompt
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, **_):
        return " ".join(str(int(i)) for i in ids)


def _model():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, use_cache=False)
    return transformers.LlamaForCausalLM._from_config(cfg, attn_implementation="eager").eval()


def _greedy(model, ids, n):
    with torch.no_grad():
        return model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n,
                              do_sample=False, use_cache=False)


@pytest.mark.parametrize("prompt_len", [4, 1])
def test_generation_from_a_static_trace_matches_greedy(prompt_len):
    model = _model()
    sample = torch.randint(1, 64, (1, 6))
    dlb = DLBacktrace(model, (sample, torch.ones_like(sample)), device="cpu", verbose=False)  # no dynamic_shapes
    ids = torch.randint(1, 64, (1, prompt_len))
    out = dlb.run_task(task="generation", inputs={"input_ids": ids, "attention_mask": torch.ones_like(ids)},
                       tokenizer=_Tok(), max_new_tokens=3, return_relevance=True)
    assert torch.equal(out["generated_ids"].cpu(), _greedy(model, ids, 3))
    assert len(out["relevance_trace"]) == 3


def test_sequence_dynamic_shapes_widens_a_one_token_sample():
    ids = torch.tensor([[7]])
    sample, shapes = sequence_dynamic_shapes((ids, torch.ones_like(ids), 3))
    assert [t.shape[1] for t in sample[:2]] == [2, 2] and sample[2] == 3
    assert shapes == ({1: torch.export.Dim.AUTO}, {1: torch.export.Dim.AUTO}, None)


def test_server_generates_from_a_one_token_prompt(tmp_path, monkeypatch):
    from dl_backtrace.cli import server

    state = server._server_state
    monkeypatch.setattr(state, "model", _model(), raising=False)
    monkeypatch.setattr(state, "tokenizer", _Tok(), raising=False)
    monkeypatch.setattr(state, "device", "cpu", raising=False)
    monkeypatch.setattr(state, "model_id", "tiny", raising=False)
    monkeypatch.setattr(state, "backend", "pytorch", raising=False)
    req = server.GenerateRequest(prompt="hi", max_new_tokens=2, return_relevance=False, return_scores=False)
    out = state.run_pytorch_generate(req, str(tmp_path / "cache"))
    assert out["num_tokens_generated"] == 2
