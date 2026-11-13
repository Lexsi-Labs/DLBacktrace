"""A causal LM that also returns its KV cache must still get relevance on the logits only.

With use_cache=True the graph output holds the logits plus every layer's key and
value tensors, but the relevance seed is one array shaped like the logits. It was
handed to every output source, so propagation failed with "Cannot align relevance
(1, 6, vocab, 1) -> (1, heads, 6, head_dim)".
"""
import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from dl_backtrace.pytorch_backtrace import DLBacktrace


def test_evaluation_with_kv_cache_outputs():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, use_cache=True)
    model = transformers.LlamaForCausalLM._from_config(cfg, attn_implementation="eager").eval()
    ids = torch.randint(1, 64, (1, 6))
    mask = torch.ones_like(ids)
    dlb = DLBacktrace(model, (ids, mask), device="cpu", verbose=False)
    node_io = dlb.predict(ids, mask)
    assert len(node_io["output"]["input_sources"]) > 1  # logits plus cache tensors
    relevance = dlb.evaluation(mode="default", multiplier=100.0, task="generation")
    r = relevance["input_ids"]
    r = np.asarray(r[0] if isinstance(r, list) else r, dtype=np.float64)
    assert np.isfinite(r).all() and r.any()
