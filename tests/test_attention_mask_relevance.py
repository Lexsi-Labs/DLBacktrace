"""Attention relevance respects the attention mask.

Before, padding and future tokens received relevance: the attention rule spreads each
head's relevance over all positions in proportion to Q/K/V values, and the token-id
relevance was taken from the embedding by collapsing the batch axis, which mixed
positions together. Masked positions now get exactly zero and per-head totals are kept.
"""
import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from dl_backtrace.pytorch_backtrace import DLBacktrace
from dl_backtrace.pytorch_backtrace.dlbacktrace.core.relevance_propagation import align_relevance, sdpa_relevance
from dl_backtrace.pytorch_backtrace.dlbacktrace.utils import default_v2 as UD2

T = 7
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


def _launch(r, q, k, v, m):
    return UD2.launch_self_attention("original", r, q, k, v, m)


def _qkv(hq=4, hk=4, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal((1, h, T, d)).astype(np.float32) for h in (hq, hk, hk)]


def _head_sums(x):
    return np.maximum(x, 0).sum((-2, -1)), np.minimum(x, 0).sum((-2, -1))


def test_token_relevance_keeps_positions():
    r = np.random.default_rng(0).standard_normal((1, T, 16)).astype(np.float32)
    np.testing.assert_allclose(align_relevance(r, (1, T)), r.sum(-1), rtol=1e-6, atol=1e-6)


def test_unmasked_attention_is_unchanged():
    Q, K, V = _qkv()
    R = np.random.default_rng(1).standard_normal(Q.shape).astype(np.float32)
    raw = _launch(R, Q, K, V, None)
    for mask in (None, np.zeros((1, 1, T, T), np.float32), np.ones((1, 1, T, T), bool)):
        out = sdpa_relevance(_launch, R, Q, K, V, mask, {})
        for a, b in zip(out[:3], raw[:3]):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kv_heads", [4, 2])
def test_future_keys_get_nothing_and_totals_are_kept(kv_heads):
    Q, K, V = _qkv(hk=kv_heads)
    R = np.zeros_like(Q)
    R[..., 3, :] = np.random.default_rng(2).standard_normal(R[..., 3, :].shape)  # only query 3 carries relevance
    hp = {"is_causal": True}
    RQ, RK, RV, _ = sdpa_relevance(_launch, R, Q, K, V, None, hp)
    groups = Q.shape[1] // kv_heads
    rep = lambda t: np.repeat(t, groups, axis=1)
    raw = _launch(R, Q, rep(K), rep(V), np.where(np.triu(np.ones((T, T), bool), 1), np.finfo(np.float32).min, 0)[None, None])
    assert np.all(RK[..., 4:, :] == 0) and np.all(RV[..., 4:, :] == 0)
    for got, base in ((RQ, raw[0]), (RK, raw[1].reshape(1, kv_heads, groups, T, -1).sum(2)),
                      (RV, raw[2].reshape(1, kv_heads, groups, T, -1).sum(2))):
        for s_got, s_base in zip(_head_sums(got), _head_sums(base)):
            np.testing.assert_allclose(s_got, s_base, rtol=1e-5, atol=1e-6)


def _model(arch, attn):
    torch.manual_seed(0)
    if arch == "bert":
        cfg = transformers.BertConfig(vocab_size=64, hidden_size=16, num_hidden_layers=2, num_attention_heads=4,
                                      intermediate_size=32, num_labels=3)
        return transformers.BertForSequenceClassification._from_config(cfg, attn_implementation=attn).eval(), cfg
    if arch == "gpt2":
        cfg = transformers.GPT2Config(vocab_size=64, n_embd=16, n_layer=2, n_head=4, n_positions=32, use_cache=False)
        inner = transformers.GPT2LMHeadModel._from_config(cfg, attn_implementation=attn).eval()

        class Wrap(torch.nn.Module):  # GPT-2's second positional argument is past_key_values
            def __init__(self):
                super().__init__()
                self.inner = inner

            def forward(self, input_ids, attention_mask):
                return self.inner(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits

        return Wrap().eval(), cfg
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2 if arch == "gqa" else 4, use_cache=False)
    return transformers.LlamaForCausalLM._from_config(cfg, attn_implementation=attn).eval(), cfg


def _token_relevance(arch, attn, pad, device):
    model, cfg = _model(arch, attn)
    model = model.to(device)
    torch.manual_seed(1)
    ids = torch.randint(5, cfg.vocab_size, (1, T))
    am = torch.ones(1, T, dtype=torch.long)
    if pad == "right":
        am[0, -2:] = 0
    elif pad == "left":
        am[0, :2] = 0
    ids, am = ids.to(device), am.to(device)
    dlb = DLBacktrace(model, (ids, am), device=device, verbose=False)
    dlb.predict(ids, am)
    task = "multi-class classification" if arch == "bert" else "generation"
    r = dlb.evaluation(mode="default", multiplier=100.0, task=task)["input_ids"]
    r = r[0] if isinstance(r, list) else r
    r = np.asarray(r.detach().cpu() if torch.is_tensor(r) else r, dtype=np.float64).reshape(T)
    return r, [i for i in range(T) if int(am[0, i]) == 0]


CASES = [("bert", "sdpa", "right"), ("bert", "eager", "right"), ("llama", "sdpa", "left"),
         ("llama", "eager", "left"), ("gpt2", "eager", "left"), ("gqa", "sdpa", "left")]


@pytest.mark.parametrize("arch,attn,pad", CASES)
@pytest.mark.parametrize("device", DEVICES)
def test_padding_gets_no_relevance(arch, attn, pad, device):
    r, masked = _token_relevance(arch, attn, pad, device)
    assert masked and all(r[i] == 0 for i in masked)
    assert abs(r.sum()) > 1.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
@pytest.mark.parametrize("arch,attn,pad", CASES)
def test_cuda_matches_cpu(arch, attn, pad):
    cpu, _ = _token_relevance(arch, attn, pad, "cpu")
    gpu, _ = _token_relevance(arch, attn, pad, "cuda")
    np.testing.assert_allclose(gpu, cpu, rtol=1e-4, atol=1e-4)
