"""GPT-2 / nanoGPT-style eager attention: a causal-mask buffer named `bias`,
`torch.where` against a `full([])` mask value, and division by a `full([])` scale.

Before the fix the forward pass returned None (`full` rejected the 0-d size and the
`bias` buffer was treated as a layer bias and dropped), and `where` passed its
relevance to the boolean condition instead of the attention scores.
"""
import math

import pytest
import torch
import torch.nn as nn

from dl_backtrace.pytorch_backtrace import DLBacktrace

T, D = 5, 8


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D)
        self.out = nn.Linear(D, 2)
        self.register_buffer("bias", torch.tril(torch.ones(T, T, dtype=torch.bool)).view(1, T, T))

    def forward(self, x):
        q, k, v = self.qkv(x).split(D, dim=-1)
        att = torch.matmul(q, k.transpose(-1, -2))
        att = att / torch.full([], math.sqrt(D), dtype=att.dtype, device=att.device)
        mask_value = torch.full([], torch.finfo(att.dtype).min, dtype=att.dtype, device=att.device)
        att = torch.where(self.bias[:, :T, :T], att, mask_value)
        y = torch.matmul(att.softmax(dim=-1), v)
        return self.out(y.mean(dim=1))


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


def _run(device):
    torch.manual_seed(0)
    model = CausalSelfAttention().eval().to(device)
    x = torch.randn(1, T, D).to(device)
    dlb = DLBacktrace(model=model, input_for_graph=(x,), device=device, verbose=False)
    node_io = dlb.predict(x)
    return model, x, dlb, node_io


@pytest.mark.parametrize("device", DEVICES)
def test_forward_matches_torch(device):
    model, x, _, node_io = _run(device)
    out = node_io["output"]["output_values"]
    assert out is not None
    assert torch.allclose(out.to(device), model(x), atol=1e-5)


@pytest.mark.parametrize("device", DEVICES)
def test_where_relevance_skips_the_condition(device):
    _, _, dlb, node_io = _run(device)
    rel = dlb.evaluation(mode="default", multiplier=100.0, task="binary-classification")
    where = [k for k in node_io if k.startswith("where")]
    assert where
    r_cond, r_input = (float(torch.as_tensor(r).float().sum()) for r in rel[where[0]][:2])
    assert r_cond == 0.0
    assert r_input != 0.0
    for v in rel.values():
        for a in (v if isinstance(v, list) else [v]):
            assert torch.isfinite(torch.as_tensor(a).float()).all()


class LastTokenAttention(CausalSelfAttention):
    """Reads only the last position, so earlier tokens matter only through keys and values."""

    def forward(self, x):
        q, k, v = self.qkv(x).split(D, dim=-1)
        att = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(D)
        y = torch.matmul(att.softmax(dim=-1), v)
        return self.out(y[:, -1])


def test_matmul_relevance_conserves_and_splits():
    from dl_backtrace.pytorch_backtrace.dlbacktrace.core.relevance_propagation import matmul_relevance

    torch.manual_seed(0)
    A, B = torch.randn(2, 3, 4, dtype=torch.float64), torch.randn(2, 4, 5, dtype=torch.float64)
    R = torch.randn(2, 3, 5, dtype=torch.float64)
    RA, RB = matmul_relevance(R, A, B)
    assert RA.shape == A.shape and RB.shape == B.shape
    assert torch.allclose(RA.sum() + RB.sum(), R.sum())
    assert torch.allclose(RA.sum(), RB.sum())  # an even split
    RA_np, RB_np = matmul_relevance(R.numpy(), A.numpy(), B.numpy(), b_is_weight=True)
    assert abs(RA_np.sum() - R.sum().item()) < 1e-9 and RB_np.sum() == 0


@pytest.mark.parametrize("device", DEVICES)
def test_relevance_reaches_earlier_tokens_through_attention(device):
    torch.manual_seed(0)
    model = LastTokenAttention().eval().to(device)
    x = torch.randn(1, T, D).to(device)
    dlb = DLBacktrace(model=model, input_for_graph=(x,), device=device, verbose=False)
    dlb.predict(x)
    rel = dlb.evaluation(mode="default", multiplier=100.0, task="binary-classification")
    r = torch.as_tensor(rel["x"]).float().reshape(T, D)
    assert torch.isfinite(r).all()
    assert r[:-1].abs().sum() > 0  # before, matmul gave all relevance to its first operand
