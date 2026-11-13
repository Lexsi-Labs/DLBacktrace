"""scaled_dot_product_attention called with is_causal=True and no mask (as newer
transformers do), and with fewer K/V heads (enable_gqa=True).

Before, the graph builder missed sdpa's keyword arguments (the engine got a float as
the mask and dropped is_causal / enable_gqa), the GPU relevance paths failed on the
missing mask, and grouped K/V heads did not broadcast against Q in the relevance rule.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_backtrace.pytorch_backtrace import DLBacktrace

T, D, HQ = 5, 16, 4
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]


class CausalAttention(nn.Module):

    def __init__(self, kv_heads):
        super().__init__()
        self.kv_heads = kv_heads
        self.q = nn.Linear(D, D)
        self.kv = nn.Linear(D, 2 * D // HQ * kv_heads)
        self.out = nn.Linear(D, 2)

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, HQ, D // HQ).transpose(1, 2)
        k, v = (z.view(b, t, self.kv_heads, D // HQ).transpose(1, 2) for z in self.kv(x).chunk(2, dim=-1))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.kv_heads != HQ)
        return self.out(y.transpose(1, 2).reshape(b, t, D)[:, -1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kv_heads", [HQ, 2])
def test_causal_and_grouped_sdpa_trace_and_explain(device, kv_heads):
    torch.manual_seed(0)
    model = CausalAttention(kv_heads).eval().to(device)
    x = torch.randn(1, T, D).to(device)
    dlb = DLBacktrace(model=model, input_for_graph=(x,), device=device, verbose=False)
    out = dlb.predict(x)["output"]["output_values"]
    assert torch.allclose(out.to(device), model(x), atol=1e-5)
    rel = dlb.evaluation(mode="default", multiplier=100.0, task="binary-classification")
    r = torch.as_tensor(rel["x"]).float().reshape(T, D)
    assert torch.isfinite(r).all() and r.abs().sum() > 0
