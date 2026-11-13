"""The same model gives the same relevance with eager and with sdpa attention.

Eager attention (Q @ K^T, scaling, mask, softmax, P @ V as separate ops) goes through the
same relevance rule as scaled_dot_product_attention. Before, its ops were propagated one by
one and the scaling constant and the additive mask took a share at each step, so a model's
input relevance depended several-fold on its attn_implementation.
"""
import contextlib
import io

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from dl_backtrace.pytorch_backtrace import DLBacktrace

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"))]
IDS = torch.tensor([[5, 9, 13, 21, 7, 3, 11]])


def _model(arch, attn, kv):
    torch.manual_seed(0)
    if arch == "llama":
        cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                       num_attention_heads=4, num_key_value_heads=kv, max_position_embeddings=64,
                                       use_cache=False)
        cls = transformers.LlamaForCausalLM
    elif arch == "gpt2":
        cfg = transformers.GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=4, n_positions=64, use_cache=False)
        cls = transformers.GPT2LMHeadModel
    else:
        cfg = transformers.BertConfig(vocab_size=64, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                                      intermediate_size=64, max_position_embeddings=64, num_labels=3)
        cls = transformers.BertForSequenceClassification
    cfg._attn_implementation = attn  # set before construction: some models pick their attention class there
    m = cls(cfg).eval()

    class Logits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            kw = {} if arch == "bert" else {"use_cache": False}
            return self.m(input_ids=input_ids, attention_mask=attention_mask, **kw).logits

    return Logits().eval()


def _token_relevance(arch, attn, kv, pad, device):
    ids = IDS.to(device)
    am = torch.ones_like(ids)
    if pad == "right":
        am[0, -2:] = 0
    elif pad == "left":
        am[0, :2] = 0
    model = _model(arch, attn, kv).to(device)
    with contextlib.redirect_stdout(io.StringIO()):
        dlb = DLBacktrace(model, (ids, am), device=device, verbose=False)
        dlb.predict(ids, am)
        rel = dlb.evaluation(mode="default", multiplier=100.0,
                             task="multi-class classification" if arch == "bert" else "generation")
    r = rel["input_ids"]
    r = r[0] if isinstance(r, list) else r
    a = np.asarray(r.detach().cpu() if torch.is_tensor(r) else r, dtype=np.float64)
    return a.reshape(a.shape[0], a.shape[1], -1).sum(-1)[0]


CASES = [("llama", 4, "none"), ("llama", 4, "left"), ("llama", 2, "none"), ("llama", 2, "left"),
         ("gpt2", 4, "none"), ("gpt2", 4, "left"), ("bert", 4, "none"), ("bert", 4, "right")]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arch,kv,pad", CASES)
def test_eager_and_sdpa_give_the_same_token_relevance(arch, kv, pad, device):
    sdpa = _token_relevance(arch, "sdpa", kv, pad, device)
    eager = _token_relevance(arch, "eager", kv, pad, device)
    assert np.isfinite(sdpa).all() and np.isfinite(eager).all()
    assert abs(sdpa.sum() - eager.sum()) <= 1e-4 * abs(sdpa.sum())
    np.testing.assert_allclose(eager, sdpa, rtol=0, atol=1e-4 * np.abs(sdpa).max())
