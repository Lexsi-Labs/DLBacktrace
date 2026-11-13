"""`all` / `any` must keep their dim and keepdim arguments.

The execution engine had no hyperparameter entry for `all`, so `all(x, dim=-1,
keepdim=True)` was replayed over dim 0 without keepdim. transformers builds the
SDPA attention mask with exactly this call on CUDA (`_unmask_unattended`), so
causal LMs with the default sdpa attention gave wrong logits on GPU.
"""
import pytest
import torch
import torch.nn as nn

from dl_backtrace.pytorch_backtrace import DLBacktrace

X = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0], [0.0, 2.0, 0.0]]])


class Reduce(nn.Module):
    def __init__(self, op, kwargs):
        super().__init__()
        self.op, self.kwargs = op, kwargs

    def forward(self, x):
        mask = getattr(torch, self.op)(x == 0, **self.kwargs)
        return x + mask.float()


@pytest.mark.parametrize("op", ["all", "any"])
@pytest.mark.parametrize("kwargs", [{"dim": -1, "keepdim": True}, {"dim": 1}, {}])
def test_reduction_matches_torch(op, kwargs):
    model = Reduce(op, kwargs).eval()
    dlb = DLBacktrace(model=model, input_for_graph=(X,), device="cpu", verbose=False)
    out = dlb.predict(X)["output"]["output_values"]
    assert torch.equal(torch.as_tensor(out), model(X))
