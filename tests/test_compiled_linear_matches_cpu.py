"""The compiled CUDA linear kernel must give the same relevance as the CPU rule.

Its block reduction raced: every thread stored its partial sum in the shared total,
so the kernel redistributed relevance with whichever partial was written last and
lost 35-45% of it per linear layer. Runs only where the compiled ops load on CUDA.
"""
import numpy as np
import pytest
import torch

from dl_backtrace.pytorch_backtrace.dlbacktrace.core.config import activation_master
from dl_backtrace.pytorch_backtrace.dlbacktrace.utils import default_v2 as UD2

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or UD2.compiled_ops_error() is not None,
    reason="needs CUDA and the compiled ops",
)


@pytest.mark.parametrize("rows, d_in, d_out, bias", [(1, 8, 3, True), (6, 16, 16, True), (1, 8, 3, False),
                                                     (4, 300, 8, True), (2, 1024, 64, False)])
def test_gpu_linear_matches_cpu_rule(rows, d_in, d_out, bias):
    torch.manual_seed(0)
    x, w = torch.randn(rows, d_in), torch.randn(d_out, d_in)
    b = torch.randn(d_out) if bias else None
    r = torch.rand(rows, d_out)
    act = activation_master["None"]
    cpu = np.asarray(UD2.calculate_wt_fc_original_linear(r.numpy(), x.numpy(), w.numpy(),
                                                        None if b is None else b.numpy(), act))
    gpu = UD2.launch_linear_gpu(r.cuda(), x.cuda(), w.cuda(), None if b is None else b.cuda(), act).cpu().numpy()
    assert np.allclose(gpu, cpu, rtol=1e-4, atol=1e-5)
    if not bias:
        assert np.isclose(gpu.sum(), r.sum().item(), rtol=1e-4)  # relevance is conserved
