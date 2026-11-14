import subprocess
import sys

BLOCK_COMPILED_OPS = """
import importlib.machinery

import torch
import torch.nn as nn

create_module = importlib.machinery.ExtensionFileLoader.create_module


def block_compiled_ops(self, spec):
    if spec.name.startswith("wt_"):
        raise ImportError("no compiled ops for this platform")
    return create_module(self, spec)


importlib.machinery.ExtensionFileLoader.create_module = block_compiled_ops
"""


# The ops are missing either because loading is blocked above or because the package
# has no build for this platform/Python (e.g. macOS), which reports "not found".
LOAD_ERRORS = ("no compiled ops for this platform", "not found")


def load_errors(out):
    return sum(any(e in line for e in LOAD_ERRORS) for line in out.splitlines())


def run_python(code):
    result = subprocess.run([sys.executable, "-c", BLOCK_COMPILED_OPS + code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    return result.stdout


def test_cpu_relevance_runs_without_compiled_ops():
    out = run_python(
        """
from dl_backtrace.pytorch_backtrace import DLBacktrace

torch.manual_seed(0)
model = nn.Sequential(nn.Linear(10, 4), nn.ReLU(), nn.Linear(4, 1)).eval()
x = torch.randn(1, 10)
dlb = DLBacktrace(model=model, input_for_graph=(x,), device="cpu", verbose=False)
dlb.predict(x)
relevance = dlb.evaluation(mode="default", multiplier=100.0, task="binary-classification")
print("ok", len(relevance))
"""
    )
    assert "ok" in out


def test_compiled_kernels_fail_on_call_with_the_load_error():
    out = run_python(
        """
from dl_backtrace.pytorch_backtrace.dlbacktrace.utils import default_v2

for kernel in (
    default_v2.calculate_wt_fc_cuda_linear,
    default_v2.calculate_wt_fc_cuda_tensor_linear,
    default_v2.calculate_wt_embedding_cuda,
    default_v2.calculate_wt_self_attention_cuda,
):
    try:
        kernel()
    except ImportError as e:
        print(e)
"""
    )
    assert load_errors(out) == 4


def test_gpu_launchers_do_not_fall_back_silently():
    out = run_python(
        """
from dl_backtrace.pytorch_backtrace.dlbacktrace.utils import default_v2

for launcher in (default_v2.launch_linear_gpu, default_v2.launch_embedding_gpu):
    try:
        launcher(*[None] * launcher.__code__.co_argcount)
    except (ImportError, RuntimeError) as e:
        print(type(e).__name__, e)
"""
    )
    assert load_errors(out) == 2


def test_cuda_fallback_warning_points_at_the_caller():
    out = run_python(
        """
import warnings

torch.cuda.is_available = lambda: True

from dl_backtrace.pytorch_backtrace import DLBacktrace

model = nn.Sequential(nn.Linear(4, 2)).eval()
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    dlb = DLBacktrace(model=model, input_for_graph=(torch.randn(1, 4),), device="cuda", verbose=False)

assert "cuda" not in dlb.layer_implementation.values(), dlb.layer_implementation
fallback = [w for w in caught if "compiled CUDA ops could not be loaded" in str(w.message)]
assert len(fallback) == 1, [str(w.message) for w in caught]
print("warned from", fallback[0].filename)
"""
    )
    assert "warned from <string>" in out


def test_cuda_fallback_notice_is_shown_after_a_cpu_instance():
    # Building any DLBacktrace imports core/reproducibility.py, which installs a global
    # "ignore" warnings filter; the fallback notice must still reach the user after that
    out = run_python(
        """
import sys

sys.stderr = sys.stdout

from dl_backtrace.pytorch_backtrace import DLBacktrace

model = nn.Sequential(nn.Linear(4, 2)).eval()
x = torch.randn(1, 4)
DLBacktrace(model=model, input_for_graph=(x,), device="cpu", verbose=False)
print("NOTICE-START")

torch.cuda.is_available = lambda: True
dlb = DLBacktrace(model=model, input_for_graph=(x,), device="cuda", verbose=False)
assert "cuda" not in dlb.layer_implementation.values(), dlb.layer_implementation
"""
    )
    after = out.split("NOTICE-START", 1)[1]
    assert "compiled CUDA ops could not be loaded" in after
    assert load_errors(after) > 0
