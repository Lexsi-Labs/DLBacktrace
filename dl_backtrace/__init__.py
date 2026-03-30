import os
import torch


# ── Version ─────────────────────────────────────────────────
try:
    from .version import __version__
except ImportError:
    __version__ = "0.1.1"

# ── Public API ──────────────────────────────────────────────
from .pytorch_backtrace import DLBacktrace

# MoE backend (optional — may not be needed by all users)
try:
    from .moe_pytorch_backtrace import Backtrace as MoEBacktrace
except ImportError:
    MoEBacktrace = None

__all__ = ["DLBacktrace", "MoEBacktrace", "__version__"]