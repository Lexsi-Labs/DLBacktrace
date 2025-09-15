# DL-Backtrace/dl_backtrace/pytorch_backtrace/dlbacktrace/core/config.py


from dl_backtrace.pytorch_backtrace.dlbacktrace.activation import activation_master
from dl_backtrace.pytorch_backtrace.dlbacktrace.aten_operations import ATEN_DEFAULTS, ATEN_HYPERPARAMS
from torch.export import default_decompositions

# Core config tables
ACTIVATION_MASTER = activation_master
ATEN_DEFAULTS_TABLE = ATEN_DEFAULTS
ATEN_HYPERPARAMS_TABLE = ATEN_HYPERPARAMS
ATEN_DECOMP_TABLE = default_decompositions()
