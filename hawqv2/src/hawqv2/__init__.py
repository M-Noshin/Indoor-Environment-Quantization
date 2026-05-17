"""Standalone HAWQ-v2 weight-only selector."""

from .bitwidth_export import extract_layer_weight_bits
from .bitwidth_export import load_bitwidths
from .bitwidth_export import save_bitwidths_json
from .config import ActivationBitwidthConfig
from .config import HAWQConfig

StandaloneHAWQPrecisionInitializer = None
run_hawqv2 = None
save_hawq_result = None

try:
    from .initializer import StandaloneHAWQPrecisionInitializer
    from .selector import run_hawqv2
    from .selector import save_hawq_result
except ModuleNotFoundError:
    # Allow lightweight JSON/export utilities to work even when torch is unavailable.
    pass

__all__ = [
    "ActivationBitwidthConfig",
    "HAWQConfig",
    "StandaloneHAWQPrecisionInitializer",
    "extract_layer_weight_bits",
    "load_bitwidths",
    "run_hawqv2",
    "save_bitwidths_json",
    "save_hawq_result",
]
