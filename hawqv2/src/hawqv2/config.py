"""Configuration objects for the standalone HAWQ-v2 initializer."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field


@dataclass
class ActivationBitwidthConfig:
    """Activation precision policy.

    `mode="fixed"` is the practical choice for weight-only workflows.
    `mode="inherit"` assigns each activation the max adjacent selected weight precision.
    """

    mode: str = "fixed"
    bits: int = 8


@dataclass
class HAWQConfig:
    candidate_bits: list[int] = field(default_factory=lambda: [2, 4, 8])
    selection: str = "pareto"
    compression_ratio: float | None = None
    num_data_points: int = 100
    max_trace_iters: int = 200
    tolerance: float = 1e-4
    device: str = "cuda"
    quantization_mode: str = "asymmetric"
    per_channel: bool = False
    search: str = "monotonic"
    activation: ActivationBitwidthConfig = field(default_factory=ActivationBitwidthConfig)
