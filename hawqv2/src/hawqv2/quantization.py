"""Weight quantization helpers used by the standalone HAWQ-v2 selector."""

from __future__ import annotations

import torch


def quantize_weight(
    weight: torch.Tensor,
    num_bits: int,
    mode: str = "asymmetric",
    per_channel: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    if mode == "asymmetric":
        return _asymmetric_quantize(weight, num_bits, per_channel=per_channel, eps=eps)
    if mode == "symmetric":
        return _symmetric_quantize(weight, num_bits, per_channel=per_channel, eps=eps)
    raise ValueError(f"Unsupported quantization mode: {mode}")


def quantization_perturbation(
    weight: torch.Tensor,
    num_bits: int,
    mode: str = "asymmetric",
    per_channel: bool = False,
) -> torch.Tensor:
    quantized = quantize_weight(weight, num_bits, mode=mode, per_channel=per_channel)
    return torch.norm(weight - quantized, p=2) ** 2


def _asymmetric_quantize(
    weight: torch.Tensor,
    num_bits: int,
    per_channel: bool,
    eps: float,
) -> torch.Tensor:
    qmin = 0
    qmax = (1 << num_bits) - 1
    dims = _reduce_dims(weight.dim(), per_channel)
    min_val = weight.amin(dim=dims, keepdim=True)
    max_val = weight.amax(dim=dims, keepdim=True)
    min_val = torch.minimum(min_val, torch.zeros_like(min_val))
    max_val = torch.maximum(max_val, torch.zeros_like(max_val))
    scale = (max_val - min_val).clamp_min(eps) / float(qmax - qmin)
    zero_point = torch.round(qmin - min_val / scale).clamp(qmin, qmax)
    quantized = torch.round(weight / scale + zero_point).clamp(qmin, qmax)
    return (quantized - zero_point) * scale


def _symmetric_quantize(
    weight: torch.Tensor,
    num_bits: int,
    per_channel: bool,
    eps: float,
) -> torch.Tensor:
    level_high = (1 << (num_bits - 1)) - 1
    level_low = -level_high
    dims = _reduce_dims(weight.dim(), per_channel)
    scale = weight.abs().amax(dim=dims, keepdim=True).clamp_min(eps) / float(level_high)
    quantized = torch.round(weight / scale).clamp(level_low, level_high)
    return quantized * scale


def _reduce_dims(num_dims: int, per_channel: bool) -> tuple[int, ...] | None:
    if not per_channel:
        return None
    return tuple(range(1, num_dims))
