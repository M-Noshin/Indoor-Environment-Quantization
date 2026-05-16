"""Model inspection helpers for the standalone HAWQ-v2 initializer."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class LayerInfo:
    name: str
    module: nn.Module
    weight_module: nn.Module
    weight: torch.nn.Parameter


def collect_target_layers(model: nn.Module, layer_names: list[str] | None = None) -> list[LayerInfo]:
    requested = set(layer_names or [])
    layers = []
    selected_names = []
    for name, module in model.named_modules():
        if requested and name not in requested:
            continue
        if not requested and _has_selected_ancestor(name, selected_names):
            continue
        weight_module = get_weight_module(module)
        if weight_module is None:
            continue
        layers.append(
            LayerInfo(
                name=name,
                module=module,
                weight_module=weight_module,
                weight=weight_module.weight,
            )
        )
        selected_names.append(name)
    if requested:
        found = {layer.name for layer in layers}
        missing = sorted(requested.difference(found))
        if missing:
            raise ValueError(f"Requested layers were not found or unsupported: {missing}")
        layers.sort(key=lambda layer: layer_names.index(layer.name))
    return layers


def profile_layer_complexities(model: nn.Module, layers: list[LayerInfo], sample_inputs) -> OrderedDict[str, float]:
    hooks = []
    outputs = {}

    def hook_fn(layer_name):
        def _hook(module, inputs, output):
            outputs[layer_name] = (inputs, output)
        return _hook

    for layer in layers:
        hooks.append(layer.module.register_forward_hook(hook_fn(layer.name)))
    with torch.no_grad():
        call_model(model, sample_inputs)
    for hook in hooks:
        hook.remove()

    complexities = OrderedDict()
    for layer in layers:
        inputs, output = outputs[layer.name]
        complexities[layer.name] = float(compute_complexity(layer.weight_module, inputs, output))
    return complexities


def get_weight_module(module: nn.Module) -> nn.Module | None:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return module
    op = getattr(module, "op", None)
    if isinstance(op, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        return op
    return None


def compute_complexity(weight_module: nn.Module, inputs, output) -> float:
    if isinstance(weight_module, nn.Linear):
        return float(weight_module.in_features * weight_module.out_features)
    if isinstance(weight_module, nn.Conv1d):
        output_length = output.shape[-1]
        kernel_size = weight_module.kernel_size[0]
        in_channels = weight_module.in_channels // weight_module.groups
        return float(weight_module.out_channels * in_channels * kernel_size * output_length)
    if isinstance(weight_module, nn.Conv2d):
        output_height, output_width = output.shape[-2:]
        kernel_height, kernel_width = weight_module.kernel_size
        in_channels = weight_module.in_channels // weight_module.groups
        return float(
            weight_module.out_channels * in_channels * kernel_height * kernel_width * output_height * output_width
        )
    raise TypeError(f"Unsupported module type for complexity calculation: {type(weight_module)!r}")


def split_batch(batch):
    if isinstance(batch, dict):
        inputs = batch.get("inputs", batch.get("input"))
        targets = batch.get("targets", batch.get("target"))
        if inputs is None or targets is None:
            raise ValueError("Dictionary batches must contain input(s) and target(s)")
        return inputs, targets
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Expected dataloader batches to contain inputs and targets")


def move_batch_to_device(batch, device: str):
    inputs, targets = batch
    return move_to_device(inputs, device), move_to_device(targets, device)


def move_to_device(obj, device: str):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(move_to_device(item, device) for item in obj)
    if isinstance(obj, list):
        return [move_to_device(item, device) for item in obj]
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    return obj


def call_model(model: nn.Module, inputs):
    if isinstance(inputs, tuple):
        return model(*inputs)
    if isinstance(inputs, list):
        return model(*inputs)
    if isinstance(inputs, dict):
        return model(**inputs)
    return model(inputs)


def infer_batch_size(inputs, targets) -> int:
    if torch.is_tensor(targets) and targets.ndim > 0:
        return int(targets.shape[0])
    if torch.is_tensor(inputs):
        return int(inputs.shape[0])
    if isinstance(inputs, (tuple, list)) and inputs and torch.is_tensor(inputs[0]):
        return int(inputs[0].shape[0])
    return 1


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def _has_selected_ancestor(name: str, selected_names: list[str]) -> bool:
    return any(name.startswith(parent + ".") for parent in selected_names)

