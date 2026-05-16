"""
Adapted from Intel NNCF's HAWQ implementation to standalone form.

Original work:
Copyright (c) 2020 Intel Corporation
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import List
from typing import Union

import torch
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.nn.modules.loss import _Loss

from .model import call_model
from .model import infer_batch_size
from .model import move_batch_to_device
from .model import split_batch


class ParameterHandler:
    def __init__(self, parameters: List[Parameter], device: torch.device | str):
        self._device = device
        self._parameters = parameters

    @property
    def parameters(self) -> List[Parameter]:
        return self._parameters

    def get_gradients(self) -> List[Union[Tensor, float]]:
        gradients = []
        for parameter in self.parameters:
            gradients.append(0.0 if parameter.grad is None else parameter.grad + 0.0)
        return gradients

    def sample_rademacher_like_params(self) -> List[Tensor]:
        samples = []
        for parameter in self.parameters:
            sample = torch.randint_like(parameter, high=2, device=self._device)
            samples.append(sample.masked_fill_(sample == 0, -1))
        return samples


class GradientsCalculator:
    def __init__(
        self,
        model: nn.Module,
        criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor],
        criterion: _Loss,
        data_loader,
        num_data_points: int,
        parameter_handler: ParameterHandler,
        device: str,
    ):
        self._model = model
        self._criterion_fn = criterion_fn
        self._criterion = criterion
        self._data_loader = data_loader
        self._num_data_points = num_data_points
        self._parameter_handler = parameter_handler
        self._device = device

    def __iter__(self):
        self._data_loader_iter = iter(self._data_loader)
        self._processed_points = 0
        return self

    def __next__(self):
        if self._processed_points >= self._num_data_points:
            raise StopIteration

        batch = next(self._data_loader_iter)
        inputs, targets = move_batch_to_device(split_batch(batch), self._device)
        batch_size = infer_batch_size(inputs, targets)
        self._processed_points += batch_size

        self._model.zero_grad(set_to_none=True)
        outputs = call_model(self._model, inputs)
        loss = self._criterion_fn(outputs, targets, self._criterion)
        grads = torch.autograd.grad(loss, self._parameter_handler.parameters, create_graph=True, retain_graph=True)
        self._model.zero_grad(set_to_none=True)
        return grads, batch_size


class HessianTraceEstimator:
    """Performs estimation of Hessian trace based on the Hutchinson algorithm."""

    def __init__(
        self,
        model: nn.Module,
        criterion_fn: Callable[[Any, Any, _Loss], torch.Tensor],
        criterion: _Loss,
        device: torch.device | str,
        data_loader,
        num_data_points: int,
        parameters: list[Parameter] | None = None,
    ):
        self._model = model
        parameters = parameters or [p for p in model.parameters() if p.requires_grad]
        self._parameter_handler = ParameterHandler(parameters, device)
        self._gradients_calculator = GradientsCalculator(
            model=self._model,
            criterion_fn=criterion_fn,
            criterion=criterion,
            data_loader=data_loader,
            num_data_points=num_data_points,
            parameter_handler=self._parameter_handler,
            device=str(device),
        )
        self._diff_eps = 1e-6

    def get_average_traces(self, max_iter: int = 500, tolerance: float = 1e-5) -> Tensor:
        avg_total_trace = 0.0
        avg_traces_per_iter = []
        mean_avg_traces_per_param = None

        for _ in range(max_iter):
            avg_traces_per_iter.append(self._calc_avg_traces_per_param())
            mean_avg_traces_per_param = self._get_mean(avg_traces_per_iter)
            mean_avg_total_trace = torch.sum(mean_avg_traces_per_param)
            diff_avg = abs(mean_avg_total_trace - avg_total_trace) / (avg_total_trace + self._diff_eps)
            if diff_avg < tolerance:
                return mean_avg_traces_per_param
            avg_total_trace = mean_avg_total_trace

        return mean_avg_traces_per_param

    def _calc_avg_traces_per_param(self) -> Tensor:
        v = self._parameter_handler.sample_rademacher_like_params()
        vhp = [torch.zeros_like(parameter) for parameter in self._parameter_handler.parameters]
        num_all_data = 0
        for gradients, batch_size in self._gradients_calculator:
            vhp_curr = torch.autograd.grad(
                gradients,
                self._parameter_handler.parameters,
                grad_outputs=v,
                only_inputs=True,
                retain_graph=False,
            )
            vhp = [a + b * float(batch_size) + 0.0 for a, b in zip(vhp, vhp_curr)]
            num_all_data += batch_size
        vhp = [a / float(num_all_data) for a in vhp]
        return torch.stack([torch.sum(a * b) / a.size().numel() for (a, b) in zip(vhp, v)])

    @staticmethod
    def _get_mean(data: List[Tensor]) -> Tensor:
        return torch.mean(torch.stack(data), dim=0)
