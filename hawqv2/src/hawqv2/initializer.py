"""
Standalone HAWQ-v2 initializer with method boundaries modeled after Intel NNCF's HAWQ implementation.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Callable

import torch
from torch import Tensor
from torch import nn

from .compression_ratio import CompressionRatioCalculator
from .config import HAWQConfig
from .hessian_trace import HessianTraceEstimator
from .model import LayerInfo
from .model import collect_target_layers
from .model import move_batch_to_device
from .model import profile_layer_complexities
from .model import resolve_device
from .model import split_batch
from .perturbations import Perturbations
from .quantization import quantization_perturbation
from .traces_order import TracesPerLayer


@dataclass
class HAWQSelection:
    layer_bits: OrderedDict[str, int]
    metric: float
    compression_ratio: float
    bit_complexity: float
    activation_policy: dict[str, Any]
    activation_bits: OrderedDict[str, int] | None


class StandaloneHAWQPrecisionInitializer:
    def __init__(
        self,
        model: nn.Module,
        data_loader,
        criterion,
        config: HAWQConfig | None = None,
        criterion_fn: Callable | None = None,
        layer_names: list[str] | None = None,
    ):
        self._config = config or HAWQConfig()
        self._criterion_fn = criterion_fn or self._default_criterion_fn
        self._criterion = criterion
        self._data_loader = data_loader
        self._device = resolve_device(self._config.device)
        self._model = model.to(self._device)
        self._layers = collect_target_layers(self._model, layer_names=layer_names)
        if not self._layers:
            raise ValueError("No supported target layers were found")

    def apply_init(self) -> dict[str, Any]:
        original_training = self._model.training
        self._model.eval()

        sample_inputs, _ = move_batch_to_device(split_batch(next(iter(self._data_loader))), self._device)
        complexities = profile_layer_complexities(self._model, self._layers, sample_inputs)
        ratio_calculator = CompressionRatioCalculator(complexities)
        min_ratio, max_ratio = ratio_calculator.ratio_limits(self._config.candidate_bits)
        if (
            self._config.selection == "compression_ratio"
            and self._config.compression_ratio is None
        ):
            raise ValueError("selection='compression_ratio' requires compression_ratio to be set")
        if (
            self._config.selection == "compression_ratio"
            and not min_ratio <= self._config.compression_ratio <= max_ratio
        ):
            raise ValueError(
                f"Invalid compression_ratio={self._config.compression_ratio}. "
                f"Expected range is [{min_ratio:.4f}, {max_ratio:.4f}]"
            )

        traces_per_layer = self._calc_traces(
            self._criterion_fn,
            self._criterion,
            self._config.max_trace_iters,
            self._config.tolerance,
        )
        bits_configurations = self.get_configs_constrained_by_traces_order(
            self._config.candidate_bits,
            len(self._layers),
            self._config.search,
        )
        perturbations = self.calc_quantization_noise()
        flops_bits_per_config = self.get_flops_bits_per_config(bits_configurations, traces_per_layer, ratio_calculator)
        omega_terms = self.calc_omega_terms(perturbations, traces_per_layer)
        configuration_metric = self.calc_hawq_metric_per_configuration(
            bits_configurations,
            perturbations,
            traces_per_layer,
            self._device,
        )
        evaluations = self._build_evaluations(
            bits_configurations,
            traces_per_layer,
            configuration_metric,
            flops_bits_per_config,
            complexities,
            omega_terms,
        )
        pareto_frontier = self.build_pareto_frontier(evaluations)
        selection = self.choose_selection(evaluations, pareto_frontier, complexities)

        if original_training:
            self._model.train()

        return {
            "method": "hawqv2_standalone",
            "config": {
                **asdict(self._config),
                "device": self._device,
            },
            "layer_names": [layer.name for layer in self._layers],
            "trace_order_low_to_high": [
                self._layers[traces_per_layer.traces_order.get_execution_index_by_traces_index(i)].name
                for i in range(len(self._layers))
            ],
            "traces": OrderedDict(
                (layer.name, float(traces_per_layer.get_by_execution_index(i).cpu()))
                for i, layer in enumerate(self._layers)
            ),
            "perturbations": {
                layer.name: {str(bits): float(perturbations.get(i, bits).cpu()) for bits in self._config.candidate_bits}
                for i, layer in enumerate(self._layers)
            },
            "omega_terms": OrderedDict(
                (layer.name, omega_terms[str(i)])
                for i, layer in enumerate(self._layers)
            ),
            "complexities": {name: float(value) for name, value in complexities.items()},
            "pareto_frontier": pareto_frontier,
            "selected_config": self._selection_to_payload(selection),
            "evaluated_configs": evaluations,
        }

    def _calc_traces(self, criterion_fn, criterion, iter_number: int, tolerance: float) -> TracesPerLayer:
        trace_estimator = HessianTraceEstimator(
            model=self._model,
            criterion_fn=criterion_fn,
            criterion=criterion,
            device=self._device,
            data_loader=self._data_loader,
            num_data_points=self._config.num_data_points,
            parameters=[layer.weight for layer in self._layers],
        )
        avg_traces = trace_estimator.get_average_traces(max_iter=iter_number, tolerance=tolerance)
        return TracesPerLayer(avg_traces)

    @staticmethod
    def get_configs_constrained_by_traces_order(bits_: list[int], num_layers: int, search: str) -> list[list[int]]:
        bits = sorted(bits_)
        if search == "all":
            return [list(config) for config in itertools.product(bits, repeat=num_layers)]
        if search != "monotonic":
            raise ValueError(f"Unsupported search strategy: {search}")
        bit_configs = []
        if num_layers == 0:
            return bit_configs
        for num_groups in range(1, len(bits) + 1):
            for combo_bits in itertools.combinations(bits, num_groups):
                for combo_partitions in itertools.combinations(list(range(1, num_layers)), num_groups - 1):
                    bit_config = []
                    prev_p = 0
                    for partition, bitwidth in zip(combo_partitions + (num_layers,), combo_bits):
                        bit_config += [bitwidth] * (partition - prev_p)
                        prev_p = partition
                    bit_configs.append(bit_config)
        return bit_configs

    def calc_quantization_noise(self) -> Perturbations:
        perturbations = Perturbations()
        for layer_id, layer in enumerate(self._layers):
            for bitwidth in self._config.candidate_bits:
                perturbation = quantization_perturbation(
                    layer.weight.detach(),
                    bitwidth,
                    mode=self._config.quantization_mode,
                    per_channel=self._config.per_channel,
                ).to(self._device)
                perturbations.add(layer_id=layer_id, bitwidth=bitwidth, perturbation=perturbation)
        return perturbations

    @staticmethod
    def calc_hawq_metric_per_configuration(
        bits_configurations: list[list[int]],
        perturbations: Perturbations,
        traces_per_layer: TracesPerLayer,
        device: str,
    ) -> list[Tensor]:
        configuration_metric = []
        for bits_config in bits_configurations:
            hawq_metric = torch.Tensor([0]).to(device)
            for trace_index, layer_bits in enumerate(bits_config):
                execution_index = traces_per_layer.traces_order.get_execution_index_by_traces_index(trace_index)
                hawq_metric += traces_per_layer.get_by_trace_index(trace_index) * perturbations.get(
                    layer_id=execution_index,
                    bitwidth=layer_bits,
                )
            configuration_metric.append(hawq_metric)
        return configuration_metric

    @staticmethod
    def calc_omega_terms(
        perturbations: Perturbations,
        traces_per_layer: TracesPerLayer,
    ) -> dict[str, dict[str, float]]:
        omega_terms = {}
        for execution_index in range(len(traces_per_layer.get_all())):
            trace = traces_per_layer.get_by_execution_index(execution_index)
            layer_terms = {}
            for bitwidth, perturbation in perturbations.get_all()[execution_index].items():
                layer_terms[str(bitwidth)] = float((trace * perturbation).cpu())
            omega_terms[str(execution_index)] = layer_terms
        return omega_terms

    def get_flops_bits_per_config(
        self,
        bits_configurations: list[list[int]],
        traces_per_layer: TracesPerLayer,
        ratio_calculator: CompressionRatioCalculator,
    ) -> list[float]:
        flops_bits_per_config = []
        for bits_config in bits_configurations:
            execution_order_config = traces_per_layer.traces_order.get_execution_order_config(bits_config)
            flops_bits_per_config.append(ratio_calculator.ratio_for_bits_configuration(execution_order_config))
        return flops_bits_per_config

    @staticmethod
    def build_pareto_frontier(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        sorted_evaluations = sorted(
            evaluations,
            key=lambda item: (item["bit_complexity"], item["omega"]),
        )
        frontier = []
        best_omega_so_far = None
        for evaluation in sorted_evaluations:
            omega = evaluation["omega"]
            if best_omega_so_far is None or omega < best_omega_so_far:
                frontier.append(evaluation)
                best_omega_so_far = omega
        return frontier

    def choose_selection(
        self,
        evaluations: list[dict[str, Any]],
        pareto_frontier: list[dict[str, Any]],
        complexities: OrderedDict[str, float],
    ) -> HAWQSelection | None:
        pool = pareto_frontier if pareto_frontier else evaluations
        if self._config.selection == "pareto":
            return None
        if self._config.selection == "min_metric":
            chosen = min(pool, key=lambda item: item["omega"])
            return self._build_selection_from_evaluation(chosen, complexities)
        if self._config.selection == "compression_ratio":
            eligible = [
                item for item in pool if item["compression_ratio"] >= float(self._config.compression_ratio)
            ]
            if not eligible:
                raise ValueError(
                    f"No configurations satisfy compression_ratio >= {self._config.compression_ratio}"
                )
            chosen = min(eligible, key=lambda item: item["omega"])
            return self._build_selection_from_evaluation(chosen, complexities)
        raise ValueError(f"Unsupported selection strategy: {self._config.selection}")

    def _build_selection(
        self,
        chosen_config_in_execution_order: list[int],
        metric_tensor: Tensor,
        compression_ratio: float,
        complexities: OrderedDict[str, float],
    ) -> HAWQSelection:
        layer_bits = OrderedDict((layer.name, bitwidth) for layer, bitwidth in zip(self._layers, chosen_config_in_execution_order))
        bit_complexity = sum(complexities[name] * bits for name, bits in layer_bits.items())
        activation_bits = self._resolve_activation_bits(layer_bits)
        return HAWQSelection(
            layer_bits=layer_bits,
            metric=float(metric_tensor.item()),
            compression_ratio=float(compression_ratio),
            bit_complexity=float(bit_complexity),
            activation_policy=asdict(self._config.activation),
            activation_bits=activation_bits,
        )

    def _build_selection_from_evaluation(
        self,
        evaluation: dict[str, Any],
        complexities: OrderedDict[str, float],
    ) -> HAWQSelection:
        layer_bits = OrderedDict((str(name), int(bits)) for name, bits in evaluation["layer_bits"].items())
        bit_complexity = sum(complexities[name] * bits for name, bits in layer_bits.items())
        activation_bits = self._resolve_activation_bits(layer_bits)
        return HAWQSelection(
            layer_bits=layer_bits,
            metric=float(evaluation["metric"]),
            compression_ratio=float(evaluation["compression_ratio"]),
            bit_complexity=float(bit_complexity),
            activation_policy=asdict(self._config.activation),
            activation_bits=activation_bits,
        )

    @staticmethod
    def _selection_to_payload(selection: HAWQSelection | None) -> dict[str, Any] | None:
        if selection is None:
            return None
        return {
            "layer_bits": selection.layer_bits,
            "omega": selection.metric,
            "metric": selection.metric,
            "compression_ratio": selection.compression_ratio,
            "bit_complexity": selection.bit_complexity,
            "activation_policy": selection.activation_policy,
            "activation_bits": selection.activation_bits,
        }

    def _build_evaluations(
        self,
        bits_configurations: list[list[int]],
        traces_per_layer: TracesPerLayer,
        configuration_metric: list[Tensor],
        flops_bits_per_config: list[float],
        complexities: OrderedDict[str, float],
        omega_terms: dict[str, dict[str, float]],
    ) -> list[dict[str, Any]]:
        evaluations = []
        for bits_config, metric_tensor, ratio in zip(bits_configurations, configuration_metric, flops_bits_per_config):
            execution_order_config = traces_per_layer.traces_order.get_execution_order_config(bits_config)
            layer_bits = OrderedDict((layer.name, bitwidth) for layer, bitwidth in zip(self._layers, execution_order_config))
            omega_by_layer = OrderedDict(
                (layer.name, omega_terms[str(layer_id)][str(bitwidth)])
                for layer_id, (layer, bitwidth) in enumerate(zip(self._layers, execution_order_config))
            )
            bit_complexity = sum(complexities[name] * bits for name, bits in layer_bits.items())
            evaluations.append(
                {
                    "layer_bits": layer_bits,
                    "omega": float(metric_tensor.item()),
                    "omega_by_layer": omega_by_layer,
                    "metric": float(metric_tensor.item()),
                    "compression_ratio": float(ratio),
                    "bit_complexity": float(bit_complexity),
                    "activation_bits": self._resolve_activation_bits(layer_bits),
                }
            )
        return evaluations

    def _resolve_activation_bits(self, layer_bits: OrderedDict[str, int]) -> OrderedDict[str, int] | None:
        if self._config.activation.mode == "disabled":
            return None
        if self._config.activation.mode == "fixed":
            return OrderedDict((name, int(self._config.activation.bits)) for name in layer_bits)
        if self._config.activation.mode == "inherit":
            return OrderedDict((name, int(bits)) for name, bits in layer_bits.items())
        raise ValueError(f"Unsupported activation mode: {self._config.activation.mode}")

    @staticmethod
    def _default_criterion_fn(outputs, targets, criterion):
        return criterion(outputs, targets)


def run_hawqv2(
    model: nn.Module,
    data_loader,
    criterion,
    layer_names: list[str] | None = None,
    candidate_bits: list[int] | tuple[int, ...] = (2, 4, 8),
    device: str = "cuda",
    criterion_fn: Callable[[Any, Any, Any], torch.Tensor] | None = None,
    num_data_points: int = 100,
    max_trace_iters: int = 200,
    tolerance: float = 1e-4,
    selection: str = "pareto",
    compression_ratio: float | None = None,
    quantization_mode: str = "asymmetric",
    per_channel: bool = False,
    search: str = "monotonic",
    eval_mode: bool = True,
    activation_mode: str = "fixed",
    activation_bits: int = 8,
) -> dict[str, Any]:
    config = HAWQConfig(
        candidate_bits=list(candidate_bits),
        selection=selection,
        compression_ratio=compression_ratio,
        num_data_points=num_data_points,
        max_trace_iters=max_trace_iters,
        tolerance=tolerance,
        device=device,
        quantization_mode=quantization_mode,
        per_channel=per_channel,
        search=search,
    )
    config.activation.mode = activation_mode
    config.activation.bits = activation_bits
    initializer = StandaloneHAWQPrecisionInitializer(
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        config=config,
        criterion_fn=criterion_fn,
        layer_names=layer_names,
    )
    result = initializer.apply_init()
    if eval_mode:
        return result
    return result
