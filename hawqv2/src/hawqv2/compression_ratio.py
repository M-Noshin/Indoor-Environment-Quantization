"""Compression-ratio accounting for standalone HAWQ-v2."""

from __future__ import annotations

from collections import OrderedDict


class CompressionRatioCalculator:
    DEFAULT_NUMBER_OF_BITS = 8

    def __init__(self, complexities: OrderedDict[str, float]):
        self.complexities = complexities
        self.total_ops_count = sum(complexities.values()) * self.DEFAULT_NUMBER_OF_BITS

    def ratio_for_bits_configuration(self, execution_order_bits_config: list[int]) -> float:
        quantizer_ops = 0.0
        for num_bits, complexity in zip(execution_order_bits_config, self.complexities.values()):
            quantizer_ops += num_bits * complexity
        return self.total_ops_count / quantizer_ops

    def ratio_limits(self, bits: list[int]) -> tuple[float, float]:
        config_len = len(self.complexities)
        min_config = [min(bits)] * config_len
        max_config = [max(bits)] * config_len
        max_ratio = self.ratio_for_bits_configuration(min_config)
        min_ratio = self.ratio_for_bits_configuration(max_config)
        return min_ratio, max_ratio

