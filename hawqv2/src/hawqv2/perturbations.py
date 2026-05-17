"""
Adapted from Intel NNCF's HAWQ implementation to standalone form.

Original work:
Copyright (c) 2020 Intel Corporation
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

from typing import Dict

from torch import Tensor


class Perturbations:
    def __init__(self):
        self._perturbations = {}  # type: Dict[int, Dict[int, Tensor]]

    def add(self, layer_id: int, bitwidth: int, perturbation: Tensor):
        if layer_id in self._perturbations:
            self._perturbations[layer_id].update({bitwidth: perturbation})
        else:
            self._perturbations[layer_id] = {bitwidth: perturbation}

    def get(self, layer_id: int, bitwidth: int) -> Tensor:
        return self._perturbations[layer_id][bitwidth]

    def get_all(self) -> Dict[int, Dict[int, Tensor]]:
        return self._perturbations

