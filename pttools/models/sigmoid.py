"""Sigmoid-based model.

Not yet functional
"""

import numpy as np

from pttools import type_hints as th
from pttools.models.thermo import ThermoModel
from pttools.speedup import njit
from pttools.type_hints import FloatOrArr


@njit(cache=True)
def sigmoid(
        x: th.FloatOrArr,
        midpoint: th.FloatOrArr,
        max_val: th.FloatOrArr,
        steepness: th.FloatOrArr) -> th.FloatOrArr:
    """:wikipedia:`Logistic function <Logistic_function>`."""
    return max_val / (1 + np.exp(-steepness*(x - midpoint)))


@njit(cache=True)
def sigmoid_derivative(
        x: th.FloatOrArr,
        midpoint: th.FloatOrArr,
        max_val: th.FloatOrArr,
        steepness: th.FloatOrArr) -> th.FloatOrArr:
    """Derivative of the logistic function."""
    exp = np.exp(-steepness*(x - midpoint))
    return steepness * max_val * exp / (1 + exp)**2


class SigmoidModel(ThermoModel):
    """Preliminary idea: ThermoModel based on sigmoid functions.

    TODO: work in progress
    """

    def __init__(
            self,
            pt_temp_ge: float,
            pt_temp_gs: float,
            steepness_ge: float,
            steepness_gs: float,
            ge_s: float,
            gs_s: float,
            ge_b: float,
            gs_b: float):
        super().__init__()
        self.pt_temp_ge: float = pt_temp_ge
        self.pt_temp_gs: float = pt_temp_gs
        self.steepness_ge: float = steepness_ge
        self.steepness_gs: float = steepness_gs
        self.ge_s: float = ge_s
        self.gs_s: float = gs_s
        self.ge_b: float = ge_b
        self.gs_b: float = gs_b
        self.ge_diff: float = ge_s - ge_b
        self.gs_diff: float = gs_s - gs_b

        raise NotImplementedError

    def dge_dT[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError

    def dgs_dT[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError

    def ge[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError

    def gs[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError
