"""Sigmoid-based model"""

import numba
import numpy as np

from pttools import type_hints as th
from pttools.models.thermo import ThermoModel


@numba.njit
def sigmoid(
        x: th.FloatOrArr,
        midpoint: th.FloatOrArr,
        max_val: th.FloatOrArr,
        steepness: th.FloatOrArr) -> th.FloatOrArr:
    """`Logistic function <https://en.wikipedia.org/wiki/Logistic_function>`_"""
    return max_val / (1 + np.exp(-steepness*(x - midpoint)))


@numba.njit
def sigmoid_derivative(
        x: th.FloatOrArr,
        midpoint: th.FloatOrArr,
        max_val: th.FloatOrArr,
        steepness: th.FloatOrArr) -> th.FloatOrArr:
    """Derivative of the logistic function"""
    exp = np.exp(-steepness*(x - midpoint))
    return steepness * max_val * exp / (1 + exp)**2


class SigmoidModel(ThermoModel):
    """Preliminary idea: ThermoModel based on sigmoid functions

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
        self.pt_temp_ge = pt_temp_ge
        self.pt_temp_gs = pt_temp_gs
        self.steepness_ge = steepness_ge
        self.steepness_gs = steepness_gs
        self.ge_s = ge_s
        self.gs_s = gs_s
        self.ge_b = ge_b
        self.gs_b = gs_b
        self.ge_diff = ge_s - ge_b
        self.gs_diff = gs_s - gs_b

        raise NotImplementedError

    def dge_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        pass

    def dgs_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        pass

    def ge(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        pass

    def gs(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        pass
