"""Sigmoid-based model.

Not yet functional
"""

import typing as tp

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
        """Initialize the model. This is work in progress, and raises NotImplementedError.

        The degrees of freedom are modelled as sigmoid functions of the temperature.

        :param pt_temp_ge: temperature of the phase transition, i.e. the midpoint of the sigmoid, for $g_e$
        :param pt_temp_gs: temperature of the phase transition, i.e. the midpoint of the sigmoid, for $g_s$
        :param steepness_ge: steepness of the sigmoid for $g_e$
        :param steepness_gs: steepness of the sigmoid for $g_s$
        :param ge_s: $g_{e,s}$, degrees of freedom for energy density in the symmetric phase
        :param gs_s: $g_{s,s}$, degrees of freedom for entropy in the symmetric phase
        :param ge_b: $g_{e,b}$, degrees of freedom for energy density in the broken phase
        :param gs_b: $g_{s,b}$, degrees of freedom for entropy in the broken phase
        :raises NotImplementedError: always, as the model is not yet functional
        """
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

    @tp.override
    def dge_dT[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError

    @tp.override
    def dgs_dT[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError

    @tp.override
    def ge[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError

    @tp.override
    def gs[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError
