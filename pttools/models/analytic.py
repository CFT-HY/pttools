"""Base class for analytical models"""

import abc
import logging
import typing as tp

import numpy as np

import pttools.type_hints as th
from pttools.models.model import Model

logger = logging.getLogger(__name__)


class AnalyticModel(Model, abc.ABC):
    r"""A generic analytical model, where the temperature dependence of $g_\text{eff}$ is implemented directly in the
    equation of state.

    You should specify either the relativistic degrees of freedom $g_\text{eff}(\phi=s)$ and $g_\text{eff}(\phi=b)$,
    or the prefactors $a_s$ and $a_b$.
    The convention for the latter is as in :notes:`\ ` eq. 7.33. for the bag model, where
    $$p_s = a_sT^4 - V_s,$$
    $$p_b = a_bT^4 - V_b.$$

    :param V_s: $V_s = \epsilon_s$, the potential term of $p$ in the symmetric phase
    :param V_b: $V_b = \epsilon_b$, the potential term of $p$ in the broken phase
    :param a_s: prefactor of $p$ in the symmetric phase
    :param a_b: prefactor of $p$ in the broken phase
    :param g_s: $g_\text{eff}(\phi=s)$, degrees of freedom for $p$ in the symmetric phase at T=T0
    :param g_b: $g_\text{eff}(\phi=b)$, degrees of freedom for $p$ in the broken phase at T=T0
    :param name: custom name for the model
    """
    def __init__(
            self,
            V_s: float, V_b: float = 0,
            a_s: float = None, a_b: float = None,
            g_s: float = None, g_b: float = None,
            t_min: float = None, t_max: float = None,
            name: str = None,
            label_latex: str = None,
            label_unicode: str = None,
            allow_invalid: bool = False):
        self.a_s: float
        self.a_b: float
        if a_s is not None and a_b is not None and g_s is None and g_b is None:
            self.a_s = a_s
            self.a_b = a_b
        elif a_s is None and a_b is None and g_s is not None and g_b is not None:
            self.a_s = np.pi**2/90 * g_s
            self.a_b = np.pi**2/90 * g_b
        else:
            raise ValueError("Specify either a_s and a_b or g_s and g_b")

        self.bag_wn_const: float = 4 / 3 * (V_s - V_b)

        super().__init__(
            V_s=V_s, V_b=V_b,
            t_min=t_min, t_max=t_max,
            name=name, label_latex=label_latex, label_unicode=label_unicode,
            allow_invalid=allow_invalid
        )
        if self.a_s <= self.a_b:
            logger.warning(
                f"The model \"{self.name}\" does not satisfy a_s > a_b. "
                "Please check that the critical temperature is non-negative. "
                f"Got: a_s={self.a_s}, a_b={self.a_b}.")

    def export(self) -> tp.Dict[str, any]:
        return {
            **super().export(),
            "a_s": self.a_s,
            "a_b": self.a_b
        }

    def ge_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return 30/np.pi**2 * self.e_temp(temp, phase) / temp**4

    def gs_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr):
        return 45/(2*np.pi**2) * self.s_temp(temp, phase) / temp**4

    def gp_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr):
        return 90/np.pi**2 * self.p_temp(temp, phase) / temp**4
