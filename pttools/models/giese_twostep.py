"""Model for a two-step phase transition.

Does not work yet.
"""

import typing as tp

from pttools.models.analytic import AnalyticModel
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr


class GieseTwoStepModel(AnalyticModel):
    """Model for a two-step phase transition.

    Does not work yet. Requires support for V(temp, phase) to work.
    """

    def __init__(self, b_s: float, b_b: float, d_s: float, d_b: float):
        self.b_s: float = b_s
        self.b_b: float = b_b
        self.d_s: float = d_s
        self.d_b: float = d_b
        super().__init__()

    def p_temp[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        return tp.cast(T, self.a_s/3*temp**4 + self.V_temp(temp, phase))

    def V[T: FloatOrArr](self, phase: T) -> T:
        """The potential of this model depends on the temperature. Please use :meth:`V_temp` instead."""
        raise NotImplementedError("The potential of this model depends on the temperature. Please use V_temp().")

    def V_temp[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        r"""Temperature-dependent potential $V(T,\phi)$."""
        V_s = (self.b_s - self.d_s * temp**2)**2 - self.b_b**2
        V_b = (self.b_b - self.d_b * temp**2)**2 - self.b_b**2
        return tp.cast(T, phase * V_b + (1 - phase) * V_s)
