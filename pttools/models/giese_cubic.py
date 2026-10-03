"""SM-like model with a cubic term in the free energy density.

Not yet functional
"""

import typing as tp

import numpy as np

from pttools.models.analytic import AnalyticModel
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr


class GieseCubicModel(AnalyticModel):
    r"""SM-like model with a cubic term in the free energy density.

    $$\mathcal{F}(\phi,T) = - \frac{a_+}{3}T^4 +
    \lambda (\phi^4 - 2E\phi^3 T + \phi^2 (E^2 T_{cr}^2 + d(T^2 - T_{cr}^2)))
    + \frac{\lambda}{4}(d - E^2)^2 T_{cr}^4$$

    Does not work yet. Requires support for V(temp, phase) to work.
    """

    def __init__(self, d: float, E: float, lam: float, t_crit: float):  # noqa: ARG002 (work in progress)
        r"""Initialize the model. This is work in progress, and the base class is not initialized.

        :param d: $d$, coefficient of the temperature-dependent quadratic term of $\mathcal{F}$
        :param E: $E$, coefficient of the cubic term of $\mathcal{F}$
        :param lam: $\lambda$, overall coupling constant of the $\phi$-dependent terms of $\mathcal{F}$
        :param t_crit: $T_{cr}$, critical temperature (not used yet)
        :raises ValueError: if $d \leq E^2$
        """
        if d <= E**2:
            raise ValueError("Symmetry breaking at low temperatures requires d > E²")

        self.d: float = d
        self.E: float = E
        self.lam: float = lam

    @tp.override
    def cs2[T: FloatOrArr](self, w: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError

    @tp.override
    def e_temp[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError

    def p_temp[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        r"""Pressure $p(T,\phi)$.

        $$p_s = - \mathcal{F}(0,T)$$
        $$p_b = - \mathcal{F}(\phi_\text{min},T)$$.
        """
        return tp.cast(T, self.a_s/4 * temp**4 - self.V_temp(temp, phase))

    def phase_min(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        r"""$\phi_\text{min}(T)$, the value of the field at the minimum of $\mathcal{F}$ in the broken phase.

        See :meth:`phase_min_full` for the equation.

        :param temp: temperature $T$
        """
        return self.phase_min_full(self.d, self.E, temp, self.T_crit)

    @staticmethod
    def phase_min_full(
            d: th.FloatOrArr, E: th.FloatOrArr, temp: th.FloatOrArr, temp_crit: th.FloatOrArr) -> th.FloatOrArr:
        r"""\phi_\text{min} = \frac{3}{4}ET + \sqrt{T^2 (9E^2/8 - d)/2 - T_{cr}^2(E^2-d)/2}."""
        return 3/4*E*temp + np.sqrt(temp**2 * (9*E**2/8 - d)/2 - temp_crit**2 * (E**2 - d)/2)

    @tp.override
    def s_temp[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError

    @tp.override
    def temp[T: FloatOrArr](self, w: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError

    def V[T: FloatOrArr](self, phase: T) -> T:
        """The potential of this model depends on the temperature. Please use :meth:`V_temp` instead."""
        raise NotImplementedError("The potential of this model depends on the temperature. Please use V_temp().")

    def V_temp[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        r"""Temperature-dependent potential $V(T,\phi)$."""
        return tp.cast(
            T,
            self.lam * (
                phase**4
                - 2*self.E * phase**3 * temp
                + phase**2 * (self.E ** 2 * self.T_crit ** 2 + self.d * (temp ** 2 - self.T_crit ** 2))
            ) + self.lam/4 * (self.d - self.E**2)**2 * self.T_crit**4
        )

    @tp.override
    def w[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        raise NotImplementedError
