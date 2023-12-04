import numpy as np

from pttools.models.analytic import AnalyticModel
import pttools.type_hints as th


class GieseCubicModel(AnalyticModel):
    r"""SM-like model with a cubic term in the free energy density

    $$\mathcal{F}(\phi,T) = - \frac{a_+}{3}T^4 +
    \lambda (\phi^4 - 2E\phi^3 T + \phi^2 (E^2 T_{cr}^2 + d(T^2 - T_{cr}^2)))
    + \frac{\lambda}{4}(d - E^2)^2 T_{cr}^4$$

    Does not work yet. Requires support for V(temp, phase) to work."""
    def __init__(self, d: float, E: float, lam: float, t_crit: float):
        if d <= E**2:
            raise ValueError("Symmetry breaking at low temperatures requires d > E²")

        self.d = d
        self.E = E
        self.lam = lam

    def cs2(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        raise NotImplementedError

    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        raise NotImplementedError

    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""$$p_s = - \mathcal{F}(0,T)$$
        $$p_b = - \mathcal{F}(\phi_\text{min},T)$$
        """
        return self.a_s/4 * temp**4 - self.V(temp, phase)

    def phase_min(self, temp: th.FloatOrArr):
        return self.phase_min_full(self.d, self.E, temp, self.T_crit)

    @staticmethod
    def phase_min_full(d: th.FloatOrArr, E: th.FloatOrArr, temp: th.FloatOrArr, temp_crit: th.FloatOrArr):
        r"""\phi_\text{min} = \frac{3}{4}ET + \sqrt{T^2 (9E^2/8 - d)/2 - T_{cr}^2(E^2-d)/2}"""
        return 3/4*E*temp + np.sqrt(temp**2 * (9*E**2/8 - d)/2 - temp_crit**2 * (E**2 - d)/2)

    def s_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        raise NotImplementedError

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        raise NotImplementedError

    def V(self, temp: th.FloatOrArr, phase: th.FloatOrArr):
        return self.lam * (
            phase**4
            - 2*self.E * phase**3 * temp
            + phase**2 * (self.E ** 2 * self.T_crit ** 2 + self.d * (temp ** 2 - self.T_crit ** 2))
        ) + self.lam/4 * (self.d - self.E**2)**2 * self.T_crit**4

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        raise NotImplementedError
