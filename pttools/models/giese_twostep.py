"""Model for a two-step phase transition

Does not work yet.
"""

from pttools.models.analytic import AnalyticModel
import pttools.type_hints as th


class GieseTwoStepModel(AnalyticModel):
    """Model for a two-step phase transition

    Does not work yet. Requires support for V(temp, phase) to work."""
    def __init__(self, b_s: float, b_b: float, d_s: float, d_b: float):
        self.b_s = b_s
        self.b_b = b_b
        self.d_s = d_s
        self.d_b = d_b

    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.a_s/3*temp**4 + self.V(temp, phase)

    def V(self, temp: th.FloatOrArr, phase: th.FloatOrArr):
        V_s = (self.b_s - self.d_s * temp**2)**2 - self.b_b**2
        V_b = (self.b_b - self.d_b * temp**2)**2 - self.b_b**2
        return phase * V_b + (1 - phase) * V_s
