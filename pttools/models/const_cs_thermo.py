import pttools.type_hints as th
from .thermo import ThermoModel
from . import const_cs

import numpy as np


class ConstCSThermoModel(ThermoModel):
    DEFAULT_LABEL = "ConstCSThermoModel"
    DEFAULT_NAME = "const_cs_thermo"

    GEFF_DATA_LOG_TEMP = np.linspace(0, 3, 100)
    GEFF_DATA_TEMP = 10**GEFF_DATA_LOG_TEMP

    def __init__(
            self,
            a_s: float, a_b: float,
            css2: float, csb2: float,
            V_s: float, V_b: float = 0,
            t_min: float = None,
            t_max: float = None,
            t_ref: float = 1,
            name: str = None,
            label: str = None):
        # For validation
        const_cs.ConstCSModel(css2=css2, csb2=csb2, V_s=V_s, V_b=V_b, a_s=a_s, a_b=a_b)

        self.a_s = a_s
        self.a_b = a_b
        self.t_ref = t_ref
        self.mu_s = const_cs.cs2_to_mu(css2)
        self.mu_b = const_cs.cs2_to_mu(csb2)

        super().__init__(
            t_min=t_min, t_max=t_max,
            name=name, label=label
        )

    def dg_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""$$\frac{dg}{dT} = (self.mu_\pm - 4) \frac{90}{\pi^2} a_\pm T_0^{4-\mu} T^{\mu - 5}"""
        dg_dT_s = (self.mu_s - 4) * 90/np.pi**2 * self.a_s * self.t_ref**(4-self.mu_s) * temp**(self.mu_s - 5)
        dg_dT_b = (self.mu_b - 4) * 90/np.pi**2 * self.a_b * self.t_ref**(4-self.mu_b) * temp**(self.mu_b - 5)
        return dg_dT_s * phase + dg_dT_b * (1 - phase)

    def dge_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.dg_dT(temp, phase)

    def dgs_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.dg_dT(temp, phase)

    def g(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""$$g = \frac{90}{\pi^2} a \left( \frac{T}{T_0} \right)^{\mu - 4}"""
        g_s = 90/np.pi**2 * self.a_s * (temp / self.t_ref)**(self.mu_s - 4)
        g_b = 90/np.pi**2 * self.a_b * (temp / self.t_ref)**(self.mu_b - 4)
        return g_s * phase + g_b * (1 - phase)

    def ge(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.g(temp, phase)

    def gs(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.g(temp, phase)
