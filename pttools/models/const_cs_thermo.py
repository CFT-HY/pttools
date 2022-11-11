import typing as tp

import numpy as np

import pttools.type_hints as th
from .thermo import ThermoModel
from . import const_cs


class ConstCSThermoModel(ThermoModel):
    DEFAULT_LABEL = "ConstCSThermoModel"
    DEFAULT_NAME = "const_cs_thermo"

    GEFF_DATA_LOG_TEMP = np.linspace(-1, 3, 1000)
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
        self.V_s = V_s
        self.V_b = V_b
        self.t_ref = t_ref
        self.css2 = css2
        self.csb2 = csb2
        self.css = np.sqrt(css2)
        self.csb= np.sqrt(csb2)
        self.mu_s = const_cs.cs2_to_mu(css2)
        self.mu_b = const_cs.cs2_to_mu(csb2)
        # TODO: Generate reference values for g0 here (corresponding to a_s, a_b)

        super().__init__(
            t_min=t_min, t_max=t_max,
            name=name, label=label
        )

    def dge_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        dge_s = 30/np.pi**2 * (
            (self.mu_s - 1) * (self.mu_s - 4) * self.a_s * self.t_ref**(4 - self.mu_s) * temp**(self.mu_s - 5)
            - 4*self.V_s/temp**5
        )
        dge_b = 30/np.pi**2 * (
            (self.mu_b - 1) * (self.mu_b - 4) * self.a_b * self.t_ref**(4 - self.mu_b) * temp**(self.mu_b - 5)
            - 4*self.V_b/temp**5
        )
        return dge_b * phase + dge_s * (1 - phase)

    def dgs_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        dgs_s = 45/(2*np.pi**2) * self.mu_s * (self.mu_s - 4) * self.a_s * self.t_ref**(4 - self.mu_s) * temp**(self.mu_s - 5)
        dgs_b = 45/(2*np.pi**2) * self.mu_b * (self.mu_b - 4) * self.a_b * self.t_ref**(4 - self.mu_b) * temp**(self.mu_b - 5)
        return dgs_b * phase + dgs_s * (1 - phase)

    def export(self) -> tp.Dict[str, any]:
        return {
            **super().export(),
            "t_ref": self.t_ref,
            "css2": self.css2,
            "csb2": self.csb2,
            "mu_s": self.mu_s,
            "mu_b": self.mu_b
        }

    def ge(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        ge_s = 30/np.pi**2 * (
            (self.mu_s - 1) * self.a_s * (temp / self.t_ref) ** (self.mu_s - 4)
            + self.V_s / temp**4
        )
        ge_b = 30/np.pi**2 * (
            (self.mu_b - 1) * self.a_b * (temp / self.t_ref) ** (self.mu_b - 4)
            + self.V_b / temp**4
        )
        return ge_b * phase + ge_s * (1 - phase)

    def gs(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        gs_s = 45/(2*np.pi**2) * self.a_s * self.mu_s * (temp / self.t_ref)**(self.mu_s - 4)
        gs_b = 45/(2*np.pi**2) * self.a_b * self.mu_b * (temp / self.t_ref)**(self.mu_b - 4)
        return gs_b * phase + gs_s * (1 - phase)
