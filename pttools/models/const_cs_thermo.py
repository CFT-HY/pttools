"""ThermoModel-based constant $c_s$ model."""

import typing as tp

import numpy as np

from pttools.models.const_cs import ConstCSModel, cs2_to_mu
from pttools.models.export import CONST_CS_THERMO_MODEL_FIELDS
from pttools.models.thermo import ThermoModel
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr
from pttools.utils.fields import Fields


class ConstCSThermoModel(ThermoModel):
    """ThermoModel-based constant $c_s$ model."""

    DEFAULT_LABEL_LATEX = "Constant $c_s$ thermo-model"
    DEFAULT_LABEL_UNICODE = "Constant cₛ thermo-model"
    DEFAULT_NAME = "const_cs_thermo"
    FIELDS: tp.ClassVar[Fields] = CONST_CS_THERMO_MODEL_FIELDS
    TEMPERATURE_IS_PHYSICAL = False

    GEFF_DATA_LOG_TEMP = np.linspace(-1, 3, 1000)
    GEFF_DATA_TEMP = 10**GEFF_DATA_LOG_TEMP

    def __init__(
            self,
            a_s: float, a_b: float,
            css2: float, csb2: float,
            V_s: float, V_b: float = 0,
            T_min: float | None = None,
            T_max: float | None = None,
            t_ref: float = 1,
            name: str | None = None,
            label_latex: str | None = None,
            label_unicode: str | None = None,
            allow_invalid: bool = False):
        r"""Initialize the constant sound speed thermodynamics model.

        The parameters are validated by creating a corresponding :class:`pttools.models.const_cs.ConstCSModel`.

        :param a_s: $a_s$, prefactor of $p$ in the symmetric phase
        :param a_b: $a_b$, prefactor of $p$ in the broken phase
        :param css2: $c_{s,s}^2$, speed of sound squared in the symmetric phase
        :param csb2: $c_{s,b}^2$, speed of sound squared in the broken phase
        :param V_s: $V_s$, the potential term of $p$ in the symmetric phase
        :param V_b: $V_b$, the potential term of $p$ in the broken phase
        :param T_min: $T_\text{min}$, minimum temperature at which the model is valid
        :param T_max: $T_\text{max}$, maximum temperature at which the model is valid
        :param t_ref: $T_\text{ref}$, reference temperature
        :param name: custom name for the model
        :param label_latex: custom LaTeX label for the model
        :param label_unicode: custom Unicode label for the model
        :param allow_invalid: whether to allow invalid parameters in the validation
        """
        # For validation
        ConstCSModel(css2=css2, csb2=csb2, V_s=V_s, V_b=V_b, a_s=a_s, a_b=a_b, allow_invalid=allow_invalid)

        self.a_s: float = a_s
        self.a_b: float = a_b
        self.V_s: float = V_s
        self.V_b: float = V_b
        self.t_ref: float = t_ref
        self.css2: float = css2
        self.csb2: float = csb2
        self.css: float = np.sqrt(css2)
        self.csb: float = np.sqrt(csb2)
        self.mu_s: float = cs2_to_mu(css2)
        self.mu_b: float = cs2_to_mu(csb2)
        # TODO: Generate reference values for g0 here (corresponding to a_s, a_b)

        super().__init__(
            T_min=T_min, T_max=T_max,
            name=name, label_latex=label_latex, label_unicode=label_unicode
        )

    @tp.override
    def dge_dT[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        dge_s = 30/np.pi**2 * (
            (self.mu_s - 1) * (self.mu_s - 4) * self.a_s * self.t_ref**(4 - self.mu_s) * temp**(self.mu_s - 5)
            - 4*self.V_s/temp**5
        )
        dge_b = 30/np.pi**2 * (
            (self.mu_b - 1) * (self.mu_b - 4) * self.a_b * self.t_ref**(4 - self.mu_b) * temp**(self.mu_b - 5)
            - 4*self.V_b/temp**5
        )
        return dge_b * phase + dge_s * (1 - phase)

    @tp.override
    def dgs_dT[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        dgs_s = \
            45/(2*np.pi**2) * self.mu_s * (self.mu_s - 4) * self.a_s * \
            self.t_ref**(4 - self.mu_s) * temp**(self.mu_s - 5)
        dgs_b = \
            45/(2*np.pi**2) * self.mu_b * (self.mu_b - 4) * self.a_b * \
            self.t_ref**(4 - self.mu_b) * temp**(self.mu_b - 5)
        return dgs_b * phase + dgs_s * (1 - phase)

    @tp.override
    def ge[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        ge_s = 30/np.pi**2 * (
            (self.mu_s - 1) * self.a_s * (temp / self.t_ref) ** (self.mu_s - 4)
            + self.V_s / temp**4
        )
        ge_b = 30/np.pi**2 * (
            (self.mu_b - 1) * self.a_b * (temp / self.t_ref) ** (self.mu_b - 4)
            + self.V_b / temp**4
        )
        return tp.cast(T, ge_b * phase + ge_s * (1 - phase))

    @tp.override
    def gs[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        gs_s = 45/(2*np.pi**2) * self.a_s * self.mu_s * (temp / self.t_ref)**(self.mu_s - 4)
        gs_b = 45/(2*np.pi**2) * self.a_b * self.mu_b * (temp / self.t_ref)**(self.mu_b - 4)
        return tp.cast(T, gs_b * phase + gs_s * (1 - phase))
