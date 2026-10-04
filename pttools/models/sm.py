"""Standard Model equation of state."""

import logging
import typing as tp

import numpy as np
from scipy import interpolate

from pttools.models.thermo import ThermoModel
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr

logger: logging.Logger = logging.getLogger(__name__)


class StandardModel(ThermoModel):
    r"""Standard Model equation of state.

    Based on cubic spline interpolation for the effective degrees of freedom of the Standard Model.
    Data range $0 - 10^{5.45}$ MeV from the table S2 of :borsanyi_2016:`\ `.
    Units are in MeV.
    """

    # Todo: Should the units be changed to GeV to be compatible with Spectrum?
    DEFAULT_LABEL_LATEX = "Standard Model"
    DEFAULT_LABEL_UNICODE = DEFAULT_LABEL_LATEX
    DEFAULT_NAME = "standard_model"
    TEMPERATURE_IS_PHYSICAL = True
    TEMPERATURE_UNIT_GEV = 1e-3

    # Copied from the ArXiv file som_eos.tex
    GEFF_DATA = np.array([
        [0.00, 10.71, 1.00228],
        [0.50, 10.74, 1.00029],
        [1.00, 10.76, 1.00048],
        [1.25, 11.09, 1.00505],
        [1.60, 13.68, 1.02159],
        [2.00, 17.61, 1.02324],
        [2.15, 24.07, 1.05423],
        [2.20, 29.84, 1.07578],
        [2.40, 47.83, 1.06118],
        [2.50, 53.04, 1.04690],
        [3.00, 73.48, 1.01778],
        [4.00, 83.10, 1.00123],
        [4.30, 85.56, 1.00389],
        [4.60, 91.97, 1.00887],
        [5.00, 102.17, 1.00750],
        [5.45, 104.98, 1.00023],
    ]).T
    GEFF_DATA_LOG_TEMP = GEFF_DATA[0, :]
    GEFF_DATA_TEMP = 10 ** GEFF_DATA[0, :]
    DEFAULT_T_MIN = GEFF_DATA_TEMP[0]
    DEFAULT_T_MAX = GEFF_DATA_TEMP[-1]
    GEFF_DATA_GE = GEFF_DATA[1, :]
    GEFF_DATA_GE_GS_RATIO = GEFF_DATA[2, :]
    GEFF_DATA_GS = GEFF_DATA_GE / GEFF_DATA_GE_GS_RATIO
    # s=smoothing.
    # It's not mentioned in the article, so it's disabled to ensure that the error limits of the article hold.
    GE_SPLINE = interpolate.splrep(GEFF_DATA_LOG_TEMP, GEFF_DATA_GE, s=0)
    GS_SPLINE = interpolate.splrep(GEFF_DATA_LOG_TEMP, GEFF_DATA_GS, s=0)
    GE_GS_RATIO_SPLINE = interpolate.splrep(GEFF_DATA_LOG_TEMP, GEFF_DATA_GE_GS_RATIO, s=0)

    def __init__(
            self,
            g_mult_s: float = 1,
            g_mult_b: float = 1,
            V_s: float = 0,
            V_b: float = 0,
            name: str | None = None,
            T_min: float | None = None,
            T_max: float | None = None,
            restrict_to_valid: bool = True,
            label_latex: str | None = None,
            label_unicode: str | None = None,
            gen_cs2: bool = True,
            silence_temp: bool = False):
        r"""Initialize the Standard Model equation of state.

        :param g_mult_s: multiplier for the degrees of freedom in the symmetric phase
        :param g_mult_b: multiplier for the degrees of freedom in the broken phase
        :param V_s: $V_s$, potential in the symmetric phase
        :param V_b: $V_b$, potential in the broken phase
        :param name: custom name for the model
        :param T_min: $T_\text{min}$, minimum temperature at which the model is valid
        :param T_max: $T_\text{max}$, maximum temperature at which the model is valid
        :param restrict_to_valid: whether temperatures outside the validity range are converted to NaN
        :param label_latex: custom LaTeX label for the model
        :param label_unicode: custom Unicode label for the model
        :param gen_cs2: whether to generate the $c_s^2$ function
        :param silence_temp: whether to suppress the logging of temperatures outside the validity range
        """
        logger.debug(
            "Creating Standard Model with g_mult_s=%s, g_mult_b=%s, V_s=%s, V_b=%s",
            g_mult_s, g_mult_b, V_s, V_b
        )

        self.g_mult_s: float = g_mult_s
        self.g_mult_b: float = g_mult_b
        self.V_s: float = V_s
        self.V_b: float = V_b

        super().__init__(
            T_min=T_min, T_max=T_max,
            restrict_to_valid=restrict_to_valid,
            name=name, label_latex=label_latex, label_unicode=label_unicode,
            gen_cs2=gen_cs2,
            silence_temp=silence_temp
        )

    @tp.override
    def dge_dT[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        self.validate_temp(temp)
        return 1/(np.log(10)*temp) * interpolate.splev(np.log10(temp), self.GE_SPLINE, der=1) * self.g_mult(phase) \
            - 120/np.pi**2 * self.V(phase)/temp**5

    @tp.override
    def dgs_dT[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        self.validate_temp(temp)
        return 1/(np.log(10)*temp) * interpolate.splev(np.log10(temp), self.GS_SPLINE, der=1) * self.g_mult(phase)

    @tp.override
    def ge[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        self.validate_temp(temp)
        return tp.cast(T, interpolate.splev(np.log10(temp), self.GE_SPLINE) * self.g_mult(phase)
                          + 30/np.pi**2 * self.V(phase) / temp**4)

    @tp.override
    def gs[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        self.validate_temp(temp)
        return tp.cast(T, interpolate.splev(np.log10(temp), self.GS_SPLINE) * self.g_mult(phase))

    @tp.override
    def ge_gs_ratio[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        self.validate_temp(temp)
        if self.g_mult_s == self.g_mult_b == 1 and self.V_s == self.V_b == 0:
            return tp.cast(T, interpolate.splev(np.log10(temp), self.GE_GS_RATIO_SPLINE))
        return tp.cast(T, self.ge(temp, phase) / self.gs(temp, phase))

    def g_mult[T: FloatOrArr](self, phase: T) -> T:
        r"""Multiplier for the degrees of freedom in the phase $\phi$.

        :param phase: phase $\phi$
        """
        return tp.cast(T, self.g_mult_b * phase + self.g_mult_s * (1 - phase))

    def V[T: FloatOrArr](self, phase: T) -> T:
        r"""Potential $V(\phi)$.

        :param phase: phase $\phi$
        """
        return tp.cast(T, self.V_b * phase + self.V_s * (1 - phase))


if __name__ == "__main__":
    sm = StandardModel()
    print("GE_SPLINE")
    print(type(sm.GE_SPLINE))
    for i, elem in enumerate(sm.GE_SPLINE):
        print(f"{i}:", elem)
    print("Test")
    print(sm.cs2(np.linspace(1, 2, 5), 1))
