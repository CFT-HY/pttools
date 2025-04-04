"""Base class for thermodynamics models"""

import abc
import logging

import numba
from numba.extending import overload
import numpy as np
import scipy.interpolate

from pttools.bubble.boundary import Phase
from pttools.models.base import BaseModel
from pttools.speedup.overload import np_all_fix
import pttools.type_hints as th

logger = logging.getLogger(__name__)


class ThermoModel(BaseModel, abc.ABC):
    """The thermodynamics model characterizes the particle physics of interest"""
    # TODO: Some functions seem to return vertical arrays. Fix this!

    #: Container for the log10 temperatures of $g_\text{eff}$ data
    GEFF_DATA_LOG_TEMP: np.ndarray
    #: Container for the temperatures of $g_\text{eff}$ data
    GEFF_DATA_TEMP: np.ndarray

    # Concrete methods

    def __init__(
            self,
            name: str = None,
            T_min: float = None, T_max: float = None,
            restrict_to_valid: bool = True,
            label_latex: str = None,
            label_unicode: str = None,
            gen_cs2: bool = True,
            gen_cs2_neg: bool = False,
            silence_temp: bool = False):

        temp_data_min = np.min(self.GEFF_DATA_TEMP)
        temp_data_max = np.max(self.GEFF_DATA_TEMP)
        if T_min is None:
            self.t_min = np.min(self.GEFF_DATA_TEMP)
        elif T_min < temp_data_min:
            raise ValueError("Model must have spline data for its validity range.")

        if T_max is None:
            self.t_max = np.max(self.GEFF_DATA_TEMP)
        elif T_max > temp_data_max:
            raise ValueError("ThermoModel must have spline data for its validity range.")

        super().__init__(
            name=name,
            T_min=T_min, T_max=T_max,
            restrict_to_valid=restrict_to_valid,
            label_latex=label_latex, label_unicode=label_unicode,
            gen_cs2=gen_cs2, gen_cs2_neg=gen_cs2_neg,
            silence_temp=silence_temp
        )

    def validate_cs2(self, cs2: np.ndarray, name: str) -> bool:
        """Validate that $0 < c_s^2 < 1$"""
        err = []
        if np.any(cs2 < 0):
            err.append("cannot be negative")
        if np.any(cs2 > 1):
            err.append("cannot exceed 1")
        if err:
            msg = ", ".join(err)
            logger.error(f"Invalid {name} for {self.name}: {msg}, got range: {np.min(cs2)} - {np.max(cs2)}")
            return False
        return True

    def gen_cs2(self) -> th.CS2Fun:
        cs2_s = self.cs2_full(self.GEFF_DATA_TEMP, Phase.SYMMETRIC)
        cs2_b = self.cs2_full(self.GEFF_DATA_TEMP, Phase.BROKEN)
        self.validate_cs2(cs2_s, "cs2_s")
        self.validate_cs2(cs2_b, "cs2_b")
        # with np.printoptions(threshold=np.inf):
        #     print(np.array([self.GEFF_DATA_TEMP, cs2_s]).T)

        cs2_spl_s = scipy.interpolate.splrep(
            np.log10(self.GEFF_DATA_TEMP),
            cs2_s,
            k=1
        )
        cs2_spl_b = scipy.interpolate.splrep(
            np.log10(self.GEFF_DATA_TEMP),
            cs2_b,
            k=1
        )

        t_min = self.t_min
        t_max = self.t_max

        @numba.njit
        def cs2_compute(temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            if np_all_fix(phase == Phase.SYMMETRIC.value):
                return scipy.interpolate.splev(np.log10(temp), cs2_spl_s)
            if np_all_fix(phase == Phase.BROKEN.value):
                return scipy.interpolate.splev(np.log10(temp), cs2_spl_b)
            return scipy.interpolate.splev(np.log10(temp), cs2_spl_b) * phase \
                + scipy.interpolate.splev(np.log10(temp), cs2_spl_s) * (1 - phase)

            # if np.any(ret > 1):
            #     with numba.objmode:
            #         logger.warning("Got cs2 > 1: %s", np.max(ret))
            # if np.any(ret < 0):
            #     with numba.objmode:
            #         logger.warning("Got cs2 < 0: %s", np.min(ret))
            # return ret

        @numba.njit
        def cs2_scalar_temp(temp: float, phase: th.FloatOrArr) -> th.FloatOrArr:
            if temp < t_min or temp > t_max:
                return np.nan
            return cs2_compute(temp, phase)

        @numba.njit
        def cs2_arr_temp(temp: np.ndarray, phase: th.FloatOrArr) -> np.ndarray:
            # This check somehow fixes a compilation bug in Numba 0.60.0
            if np.isscalar(temp):
                raise TypeError
            invalid = np.logical_or(temp < t_min, temp > t_max)
            if np.any(invalid):
                temp2 = temp.copy()
                temp2[invalid] = np.nan
            return cs2_compute(temp, phase)

        def cs2(temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            """The validate_temp function cannot be called from jitted functions,
            and therefore we have to use the validate_temp"""
            if isinstance(temp, float):
                return cs2_scalar_temp(temp, phase)
            if isinstance(temp, np.ndarray):
                if not temp.ndim:
                    return cs2_scalar_temp(temp.item(), phase)
                return cs2_arr_temp(temp, phase)
            raise TypeError(f"Unknown type for temp")

        @overload(cs2, jit_options={"nopython": True})
        def cs2_numba(temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArrNumba:
            if isinstance(temp, numba.types.Float):
                return cs2_scalar_temp
            if isinstance(temp, numba.types.Array):
                return cs2_arr_temp
            raise TypeError(f"Unknown type for temp: {type(temp)}")

        return cs2

    def cs2(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        Sound speed squared, $c_s^2$, interpolated from precomputed values.
        Takes in $T$ instead of $w$, unlike the equation of state model.

        :param temp: temperature $T$ (MeV)
        :param phase: phase $phi$
        :return: $c_s^2$
        """
        raise RuntimeError("The cs2(T, phase) function has not yet been loaded")

    def cs2_neg(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return -self.cs2(temp, phase)

    def cs2_full(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        """Full evaluation of $c_s^2$ from the underlying quantities"""
        # This hopefully reduces numerical errors
        return (self.dgp_dT(temp, phase)*temp + 4*self.gp(temp, phase)) / \
               (3 * (self.dge_dT(temp, phase)*temp + 4*self.ge(temp, phase)))
        # return self.dp_dt(temp, phase) / self.de_dt(temp, phase)

    def ge_gs_ratio(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.ge(temp, phase) / self.gs(temp, phase)

    def dgp_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return 4*self.dgs_dT(temp, phase) - 3*self.dge_dT(temp, phase)

    def dp_dt(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        $\frac{dp}{dT}$
        """
        return np.pi**2/90 * (self.dgp_dT(temp, phase) * temp**4 + 4*self.gp(temp, phase)*temp**3)

    def de_dt(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        $\frac{de}{dT}$
        """
        return np.pi**2/30 * (self.dge_dT(temp, phase) * temp**4 + 4*self.ge(temp, phase)*temp**3)

    def gp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Effective degrees of freedom for pressure, $g_{\text{eff},p}(T,\phi)$
        $$g_{\text{eff},p}(T,\phi) = 4g_s(T,\phi) - 3g_e(T,\phi)$$
        """
        # + \frac{90 V(\phi)}{\pi^2 T^4}
        self.validate_temp(temp)
        return 4*self.gs(temp, phase) - 3*self.ge(temp, phase)

    # Abstract methods

    @abc.abstractmethod
    def dge_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        $\frac{dg_e}{dT}$
        """

    @abc.abstractmethod
    def dgs_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        $\frac{dg_s}{dT}$
        """

    @abc.abstractmethod
    def ge(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        Effective degrees of freedom for the energy density $g_{\text{eff},e}(T)$

        :param temp: temperature $T$ (MeV)
        :param phase: phase $\phi$
        :return: $g_{\text{eff},e}$
        """

    @abc.abstractmethod
    def gs(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        Effective degrees of freedom for the entropy density, $g_{\text{eff},s}(T)$

        :param temp: temperature $T$ (MeV)
        :param phase: phase $\phi$
        :return: $g_{\text{eff},s}$
        """
