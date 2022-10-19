import abc
import logging

import numba
import numpy as np
import scipy.interpolate

from pttools.bubble.bag import CS2Fun
from pttools.bubble.boundary import Phase
from pttools.models.base import BaseModel
import pttools.type_hints as th

logger = logging.getLogger(__name__)


class ThermoModel(BaseModel, abc.ABC):
    """
    The thermodynamic model characterizes the particle physics of interest.

    TODO: Some functions seem to return vertical arrays. Fix this!
    """
    #: Container for the log10 temperatures of $g_\text{eff}$ data
    GEFF_DATA_LOG_TEMP: np.ndarray
    #: Container for the temperatures of $g_\text{eff}$ data
    GEFF_DATA_TEMP: np.ndarray

    # Concrete methods

    def gen_cs2(self) -> CS2Fun:
        cs2_s = self.cs2_full(self.GEFF_DATA_TEMP, Phase.SYMMETRIC)
        cs2_b = self.cs2_full(self.GEFF_DATA_TEMP, Phase.BROKEN)
        if np.any(cs2_s < 0):
            raise ValueError("cs2_s cannot be negative")
        if np.any(cs2_s > 1):
            raise ValueError("cs2_s cannot exceed 1")
        if np.any(cs2_b < 0):
            raise ValueError("cs2_b cannot be negative")
        if np.any(cs2_b > 1):
            raise ValueError("cs2_b cannot exceed 1")

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
            if np.all(phase == Phase.SYMMETRIC.value):
                return scipy.interpolate.splev(np.log10(temp), cs2_spl_s)
            if np.all(phase == Phase.BROKEN.value):
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
            invalid = np.logical_or(temp < t_min, temp > t_max)
            if np.any(invalid):
                temp2 = temp.copy()
                temp2[invalid] = np.nan
            return cs2_compute(temp, phase)

        @numba.generated_jit(nopython=True)
        def cs2(temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArrNumba:
            """The validate_temp function cannot be called from jitted functions,
            and therefore we have to use the validate_temp"""
            if isinstance(temp, numba.types.Float):
                return cs2_scalar_temp
            if isinstance(temp, numba.types.Array):
                return cs2_arr_temp
            if isinstance(temp, float):
                return cs2_scalar_temp(temp, phase)
            if isinstance(temp, np.ndarray):
                if not temp.ndim:
                    return cs2_scalar_temp(temp.item(), phase)
                return cs2_arr_temp(temp, phase)
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

    def cs2_full(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        """Full evaluation of $c_s^2$ from the underlying quantities"""
        return self.dp_dt(temp, phase) / self.de_dt(temp, phase)

    def dp_dt(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        $\frac{dp}{dT}$

        TODO: it may be necessary to use gp instead of gs
        """
        return np.pi**2/90 * (self.dgs_dT(temp, phase) * temp ** 4 + 4 * self.gs(temp, phase)*temp**3)

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
        # TODO: Check that this is correct
        return (4*self.gs(temp, phase) - self.ge(temp, phase))/3  # + (90*self.V(phase)) / (np.pi**2 * temp**4)

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
