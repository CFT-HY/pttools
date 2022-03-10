import abc

import numba
import numpy as np
import scipy.interpolate

from pttools.bubble.boundary import Phase
import pttools.type_hints as th


class ThermoModel(abc.ABC):
    """
    The thermodynamic model characterizes the particle physics of interest.
    """
    GEFF_DATA_TEMP: np.ndarray

    def __init__(self):
        self.cs2 = self._gen_cs2()

    def _gen_cs2(self) -> callable:
        cs2_spl_s = scipy.interpolate.splrep(
            np.log10(self.GEFF_DATA_TEMP),
            self.cs2_full(self.GEFF_DATA_TEMP, Phase.SYMMETRIC),
            k=1)
        cs2_spl_b = scipy.interpolate.splrep(
            np.log10(self.GEFF_DATA_TEMP),
            self.cs2_full(self.GEFF_DATA_TEMP, Phase.BROKEN),
            k=1)

        @numba.njit
        def cs2(temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            if phase == Phase.SYMMETRIC.value:
                return scipy.interpolate.splev(np.log10(temp), cs2_spl_s)
            elif phase == Phase.BROKEN.value:
                return scipy.interpolate.splev(np.log10(temp), cs2_spl_b)
            return scipy.interpolate.splev(np.log10(temp), cs2_spl_b) * phase \
                + scipy.interpolate.splev(np.log10(temp), cs2_spl_s) * (1 - phase)
        return cs2

    def cs2(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        Sound speed squared, $c_s^2$, interpolated from precomputed values.

        :param temp: temperature $T$ (MeV)
        :param phase: phase $phi$
        :return: $c_s^2$
        """
        raise RuntimeError("The cs2(T, phase) function has not yet been loaded")

    def cs2_full(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        """Full evaluation of $c_s^2$ from the underlying quantities"""
        return self.dp_dt(temp, phase) / self.de_dt(temp, phase)

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
        Effective degrees of freedom for the energy density $g_{eff,e}(T)$

        :param temp: temperature $T$ (MeV)
        :param phase: phase $\phi$
        :return: $g_{eff,e}$
        """

    @abc.abstractmethod
    def gs(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        Effective degrees of freedom for the entropy density, $g_{eff,s}(T)$

        :param temp: temperature $T$ (MeV)
        :param phase: phase $\phi$
        :return: $g_{eff,s}$
        """

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
