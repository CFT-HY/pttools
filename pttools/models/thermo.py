import abc

import numpy as np

import pttools.type_hints as th


class ThermoModel(abc.ABC):
    """
    The thermodynamic model characterizes the particle physics of interest.
    """

    def cs2(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        pass

    @abc.abstractmethod
    def dge_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        pass

    @abc.abstractmethod
    def dgs_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        pass

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
        # TODO: use gp instead of gs, at least for the latter part. And check the units!
        return np.pi**2/30 * (self.dgs_dT(temp, phase) * temp ** 4 + 4 * self.gs(temp, phase)*temp**3)

    def dp_dt(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return np.pi**2/90 * (self.dge_dT(temp, phase) * temp**4 + 4*self.ge(temp, phase)*temp**3)
