import abc

import pttools.type_hints as th


class ThermoModel(abc.ABC):
    """
    The thermodynamic model characterizes the particle physics of interest.
    """

    @abc.abstractmethod
    def ge(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        Effective degrees of freedom for the energy density $g_{eff,e}(T)$

        :param temp: temperature $T$ (MeV)
        :return: $g_{eff,e}$
        """

    @abc.abstractmethod
    def gs(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        Effective degrees of freedom for the entropy density, $g_{eff,s}(T)$

        :param temp: temperature $T$ (MeV)
        :return: $g_{eff,s}$
        """
        pass

    @abc.abstractmethod
    def cs2(self):
        pass
