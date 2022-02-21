"""
Effective degrees of freedom for the Standard Model as a function of temperature.
Cubic spline interpolation for the range $0 - 10^{5.45}$ MeV from the table S2 of
:borsanyi_2016:.

.. plot:: fig/thermo.py

"""

import abc

import numpy as np
from scipy import interpolate

import pttools.type_hints as th


class ThermoModel(abc.ABC):
    @abc.abstractmethod
    def grho(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        pass

    @abc.abstractmethod
    def gs(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        pass


class StandardModel(ThermoModel):
    # Copied from the ArXiv file som_eos.tex
    GEFF_DATA = np.array([[
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
    ]]).T
    GEFF_DATA_TEMP = 10 ** GEFF_DATA[0, :]
    GEFF_DATA_GRHO = GEFF_DATA[1, :]
    GEFF_DATA_GRHO_GS_RATIO = GEFF_DATA[2, :]
    GEFF_DATA_GS = GEFF_DATA_GRHO / GEFF_DATA_GRHO_GS_RATIO
    # s=smoothing.
    # It's not mentioned in the article, so it's disabled to ensure that the error limits of the article hold.
    GRHO_SPLINE = interpolate.splrep(GEFF_DATA[0, :], GEFF_DATA_GRHO, s=0)
    GS_SPLINE = interpolate.splrep(GEFF_DATA[0, :], GEFF_DATA_GS, s=0)
    GRHO_GS_RATIO_SPLINE = interpolate.splrep(GEFF_DATA[0, :], GEFF_DATA_GRHO_GS_RATIO, s=0)

    def grho(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        Effective degrees of freedom for the energy density $g_{eff,\rho}(T)$

        :param temp: temperature $T$ (MeV)
        :return: $g_{eff,\rho}$
        """
        return interpolate.splev(np.log10(temp), self.GRHO_SPLINE)

    def gs(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        r"""
        Effective degrees of freedom for the entropy density, $g_{eff,s}(T)$

        :param temp: temperature $T$ (MeV)
        :return: $g_{eff,\rho}$
        """
        return interpolate.splev(np.log10(temp), self.GS_SPLINE)

    def grho_gs_ratio(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        return interpolate.splev(np.log10(temp), self.GRHO_GS_RATIO_SPLINE)
