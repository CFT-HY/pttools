"""
Effective degrees of freedom for the Standard Model as a function of temperature.
Cubic spline interpolation for the range $0 - 10^{5.45}$ MeV from the table S2 of
`Borsanyi et al., 2016 <https://arxiv.org/abs/1606.07494>`_

.. plot:: fig/geff.py

"""
# TODO: Should this be a class so that other models could be implemented?

import numpy as np
from scipy import interpolate

import pttools.type_hints as th


# Copied from the ArXiv file som_eos.tex
DATA = np.array([[
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
DATA_TEMP = 10 ** DATA[0, :]
DATA_G_RHO = DATA[1, :]
DATA_GRHO_GS_RATIO = DATA[2, :]
# s=smoothing. It's not mentioned in the article, so it's disabled to ensure that the error limits of the article hold.
__g_rho_spline = interpolate.splrep(DATA[0, :], DATA_G_RHO, s=0)
__g_s_spline = interpolate.splrep(DATA[0, :], DATA_G_RHO / DATA_GRHO_GS_RATIO, s=0)
__grho_gs_ratio_spline = interpolate.splrep(DATA[0, :], DATA_GRHO_GS_RATIO, s=0)


def g_rho(temp: th.FloatOrArr):
    r"""
    Effective degrees of freedom for the energy density $g_{eff,\rho}(T)$

    :param temp: temperature $T$ (MeV)
    :return: $g_{eff,\rho}$
    """
    return interpolate.splev(np.log10(temp), __g_rho_spline)


def g_s(temp: th.FloatOrArr):
    r"""
    Effective degrees of freedom for the entropy density, $g_{eff,s}(T)$

    :param temp: temperature $T$ (MeV)
    :return: $g_{eff,\rho}$
    """
    return interpolate.splev(np.log10(temp), __g_s_spline)


def grho_gs_ratio(temp: th.FloatOrArr):
    return interpolate.splev(np.log10(temp), __grho_gs_ratio_spline)
