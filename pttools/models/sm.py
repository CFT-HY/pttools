"""
Effective degrees of freedom for the Standard Model as a function of temperature.
Cubic spline interpolation for the range $0 - 10^{5.45}$ MeV from the table S2 of
:borsanyi_2016:.

.. plot:: fig/standard_model.py

"""

import numpy as np
from scipy import interpolate

import pttools.type_hints as th
from pttools.models.thermo import ThermoModel


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
    # MIN_TEMP = GEFF_DATA_TEMP[0]
    # MAX_TEMP = GEFF_DATA_TEMP[-1]
    GEFF_DATA_GE = GEFF_DATA[1, :]
    GEFF_DATA_GE_GS_RATIO = GEFF_DATA[2, :]
    GEFF_DATA_GS = GEFF_DATA_GE / GEFF_DATA_GE_GS_RATIO
    # s=smoothing.
    # It's not mentioned in the article, so it's disabled to ensure that the error limits of the article hold.
    GE_SPLINE = interpolate.splrep(GEFF_DATA[0, :], GEFF_DATA_GE, s=0)
    GS_SPLINE = interpolate.splrep(GEFF_DATA[0, :], GEFF_DATA_GS, s=0)
    GE_GS_RATIO_SPLINE = interpolate.splrep(GEFF_DATA[0, :], GEFF_DATA_GE_GS_RATIO, s=0)

    def dge_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return 1/(np.log(10)*temp) * interpolate.splev(np.log10(temp), self.GE_SPLINE, der=1)

    def dgs_dT(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return 1/(np.log(10)*temp) * interpolate.splev(np.log10(temp), self.GS_SPLINE, der=1)

    def ge(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return interpolate.splev(np.log10(temp), self.GE_SPLINE)

    def gs(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return interpolate.splev(np.log10(temp), self.GS_SPLINE)

    def ge_gs_ratio(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        return interpolate.splev(np.log10(temp), self.GE_GS_RATIO_SPLINE)


if __name__ == "__main__":
    sm = StandardModel()
    print(type(sm.GE_SPLINE))
    for elem in sm.GE_SPLINE:
        print(elem)
