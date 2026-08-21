r"""Compare with the results of Correia et al. (2026)

Compare with the results of :correia_2026:`\ `.
"""

import unittest
import pytest

import numpy as np
from numpy.testing import assert_array_less

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.omgw0 import Spectrum
from pttools.type_hints import FloatArr1D, FloatArr2D

#: :correia_2026:`\ ` tables I-V & VII-IX
CORREIA_2026_DATA: FloatArr2D = np.array([
    [0.67, 0.50],    # alpha_n
    [0.92, 0.44],    # v_wall
    [0.23, 0.25],    # T_n / T_c
    [0.15, 3.5],     # \eta / T_c
    [40, 100],       # Interval [T_c^{-1}]
    [1000, 4500],    # t_ref T_c
    [0.35, 0.21],    # max(U_compressional)
    [0.04, 0.09],    # max(U_vortical)
    [2920, 4958],    # t_sh[T_c^{-1}]
    [27535, 11865],  # t_ed[T_c^{-1}]
    [1.48, 0.86],    # Compressional kinetic energy \zeta
    [0.23, 0.01],    # Integral scale \lambda
    [0.140, 0.080],  # RMS 3-velocity, compressional
    [0.028, 0.050],  # RMS 3-velocity, vortical
    [0.61, 0.79],    # d_tot
    [2.40, 0.74],    # 2\zeta + 2\lambda - 1
    [0.017, 0.017]   # Fitted asymptotic density fraction, scaled
])
CORREIA_2026_ALPHA_N: FloatArr1D = CORREIA_2026_DATA[0, :]
CORREIA_2026_V_WALL: FloatArr1D = CORREIA_2026_DATA[1, :]
CORREIA_2026_UBARF: FloatArr1D = CORREIA_2026_DATA[6, :]


class CorreiaTest2026(unittest.TestCase):
    @staticmethod
    @pytest.mark.xfail(reason="The Sound Shell Model may not be applicable in this regime.")
    def test_ubarf():
        model = BagModel()
        bubbles = [
            Bubble(model, v_wall=v_wall, alpha_n=alpha_n)
            for v_wall, alpha_n in zip(CORREIA_2026_V_WALL, CORREIA_2026_ALPHA_N)
        ]
        spectra = [Spectrum(bubble) for bubble in bubbles]
        ubarf = [spectrum.ubarf for spectrum in spectra]
        assert_array_less(ubarf, CORREIA_2026_UBARF)


if __name__ == "__main__":
    unittest.main()
