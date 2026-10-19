r"""Compare with the results of Correia et al. (2025).

Compare with the results of :correia_2025:`\ `.
"""

import unittest

import numpy as np
from numpy.testing import assert_array_less
import pytest

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.omgw0 import Spectrum
from pttools.type_hints import FloatArr1D, FloatArr2D

#: :correia_2025:`\ ` tables I-V & VII-IX
CHRW25_DATA: FloatArr2D = np.array([
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
CHRW25_ALPHA_N: FloatArr1D = CHRW25_DATA[0, :]
CHRW25_V_WALL: FloatArr1D = CHRW25_DATA[1, :]
CHRW25_UBARF: FloatArr1D = CHRW25_DATA[6, :]


class CHRW25Test(unittest.TestCase):
    """Comparison of the Sound Shell Model with the simulation results of Correia et al. (2025)."""

    @staticmethod
    @pytest.mark.xfail(reason="The Sound Shell Model may not be applicable in this regime.")
    def test_ubarf() -> None:
        """Test that the SSM RMS fluid velocity is below the simulated maximum compressional velocity."""
        model = BagModel()
        bubbles = [
            Bubble(model, v_wall=v_wall, alpha_n=alpha_n)
            for v_wall, alpha_n in zip(CHRW25_V_WALL, CHRW25_ALPHA_N, strict=False)
        ]
        spectra = [Spectrum(bubble) for bubble in bubbles]
        ubarf = [spectrum.ubarf for spectrum in spectra]
        assert_array_less(ubarf, CHRW25_UBARF)


if __name__ == "__main__":
    unittest.main()
