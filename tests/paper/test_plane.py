import os.path
import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi

import pttools.type_hints as th
from tests.test_utils import TEST_DATA_PATH
from tests.paper import plane
from tests.paper import plot_plane

PLOT = True


class TestPlane(unittest.TestCase):
    @staticmethod
    def test_plane_bdf():
        validate_plane(method=spi.BDF, rtol=5e-3)

    @staticmethod
    def test_plane_dop853():
        validate_plane(method=spi.DOP853, rtol=2.1e-4)

    @staticmethod
    @unittest.expectedFailure
    def test_plane_lsoda():
        validate_plane(method=spi.LSODA)

    @staticmethod
    def test_plane_odeint():
        validate_plane(odeint=True, method=spi.LSODA)

    @staticmethod
    @unittest.expectedFailure
    def test_plane_radau():
        validate_plane(method=spi.Radau)

    @staticmethod
    def test_plane_rk23():
        validate_plane(method=spi.RK23, rtol=2.11e-2)

    @staticmethod
    def test_plane_rk45():
        validate_plane(method=spi.RK45, rtol=1.95e-3)


def validate_plane(odeint: bool = False, method: th.ODE_SOLVER = spi.LSODA, rtol: float = 1e-7):
    data = plane.xiv_plane(odeint, method)

    if PLOT:
        plot_plane.plot_plane(data)

    data_summed = np.sum(data, axis=2)
    file_path = os.path.join(TEST_DATA_PATH, "xi-v_plane.txt")

    # Generate new reference data
    # np.savetxt(file_path, data_summed)

    data_ref = np.loadtxt(file_path)
    np.testing.assert_allclose(data_summed, data_ref, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
    # if PLOT:
    #     plt.show()
