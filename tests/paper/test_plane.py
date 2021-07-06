import os.path
import unittest

import numpy as np
import scipy.integrate as spi

import pttools.type_hints as th
from pttools import bubble
from tests.test_utils import TEST_DATA_PATH


class TestPlane(unittest.TestCase):
    @staticmethod
    def test_plane_odeint():
        validate_plane(odeint=True, method=spi.LSODA)

    @staticmethod
    def test_plane_rk45():
        validate_plane(method=spi.RK45, rtol=1.95e-3)

    @staticmethod
    def test_plane_rk23():
        validate_plane(method=spi.RK23, rtol=2.11e-2)

    @staticmethod
    def test_plane_dop853():
        validate_plane(method=spi.DOP853, rtol=2.1e-4)

    @staticmethod
    @unittest.expectedFailure
    def test_plane_radau():
        validate_plane(method=spi.Radau)

    @staticmethod
    def test_plane_bdf():
        validate_plane(method=spi.BDF, rtol=5e-3)

    @staticmethod
    @unittest.expectedFailure
    def test_plane_lsoda():
        validate_plane(method=spi.LSODA)


def validate_plane(odeint: bool = False, method: th.ODE_SOLVER = spi.LSODA, rtol: float = 1e-7):
    data = xiv_plane(odeint, method)
    data_summed = np.sum(data, axis=2)
    file_path = os.path.join(TEST_DATA_PATH, "xi-v_plane.txt")

    # Generate new reference data
    # np.savetxt(file_path, data_summed)

    data_ref = np.loadtxt(file_path)
    np.testing.assert_allclose(data_summed, data_ref, rtol=rtol)


def xiv_plane(odeint: bool = False, method: th.ODE_SOLVER = spi.LSODA):
    """Slightly modified copy-paste from sound-shell-model/paper/python/fig_8r_xi-v_plane.py"""
    # parametric integration limits
    tau_forwards_end = 100.0
    tau_backwards_end = -100.0

    # Define a suitable number of default lines to plot
    my_range = 9
    xi0_step = 1 / (my_range + 1)
    xi0_array = np.linspace(xi0_step, 1 - xi0_step, my_range)

    lst_deflag_v_b = []
    lst_deflag_w_b = []
    lst_deflag_xi_b = []
    lst_deflag_v_f = []
    lst_deflag_w_f = []
    lst_deflag_xi_f = []
    for n, xi0 in enumerate(xi0_array):
        # Make lines starting from v = xi, forward and back
        deflag_v_b, deflag_w_b, deflag_xi_b, _ = bubble.fluid_integrate_param(
            xi0, 1, xi0, t_end=tau_backwards_end, n_xi=1000, odeint=odeint, method=method)
        deflag_v_f, deflag_w_f, deflag_xi_f, _ = bubble.fluid_integrate_param(
            xi0, 1, xi0, t_end=tau_forwards_end, n_xi=1000, odeint=odeint, method=method)
        # Grey out parts of line which are unphysical
        unphysical = np.logical_and(
            deflag_v_b - bubble.v_shock(deflag_xi_b) < 0,
            deflag_v_b - bubble.lorentz(deflag_xi_b, bubble.CS0) > 0)
        # But let's keep the unphysical points to look at
        deflag_v_b_grey = deflag_v_b[unphysical]
        deflag_xi_b_grey = deflag_xi_b[unphysical]
        deflag_v_b[unphysical] = np.nan
        # Unphysical doesn't quite work for last points, so ...
        deflag_v_b_grey[deflag_v_b_grey < 1e-4] = np.nan

        lst_deflag_v_b.append(deflag_v_b)
        lst_deflag_w_b.append(deflag_w_b)
        lst_deflag_xi_b.append(deflag_xi_b)
        lst_deflag_v_f.append(deflag_v_f)
        lst_deflag_w_f.append(deflag_w_f)
        lst_deflag_xi_f.append(deflag_xi_f)

    return np.array([lst_deflag_v_b, lst_deflag_w_b, lst_deflag_xi_b, lst_deflag_v_f, lst_deflag_w_f, lst_deflag_xi_f])


if __name__ == "__main__":
    unittest.main()
