import os.path
import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi

import pttools.type_hints as th
from tests.test_utils import TEST_DATA_PATH, TEST_FIGURE_PATH
from tests.paper import plane
from tests.paper import plot_plane

PLOT = True
os.makedirs(TEST_FIGURE_PATH, exist_ok=True)


class TestPlane(unittest.TestCase):
    figure: plt.Figure
    axs: np.ndarray

    @classmethod
    def setUpClass(cls) -> None:
        grid_shape = (3, 3)
        if PLOT:
            cls.figure, cls.axs = plt.subplots(*grid_shape, figsize=(16.5, 11.7))
        else:
            cls.axs = np.zeros(grid_shape)

    @classmethod
    def tearDownClass(cls) -> None:
        if PLOT:
            cls.figure.tight_layout()
            path = os.path.join(TEST_FIGURE_PATH, "integrators")
            for fmt in ("pdf", "png", "svg"):
                cls.figure.savefig(f"{path}.{fmt}")
            # plt.show()

    def test_plane_bdf(self):
        validate_plane(method=spi.BDF, rtol=5e-3, ax=self.axs[2, 0])

    def test_plane_dop853(self):
        validate_plane(method=spi.DOP853, rtol=2.1e-4, ax=self.axs[2, 1])

    @unittest.expectedFailure
    def test_plane_lsoda(self):
        validate_plane(method=spi.LSODA, ax=self.axs[0, 1])

    def test_plane_odeint(self):
        validate_plane(odeint=True, method=spi.LSODA, ax=self.axs[0, 0])

    @unittest.expectedFailure
    def test_plane_radau(self):
        validate_plane(method=spi.Radau, ax=self.axs[2, 2])

    def test_plane_rk23(self):
        validate_plane(method=spi.RK23, rtol=2.11e-2, ax=self.axs[1, 0])

    def test_plane_rk45(self):
        validate_plane(method=spi.RK45, rtol=1.95e-3, ax=self.axs[1, 1])


def validate_plane(odeint: bool = False, method: th.ODE_SOLVER = spi.LSODA, rtol: float = 1e-7, ax: plt.Axes = None):
    data = plane.xiv_plane(odeint, method)

    if ax:
        plot_plane.plot_plane(ax, data, method, odeint)

    data_summed = np.sum(data, axis=2)
    file_path = os.path.join(TEST_DATA_PATH, "xi-v_plane.txt")

    # Generate new reference data
    # np.savetxt(file_path, data_summed)

    data_ref = np.loadtxt(file_path)
    np.testing.assert_allclose(data_summed, data_ref, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
