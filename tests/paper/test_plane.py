import logging
import os.path
import shutil
import subprocess
import typing as tp
import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as spi

import pttools.type_hints as th
from tests.plotting import save_fig_multi
from tests.test_utils import TEST_DATA_PATH, TEST_FIGURE_PATH
from tests.paper import plane
from tests.paper import plot_plane

logger = logging.getLogger(__name__)

PLOT = True
os.makedirs(TEST_FIGURE_PATH, exist_ok=True)


class TestPlane(unittest.TestCase):
    grid_fig: plt.Figure
    axs: np.ndarray
    FIG_PATH = os.path.join(TEST_FIGURE_PATH, "integrators")

    @classmethod
    def setUpClass(cls) -> None:
        grid_shape = (3, 3)
        if PLOT:
            cls.grid_fig, cls.axs = plt.subplots(*grid_shape, figsize=(16.5, 11.7))
            cls.grid_fig.suptitle(r"Comparison of integrators for $\xi$-$v$-plane")
            cls.ref_data = plane.xiv_plane(odeint=True, method=spi.LSODA)

    @classmethod
    def tearDownClass(cls) -> None:
        if PLOT:
            cls.grid_fig.tight_layout()
            save_fig_multi(cls.grid_fig, cls.FIG_PATH)
            # plt.show()
            if shutil.which("ffmpeg"):
                video_path = f"{cls.FIG_PATH}.mp4"
                if os.path.exists(video_path):
                    os.remove(video_path)
                ret: subprocess.CompletedProcess = subprocess.run(
                    [
                        "ffmpeg",
                        "-framerate", "0.5",
                        "-pattern_type", "glob",
                        "-i", f"{cls.FIG_PATH}_*.png",
                        "-c:v", "libx264",
                        "-r", "30",
                        video_path
                    ],
                    check=False,
                    capture_output=True
                )
                # print(ret.stdout)
                # print(ret.stderr)
                ret.check_returncode()
            else:
                logger.warning("ffmpeg was not found, so the xi-v-plane animation could not be created")

    def validate_plane(
            self,
            odeint: bool = False,
            method: th.ODE_SOLVER = spi.LSODA,
            rtol: float = 1e-7,
            i: int = None,
            ax: tp.Tuple[int, int] = None):
        data = plane.xiv_plane(odeint, method)

        if ax:
            fig: plt.Figure = plt.figure()
            ax2: plt.Axes = fig.add_subplot()
            plot_plane.plot_plane(self.axs[ax[0], ax[1]], data, method, odeint, deflag_ref=self.ref_data)
            plot_plane.plot_plane(ax2, data, method, odeint, deflag_ref=self.ref_data)
            fig_name = f"{self.FIG_PATH}_{i}_{plot_plane.get_solver_name(method, odeint)}"
            save_fig_multi(fig, fig_name)

        data_summed = np.sum(data, axis=2)
        file_path = os.path.join(TEST_DATA_PATH, "xi-v_plane.txt")

        # Generate new reference data
        # np.savetxt(file_path, data_summed)

        data_ref = np.loadtxt(file_path)
        np.testing.assert_allclose(data_summed, data_ref, rtol=rtol)

    def test_plane_bdf(self):
        self.validate_plane(method=spi.BDF, rtol=5e-3, i=5, ax=(2, 0))

    def test_plane_dop853(self):
        self.validate_plane(method=spi.DOP853, rtol=2.1e-4, i=6, ax=(2, 1))

    @unittest.expectedFailure
    def test_plane_lsoda(self):
        self.validate_plane(method=spi.LSODA, i=2, ax=(0, 1))

    def test_plane_odeint(self):
        self.validate_plane(odeint=True, method=spi.LSODA, i=1, ax=(0, 0))

    @unittest.expectedFailure
    def test_plane_radau(self):
        self.validate_plane(method=spi.Radau, i=7, ax=(2, 2))

    def test_plane_rk23(self):
        self.validate_plane(method=spi.RK23, rtol=2.11e-2, i=3, ax=(1, 0))

    def test_plane_rk45(self):
        self.validate_plane(method=spi.RK45, rtol=1.95e-3, i=4, ax=(1, 1))


if __name__ == "__main__":
    unittest.main()
