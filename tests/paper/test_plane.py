# import faulthandler
import logging
import os.path
import shutil
import subprocess
import timeit
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

# faulthandler.enable()
logger = logging.getLogger(__name__)

PLOT = True
os.makedirs(TEST_FIGURE_PATH, exist_ok=True)


class TestPlane(unittest.TestCase):
    FIG_PATH = os.path.join(TEST_FIGURE_PATH, "integrators")
    grid_shape: tp.Tuple[int, int] = (2, 5)
    grid_fig: plt.Figure
    axs: np.ndarray
    ref_data: np.ndarray

    # Dicts for solvers
    mean_diffs: tp.Dict[int, float] = {}
    names: tp.Dict[int, str] = {}
    iter_times: tp.Dict[int, float] = {}

    @classmethod
    def setUpClass(cls) -> None:
        if PLOT:
            cls.grid_fig, cls.axs = plt.subplots(*cls.grid_shape, figsize=np.array([16, 9])*1.7)
            cls.grid_fig.suptitle(r"Comparison of integrators for $\xi$-$v$-plane")
            cls.ref_data = plane.xiv_plane(method=spi.odeint)

    @classmethod
    def tearDownClass(cls) -> None:
        if PLOT:
            cls.plot_perf()
            cls.plot_diff()
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

    @classmethod
    def plot_perf(cls, i_ax: tp.Tuple[int, int] = (0, 3)):
        ax: plt.Axes = cls.axs[i_ax[0], i_ax[1]]
        inds = list(cls.names.keys())
        names = [cls.names[i] for i in inds]
        iter_times = [cls.iter_times[i] if i in cls.iter_times else None for i in inds]
        ax.bar(inds, iter_times, tick_label=names)
        ax.set_title("Execution time per run")
        ax.set_xlabel("Solver")
        ax.set_ylabel("Time (s)")
        ax.set_yscale("log")

    @classmethod
    def plot_diff(cls, i_ax: tp.Tuple[int, int] = (0, 4)):
        ax: plt.Axes = cls.axs[i_ax[0], i_ax[1]]
        diff_dict = {ind: diff for ind, diff in cls.mean_diffs.items() if np.isfinite(diff)}
        inds = list(diff_dict.keys())
        names = [cls.names[i] for i in inds]
        diffs = list(diff_dict.values())
        print(diffs)
        ax.bar(inds, diffs, tick_label=names)
        ax.set_title("Mean relative difference compared to odeint")
        ax.set_xlabel("Solver")
        ax.set_ylabel("Mean relative difference")
        ax.set_yscale("log")

    def validate_plane(
            self,
            odeint: bool = False,
            method: th.ODE_SOLVER = spi.odeint,
            rtol: float = 1e-7,
            i: int = None,
            ax: tp.Tuple[int, int] = None,
            perf_iters: int = 10):
        if i in self.names:
            raise ValueError(f"Duplicate solver index: {i}")
        name = plot_plane.get_solver_name(method)
        self.names[i] = name
        # The actual results are computed first to ensure, that the code is JIT-compiled before testing performance
        data = plane.xiv_plane(method=method)
        self.mean_diffs[i] = np.nanmean(np.abs((data - self.ref_data) / data))

        result = timeit.timeit(lambda: plane.xiv_plane(method=method), number=perf_iters)
        iter_time = result/perf_iters
        self.iter_times[i] = result/perf_iters
        text = \
            f"{name} performance with {perf_iters} iterations: " \
            f"{result:.2f} s, {iter_time:.2f} s/iteration"
        print(text)
        logger.info(text)

        if PLOT and ax:
            fig: plt.Figure = plt.figure()
            ax2: plt.Axes = fig.add_subplot()
            plot_plane.plot_plane(self.axs[ax[0], ax[1]], data, method, deflag_ref=self.ref_data)
            plot_plane.plot_plane(ax2, data, method, deflag_ref=self.ref_data)
            fig_name = f"{self.FIG_PATH}_{i}_{plot_plane.get_solver_name(method)}"
            save_fig_multi(fig, fig_name)

        data_summed = np.sum(data, axis=2)
        file_path = os.path.join(TEST_DATA_PATH, "xi-v_plane.txt")

        # Generate new reference data
        # np.savetxt(file_path, data_summed)

        data_ref = np.loadtxt(file_path)
        # Asserting is the last step to ensure, that the plots are created regardless of the results
        np.testing.assert_allclose(data_summed, data_ref, rtol=rtol)

    def test_plane_bdf(self):
        self.validate_plane(method=spi.BDF, rtol=5e-3, i=5, ax=(1, 2))

    def test_plane_dop853(self):
        self.validate_plane(method=spi.DOP853, rtol=2.1e-4, i=6, ax=(1, 3))

    @unittest.expectedFailure
    def test_plane_lsoda(self):
        self.validate_plane(method=spi.LSODA, i=2, ax=(0, 1))

    @unittest.expectedFailure
    def test_plane_numba_lsoda(self):
        try:
            self.validate_plane(method="numba_lsoda", i=8, ax=(0, 2))
        except ImportError as e:
            logger.exception("Could not load NumbaLSODA.", exc_info=e)
            self.skipTest("Could not load NumbaLSODA.")

    def test_plane_odeint(self):
        self.validate_plane(method=spi.odeint, i=1, ax=(0, 0))

    @unittest.expectedFailure
    def test_plane_radau(self):
        self.validate_plane(method=spi.Radau, i=7, ax=(1, 4))

    def test_plane_rk23(self):
        self.validate_plane(method=spi.RK23, rtol=2.11e-2, i=3, ax=(1, 0))

    def test_plane_rk45(self):
        self.validate_plane(method=spi.RK45, rtol=1.95e-3, i=4, ax=(1, 1))


if __name__ == "__main__":
    unittest.main()
