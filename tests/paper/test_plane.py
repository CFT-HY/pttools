import logging
import os.path
import shutil
import subprocess
import sys
import timeit
import typing as tp
import unittest

import matplotlib.pyplot as plt
import numpy as np

from pttools import speedup
from tests.plotting import save_fig_multi
from tests import test_utils
from tests.paper import plane
from tests.paper import plot_plane

logger = logging.getLogger(__name__)

PLOT = True
os.makedirs(test_utils.TEST_FIGURE_PATH, exist_ok=True)


class TestPlane(unittest.TestCase):
    FIGSIZE = np.array([16, 9])*1.7
    FIG_PATH = os.path.join(test_utils.TEST_FIGURE_PATH, "integrators")
    grid_shape: tp.Tuple[int, int] = (2, 5)
    grid_fig_abs: plt.Figure
    grid_fig_rel: plt.Figure
    axs_abs: np.ndarray
    axs_rel: np.ndarray
    ref_data: np.ndarray

    # Dicts for solvers
    mean_rel_diffs: tp.Dict[int, float] = {}
    mean_abs_diffs: tp.Dict[int, float] = {}
    names: tp.Dict[int, str] = {}
    iter_times: tp.Dict[int, float] = {}

    @classmethod
    def setUpClass(cls) -> None:
        if PLOT:
            cls.grid_fig_abs, cls.axs_abs = plt.subplots(*cls.grid_shape, figsize=cls.FIGSIZE)
            cls.grid_fig_rel, cls.axs_rel = plt.subplots(*cls.grid_shape, figsize=cls.FIGSIZE)
            common_title = r"Comparison of integrators for $\xi$-$v$-plane"
            cls.grid_fig_abs.suptitle(f"{common_title}, absolute errors")
            cls.grid_fig_rel.suptitle(f"{common_title}, relative errors")
            cls.ref_data = plane.xiv_plane(method="odeint")

    @classmethod
    def process_output(cls, name: str, fig: plt.Figure, axs: np.ndarray, diffs: tp.Dict[int, float]):
        cls.plot_perf(axs[0, 3])
        cls.plot_diff(axs[0, 4], name, diffs)
        fig.tight_layout()
        path = f"{cls.FIG_PATH}_{name}"
        save_fig_multi(fig, path)
        if shutil.which("ffmpeg"):
            video_path = f"{path}.mp4"
            if os.path.exists(video_path):
                os.remove(video_path)
            kwargs = {}
            if sys.version_info >= (3, 7):
                kwargs["capture_output"] = True
            ret: subprocess.CompletedProcess = subprocess.run(
                [
                    "ffmpeg",
                    "-framerate", "0.5",
                    "-pattern_type", "glob",
                    "-i", f"{path}_*.png",
                    "-c:v", "libx264",
                    "-r", "30",
                    video_path
                ],
                check=False,
                **kwargs
            )
            # print(ret.stdout)
            # print(ret.stderr)
            ret.check_returncode()
        else:
            logger.warning("ffmpeg was not found, so the xi-v-plane animation could not be created")

    @classmethod
    def tearDownClass(cls) -> None:
        if PLOT:
            cls.process_output("absolute", cls.grid_fig_abs, cls.axs_abs, cls.mean_abs_diffs)
            cls.process_output("relative", cls.grid_fig_rel, cls.axs_rel, cls.mean_rel_diffs)

    @classmethod
    def plot_perf(cls, ax: plt.Axes):
        inds = list(cls.names.keys())
        names = [cls.names[i] for i in inds]
        # None is not supported here in old Matplotlib, so 0 is used instead
        iter_times = [cls.iter_times[i] if i in cls.iter_times else 0 for i in inds]
        ax.bar(inds, iter_times, tick_label=names)
        ax.set_title("Execution time per run")
        ax.set_xlabel("Solver")
        ax.set_ylabel("Time (s)")
        ax.set_yscale("log")

    @classmethod
    def plot_diff(cls, ax: plt.Axes, name: str, diffs: tp.Dict[int, float]):
        diff_dict = {ind: diff for ind, diff in diffs.items() if np.isfinite(diff)}
        inds = list(diff_dict.keys())
        names = [cls.names[i] for i in inds]
        diff_vals = list(diff_dict.values())
        ax.bar(inds, diff_vals, tick_label=names)
        ax.set_title(f"Mean {name} error compared to odeint")
        ax.set_xlabel("Solver")
        ax.set_ylabel(f"Mean {name} error")
        ax.set_yscale("log")

    def validate_plane(
            self,
            method: str = "odeint",
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
        self.mean_abs_diffs[i] = np.nanmean(np.abs(data - self.ref_data))
        self.mean_rel_diffs[i] = np.nanmean(np.abs((data - self.ref_data) / data))

        result = timeit.timeit(lambda: plane.xiv_plane(method=method), number=perf_iters)
        iter_time = result/perf_iters
        self.iter_times[i] = result/perf_iters
        text = \
            f"{name} performance with {perf_iters} iterations: " \
            f"{result:.2f} s, {iter_time:.2f} s/iteration"
        print(text)
        logger.info(text)

        abs_tols = {
            "atol_small_diff": 1e-5,
            "atol_mid_diff": 1e-4,
            "atol_high_diff": 1e-3,
            "rtol_small_diff": 0,
            "rtol_mid_diff": 0,
            "rtol_high_diff": 0
        }
        rel_tols = {}

        if PLOT and ax:
            for name, axs, tols in zip(
                    ("absolute", "relative"),
                    (self.axs_abs, self.axs_rel),
                    (abs_tols, rel_tols)):
                fig: plt.Figure = plt.figure()
                ax2: plt.Axes = fig.add_subplot()
                plot_plane.plot_plane(axs[ax[0], ax[1]], data, method, deflag_ref=self.ref_data, **tols)
                plot_plane.plot_plane(ax2, data, method, deflag_ref=self.ref_data, **tols)
                fig_name = f"{self.FIG_PATH}_{name}_{i}_{plot_plane.get_solver_name(method)}"
                save_fig_multi(fig, fig_name)

        data_summed = np.nansum(data, axis=2)
        file_path = os.path.join(test_utils.TEST_DATA_PATH, "xi-v_plane.txt")

        # Generate new reference data
        # if method == spi.odeint:
        #     np.savetxt(file_path, data_summed)

        data_ref = np.loadtxt(file_path)
        # Asserting is the last step to ensure, that the plots are created regardless of the results
        test_utils.assert_allclose(data_summed, data_ref, rtol=rtol)

    def test_plane_bdf(self):
        self.validate_plane(method="BDF", rtol=5e-3, i=5, ax=(1, 2))

    def test_plane_dop853(self):
        self.validate_plane(method="DOP853", rtol=1.57e-2, i=6, ax=(1, 3))

    def test_plane_lsoda(self):
        self.validate_plane(method="LSODA", rtol=3.1e-3, i=2, ax=(0, 1))

    @unittest.skipIf(speedup.NUMBA_DISABLE_JIT, "NumbaLSODA cannot be used if Numba is disabled")
    def test_plane_numba_lsoda(self):
        try:
            self.validate_plane(method="numba_lsoda", rtol=4.0e-3, i=8, ax=(0, 2))
        except ImportError as e:
            logger.exception("Could not load NumbaLSODA.", exc_info=e)
            self.skipTest("Could not load NumbaLSODA.")

    def test_plane_odeint(self):
        self.validate_plane(method="odeint", i=1, ax=(0, 0))

    def test_plane_radau(self):
        self.validate_plane(method="Radau", rtol=8.24e-4, i=7, ax=(1, 4))

    def test_plane_rk23(self):
        self.validate_plane(method="RK23", rtol=2.11e-2, i=3, ax=(1, 0))

    def test_plane_rk45(self):
        self.validate_plane(method="RK45", rtol=1.95e-3, i=4, ax=(1, 1))


if __name__ == "__main__":
    unittest.main()
