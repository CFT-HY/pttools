import os.path
import typing as tp
import unittest

import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy.interpolate

from pttools import speedup
from pttools.speedup import spline
from . import utils

os.makedirs(utils.TEST_FIGURE_PATH, exist_ok=True)


@numba.njit
def jitted_spline(x: np.ndarray, tck: tp.Tuple[np.ndarray, np.ndarray, int], der: int = 0, ext: int = 0):
    return scipy.interpolate.splev(x, tck, der, ext)


class TestSpeedup(unittest.TestCase):
    @staticmethod
    def test_gradient():
        arr = np.logspace(1, 5, 10)
        utils.assert_allclose(speedup.gradient(arr), np.gradient(arr))

    @staticmethod
    def test_logspace():
        utils.assert_allclose(speedup.logspace(1, 5, 10), np.logspace(1, 5, 10))

    @staticmethod
    @unittest.expectedFailure
    def test_spline():
        x = np.linspace(0, 2*np.pi, 20)
        x2 = np.linspace(0, 2*np.pi, 40)
        y = np.sin(x)
        spl = scipy.interpolate.splrep(x, y, s=0)
        ref = scipy.interpolate.splev(x2, spl)
        data = spline.splev(x2, spl)

        fig: plt.Figure = plt.Figure()
        ax: plt.Axes = fig.add_subplot()
        ax.plot(x2, data, label="data")
        ax.plot(x2, ref, label="ref", ls=":")
        ax.legend()
        utils.save_fig_multi(fig, os.path.join(utils.TEST_FIGURE_PATH, "spline_fitpack"))
        plt.close(fig)

        utils.assert_allclose(data, ref)

    @staticmethod
    def test_spline_linear():
        x = np.linspace(0, 2*np.pi, 10)
        x2 = np.linspace(0, 2*np.pi, 20)
        y = np.cos(x)
        spl = scipy.interpolate.splrep(x, y, k=1, s=0)
        ref = scipy.interpolate.splev(x2, spl)
        data = jitted_spline(x2, spl)

        fig: plt.Figure = plt.Figure()
        ax: plt.Axes = fig.add_subplot()
        ax.plot(x2, data, label="data")
        ax.plot(x2, ref, label="ref", ls=":")
        ax.legend()
        utils.save_fig_multi(fig, os.path.join(utils.TEST_FIGURE_PATH, "spline_linear"))
        plt.close(fig)

        try:
            utils.assert_allclose(data, ref)
        except AssertionError as e:
            t, c, k = spl
            with np.printoptions(
                    edgeitems=30, linewidth=200,
                    formatter={"float": lambda f: f"{f:.4e}" if f < 0 else f" {f:.4e}"}):
                print("x:", x)
                print("y:", y)
                print("t:", t)
                print("c:", c)
                print("k:", k)
            raise e
