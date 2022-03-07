import os.path
import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from pttools import speedup
from pttools.speedup import spline
from . import utils

os.makedirs(utils.TEST_FIGURE_PATH, exist_ok=True)


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
        x = np.linspace(0, 2 * np.pi, 20)
        x2 = np.linspace(0, 2 * np.pi, 40)
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

        utils.assert_allclose(data, ref)
