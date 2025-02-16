"""Tests for plotting examples"""

import logging
import unittest

import matplotlib.pyplot as plt

from  examples.e02_const_cs import plot_model_comparison
from examples.e04_props import plot_chapman_jouguet
from examples.e06_solvers import plot_old_new
from examples.e02_const_cs import plot_const_cs_xi_v

logger = logging.getLogger(__name__)


class ExampleTest(unittest.TestCase):
    @staticmethod
    def test_plot_chapman_jouguet():
        plot = plot_chapman_jouguet.main()
        plt.close(plot.fig)

    @staticmethod
    def test_plot_const_cs_xi_v():
        fig = plot_const_cs_xi_v.main()
        plt.close(fig)

    @staticmethod
    def test_plot_const_cs_xi_v_w():
        import examples.e02_const_cs.plot_const_cs_xi_v_w as script
        script.plot.fig()

    @staticmethod
    def test_plot_delta_theta():
        from examples.e04_props import plot_delta_theta
        plot_delta_theta.plot.fig()

    @staticmethod
    def test_plot_model_comparison():
        plot = plot_model_comparison.main()
        plt.close(plot.fig)

    @staticmethod
    def test_plot_old_new():
        fig = plot_old_new.main()
        plt.close(fig)

    @staticmethod
    def test_plot_standard_model():
        import examples.e03_standard_model.plot_standard_model as script
        plt.close(script.fig)
        plt.close(script.plot.fig)
        plt.close(script.plot2.fig)
