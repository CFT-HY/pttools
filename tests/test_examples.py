"""Tests for plotting examples"""

import logging
import unittest

import matplotlib.pyplot as plt

import examples.plot_chapman_jouguet
import examples.plot_model_comparison
import examples.plot_old_new
import examples.plot_delta_theta
from examples.const_cs import plot_const_cs_xi_v

logger = logging.getLogger(__name__)


class ExampleTest(unittest.TestCase):
    @staticmethod
    def test_plot_chapman_jouguet():
        plot = examples.plot_chapman_jouguet.main()
        plt.close(plot.fig)

    @staticmethod
    def test_plot_const_cs_xi_v():
        fig = plot_const_cs_xi_v.main()
        plt.close(fig)

    @staticmethod
    def test_plot_const_cs_xi_v_w():
        import examples.const_cs.plot_const_cs_xi_v_w as script
        script.plot.fig()

    @staticmethod
    def test_plot_delta_theta():
        import examples.plot_delta_theta as script
        script.plot.fig()

    @staticmethod
    def test_plot_model_comparison():
        plot = examples.plot_model_comparison.main()
        plt.close(plot.fig)

    @staticmethod
    def test_plot_old_new():
        fig = examples.plot_old_new.main()
        plt.close(fig)

    @staticmethod
    def test_plot_standard_model():
        import examples.plot_standard_model as script
        plt.close(script.fig)
        plt.close(script.plot.fig)
        plt.close(script.plot2.fig)
