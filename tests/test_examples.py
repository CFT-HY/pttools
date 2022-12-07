"""Tests for plotting examples"""

import logging
import unittest

logger = logging.getLogger(__name__)


class ExampleTest(unittest.TestCase):
    @staticmethod
    def test_plot_const_cs_xi_w():
        import examples.plot_const_cs_xi_v_w as script
        script.plot.fig()

    @staticmethod
    def test_delta_theta():
        import examples.plot_delta_theta as script
        script.plot.fig()
