import unittest

import numpy as np

from pttools import speedup


class TestSpeedup(unittest.TestCase):
    @staticmethod
    def test_gradient():
        arr = np.logspace(1, 5, 10)
        np.testing.assert_allclose(speedup.gradient(arr), np.gradient(arr))

    @staticmethod
    def test_logspace():
        np.testing.assert_allclose(speedup.logspace(1, 5, 10), np.logspace(1, 5, 10))
