import unittest

import numpy as np

from pttools import speedup
from . import test_utils


class TestSpeedup(unittest.TestCase):
    @staticmethod
    def test_gradient():
        arr = np.logspace(1, 5, 10)
        test_utils.assert_allclose(speedup.gradient(arr), np.gradient(arr))

    @staticmethod
    def test_logspace():
        test_utils.assert_allclose(speedup.logspace(1, 5, 10), np.logspace(1, 5, 10))
