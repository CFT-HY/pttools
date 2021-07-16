import unittest

import numpy as np

from pttools import speedup
from . import utils


class TestSpeedup(unittest.TestCase):
    @staticmethod
    def test_gradient():
        arr = np.logspace(1, 5, 10)
        utils.assert_allclose(speedup.gradient(arr), np.gradient(arr))

    @staticmethod
    def test_logspace():
        utils.assert_allclose(speedup.logspace(1, 5, 10), np.logspace(1, 5, 10))
