"""Unit tests for the functions of special relativity"""

import unittest

import numpy as np

from pttools.bubble import relativity


class RelativityTest(unittest.TestCase):
    """Unit tests for the functions of special relativity"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.v = np.linspace(0.1, 0.9, 10)

    def test_gamma(self):
        gamma = relativity.gamma(self.v)
        self.assertTrue(np.all(gamma > 0))

    def test_gamma2(self):
        gamma2 = relativity.gamma2(self.v)
        self.assertTrue(np.all(gamma2 > 0))

    def test_lorentz(self):
        mu = relativity.lorentz(0.5, self.v)
        self.assertTrue(np.all(np.isfinite(mu)))
