import unittest

import numpy as np

from pttools.bubble import relativity


class RelativityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v = np.linspace(0.1, 0.9, 10)

    def test_gamma(self):
        g = relativity.gamma(self.v)
        self.assertTrue(np.all(g > 0))

    def test_gamma2(self):
        g2 = relativity.gamma2(self.v)
        self.assertTrue(np.all(g2 > 0))

    def test_lorentz(self):
        mu = relativity.lorentz(0.5, self.v)
        self.assertTrue(np.all(np.isfinite(mu)))
