import unittest

import numpy as np

from pttools.speedup import logspace
from pttools.ssm.const import DEFAULT_N_T, T_TILDE_MAX, T_TILDE_MIN
from pttools.ssm.nucleation import NucType, lifetime_distribution, lifetime_distribution_momentum


class NucleationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.T_tilde = logspace(np.log10(T_TILDE_MIN), np.log10(T_TILDE_MAX), DEFAULT_N_T)

    def test_exponential(self):
        nu = lifetime_distribution(self.T_tilde, NucType.EXPONENTIAL)
        self.assertAlmostEqual(lifetime_distribution_momentum(nu, self.T_tilde, 3), 6, delta=2e-5)

    def test_simultaneous(self):
        nu = lifetime_distribution(self.T_tilde, NucType.SIMULTANEOUS)
        self.assertAlmostEqual(lifetime_distribution_momentum(nu, self.T_tilde, 3), 6, delta=6e-7)


if __name__ == "__main__":
    unittest.main()
