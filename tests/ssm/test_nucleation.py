"""Tests for the nucleation functions."""

import typing as tp
import unittest

import numpy as np
import pytest

from pttools.speedup import logspace
from pttools.ssm.const import DEFAULT_N_T, T_TILDE_MAX, T_TILDE_MIN
from pttools.ssm.nucleation import NucType, lifetime_distribution, lifetime_distribution_momentum
import pttools.type_hints as th


class NucleationTest(unittest.TestCase):
    """Test the bubble lifetime distributions of the nucleation types."""

    T_tilde: th.FloatArr1D

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.T_tilde = logspace(np.log10(T_TILDE_MIN), np.log10(T_TILDE_MAX), DEFAULT_N_T)

    def test_exponential(self) -> None:
        """Test the third moment of the bubble lifetime distribution for exponential nucleation."""
        nu = lifetime_distribution(self.T_tilde, NucType.EXPONENTIAL)
        assert lifetime_distribution_momentum(nu, self.T_tilde, 3) == pytest.approx(6, abs=2e-5)

    def test_simultaneous(self) -> None:
        """Test the third moment of the bubble lifetime distribution for simultaneous nucleation."""
        nu = lifetime_distribution(self.T_tilde, NucType.SIMULTANEOUS)
        assert lifetime_distribution_momentum(nu, self.T_tilde, 3) == pytest.approx(6, abs=6e-7)


if __name__ == "__main__":
    unittest.main()
