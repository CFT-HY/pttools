"""Unit tests for the functions of special relativity."""

import typing as tp
import unittest

import numpy as np

from pttools.bubble import relativity
import pttools.type_hints as th


class RelativityTest(unittest.TestCase):
    """Unit tests for the functions of special relativity."""

    v: th.FloatArr1D

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.v = np.linspace(0.1, 0.9, 10)

    def test_gamma(self) -> None:
        """Test that the Lorentz factor is positive."""
        gamma = relativity.gamma(self.v)
        assert np.all(gamma > 0)

    def test_gamma2(self) -> None:
        """Test that the square of the Lorentz factor is positive."""
        gamma2 = relativity.gamma2(self.v)
        assert np.all(gamma2 > 0)

    def test_lorentz(self) -> None:
        """Test that the Lorentz transformation of velocities gives finite results."""
        mu = relativity.lorentz(0.5, self.v)
        assert np.all(np.isfinite(mu))
