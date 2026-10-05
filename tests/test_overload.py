"""Unit tests for the Numba overloads of pttools.speedup.overload."""

import typing as tp
import unittest

import numba
import numpy as np

from pttools.speedup.overload import np_all_fix


def _jit_all(x: tp.Any) -> bool | np.bool:
    return np.all(x)


def _jit_any(x: tp.Any) -> bool | np.bool:
    return np.any(x)


def _jit_all_fix(x: tp.Any) -> bool | np.bool:
    return np_all_fix(x)


class TestOverload(unittest.TestCase):
    """Test that the overloads give the same results as NumPy for booleans, scalars and arrays."""

    INPUTS: tp.ClassVar[list[tp.Any]] = [
        True, False, 0, 1, -2, 0., 1.5,
        np.array([1., 0.]), np.array([1., 2.]), np.array([0., 0.]),
        np.array([True, False]), np.array([True, True]), np.array([False, False])
    ]

    def check(self, func: tp.Callable[[tp.Any], bool | np.bool], ref: tp.Callable[[tp.Any], tp.Any]) -> None:
        """Check that the jitted function gives the same results as the reference function for all the inputs."""
        # A new dispatcher is created for each test to avoid sharing compiled signatures between the tests.
        jitted = numba.njit(func)
        for x in self.INPUTS:
            with self.subTest(x=x):
                assert bool(jitted(x)) == bool(ref(x))

    def test_all(self) -> None:
        """Test that the jitted np.all gives the same results as NumPy."""
        self.check(_jit_all, np.all)

    def test_any(self) -> None:
        """Test that the jitted np.any gives the same results as NumPy."""
        self.check(_jit_any, np.any)

    def test_all_fix(self) -> None:
        """Test that the jitted np_all_fix gives the same results as np.all."""
        self.check(_jit_all_fix, np.all)

    def test_all_fix_python(self) -> None:
        """Test that np_all_fix without jitting gives the same results as np.all."""
        for x in self.INPUTS:
            with self.subTest(x=x):
                assert bool(np_all_fix(x)) == bool(np.all(x))
