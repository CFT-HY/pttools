"""Test the experimental jitclass-based parameter storage."""

import unittest

import numba
import pytest

from pttools import speedup
from pttools.bubble.physical_params import NucArgs, PhysicalParams
from pttools.speedup import njit
from pttools.ssm.nucleation import NucType


class TestParams(unittest.TestCase):
    """Test the experimental jitclass-based parameter storage."""

    def test_nuc_args(self) -> None:
        """Test that NucArgs can be created."""
        NucArgs(0.1)

    def test_params_without_nuc(self) -> None:
        """Test that PhysicalParams without nucleation parameters have None as the nucleation type and arguments."""
        params = PhysicalParams(0.1, 0.2)
        assert params.nuc_type is None
        assert params.nuc_args is None

    def test_params_with_nuc(self) -> None:
        """Test that PhysicalParams with only the nucleation type have None as the nucleation arguments."""
        params = PhysicalParams(0.1, 0.2, NucType.SIMULTANEOUS)
        assert params.nuc_type is not None
        assert params.nuc_args is None

    def test_params_with_nuc_args(self) -> None:
        """Test that PhysicalParams can be created with both the nucleation type and arguments."""
        nuc_args = NucArgs(0.1)
        params = PhysicalParams(0.1, 0.2, NucType.SIMULTANEOUS, nuc_args)
        assert params.nuc_type is not None
        assert params.nuc_args is not None

    @unittest.skipIf(speedup.NUMBA_DISABLE_JIT, "Numba errors cannot be tested when JIT compilation is disabled.")
    def test_params_without_nuc_args_numba(self) -> None:
        """Calling jitclass constructor within jitted code without specifying all arguments fails.

        This is a known bug in Numba.
        This test will alert, when the bug is fixed.
        https://github.com/numba/numba/issues/4820.
        """
        with pytest.raises((numba.LoweringError, TypeError)):
            params_without_nuc_args_numba()
        # assert params.nuc_type is None
        # assert params.nuc_args is None

    def test_params_without_nuc_args_numba_nones(self) -> None:
        """Test creating PhysicalParams in jitted code with explicit None values for the nucleation parameters."""
        params = params_without_nuc_args_numba_nones()
        assert params.nuc_type is None
        assert params.nuc_args is None

    def test_params_with_nuc_args_numba(self) -> None:
        """Test creating PhysicalParams with nucleation arguments in jitted code."""
        params = params_with_nuc_args_numba()
        assert params.nuc_type is not None
        assert params.nuc_args is not None


# Functions that return jitclass instances cannot be cached.
@njit(cache=False)
def params_without_nuc_args_numba() -> PhysicalParams:
    """Create PhysicalParams in jitted code without the optional arguments."""
    return PhysicalParams(0.1, 0.2)


@njit(cache=False)
def params_without_nuc_args_numba_nones() -> PhysicalParams:
    """Create PhysicalParams in jitted code with explicit None values for the nucleation parameters."""
    return PhysicalParams(0.1, 0.2, None, None)


@njit(cache=False)
def params_with_nuc_args_numba() -> PhysicalParams:
    """Create PhysicalParams with nucleation arguments in jitted code."""
    nuc_args = NucArgs(0.1)
    return PhysicalParams(0.1, 0.2, NucType.SIMULTANEOUS.value, nuc_args)


if __name__ == "__main__":
    unittest.main()
