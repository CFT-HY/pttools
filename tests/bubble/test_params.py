import unittest

import numba

import pttools.bubble.physical_params as pp
from pttools import speedup


class TestParams(unittest.TestCase):
    """Test the experimental jitclass-based parameter storage"""
    def test_nuc_args(self):
        pp.NucArgs(0.1)

    def test_params_without_nuc(self):
        params = pp.PhysicalParams(0.1, 0.2)
        self.assertIsNone(params.nuc_type)
        self.assertIsNone(params.nuc_args)

    def test_params_with_nuc(self):
        params = pp.PhysicalParams(0.1, 0.2, pp.NucType.SIMULTANEOUS)
        self.assertIsNotNone(params.nuc_type)
        self.assertIsNone(params.nuc_args)

    def test_params_with_nuc_args(self):
        nuc_args = pp.NucArgs(0.1)
        params = pp.PhysicalParams(0.1, 0.2, pp.NucType.SIMULTANEOUS, nuc_args)
        self.assertIsNotNone(params.nuc_type)
        self.assertIsNotNone(params.nuc_args)

    @unittest.skipIf(speedup.NUMBA_DISABLE_JIT, "Numba errors cannot be tested when JIT-compilation is disabled.")
    def test_params_without_nuc_args_numba(self):
        """Calling jitclass constructor within jitted code without specifying all arguments fails.
        This is a known bug in Numba.
        This test will alert, when the bug is fixed.
        https://github.com/numba/numba/issues/4820
        """
        with self.assertRaises((numba.LoweringError, TypeError)):
            params_without_nuc_args_numba()
        # self.assertIsNone(params.nuc_type)
        # self.assertIsNone(params.nuc_args)

    def test_params_without_nuc_args_numba_nones(self):
        params = params_without_nuc_args_numba_nones()
        self.assertIsNone(params.nuc_type)
        self.assertIsNone(params.nuc_args)

    def test_params_with_nuc_args_numba(self):
        params = params_with_nuc_args_numba()
        self.assertIsNotNone(params.nuc_type)
        self.assertIsNotNone(params.nuc_args)


@numba.njit
def params_without_nuc_args_numba():
    return pp.PhysicalParams(0.1, 0.2)


@numba.njit
def params_without_nuc_args_numba_nones():
    return pp.PhysicalParams(0.1, 0.2, None, None)


@numba.njit
def params_with_nuc_args_numba():
    nuc_args = pp.NucArgs(0.1)
    return pp.PhysicalParams(0.1, 0.2, pp.NucType.SIMULTANEOUS.value, nuc_args)


if __name__ == "__main__":
    unittest.main()
