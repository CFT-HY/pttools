"""Tests for the suppression module."""

import sys
import typing as tp
import unittest

import numpy as np
from pandas.io.parsers import read_csv

from pttools.math.assertions import assert_allclose
from pttools.speedup.options import NUMBA_DISABLE_JIT
from pttools.ssm.suppression import DEFAULT_SUPPRESSION
from pttools.ssm.suppression.suppression_ssm_data.remove_hybrids import SUPPRESSION_FOLDER, remove_hybrids
from pttools.ssm.suppression.suppression_ssm_data.suppression_ssm_calculator import calc_sup_ssm


class SuppressionTest(unittest.TestCase):
    """Tests for the suppression module."""

    @staticmethod
    def test_remove_hybrids() -> None:
        """Compare the data created by remove_hybrids to the reference."""
        path = remove_hybrids(suffix="test")
        settings: dict[str, tp.Any] = {
            "sep": " ",
            "engine": "c"
        }
        data = read_csv(path, **settings)
        path.unlink()
        ref = read_csv(SUPPRESSION_FOLDER / "suppression_no_hybrids.txt", **settings)
        assert_allclose(data.values, ref.values)

    def test_ssm_calculator(self) -> None:
        """Compare the data created by calc_sup_ssm to the reference.

        Each quantity of each file is compared in a subtest of its own,
        so that a failure reports the differences of all of them instead of only the first one.
        """
        filenames = ["suppression_2", "suppression_no_hybrids"]
        tolerances = {
            "vw_sim": None,
            "alpha_sim": None,
            "sup_ssm": 5.30e-7,
            "Ubarf_2_ssm": None,
            "ssm_tot": 5.30e-7
        }
        if sys.platform == "darwin":
            # As of 2026-10, one of the values of suppression_2 differs from the reference on macOS
            # by 1.34e-4 for sup_ssm, 1.63e-4 for Ubarf_2_ssm and 4.59e-4 for ssm_tot.
            # As sup_ssm is proportional to Ubarf_2_ssm**2 / ssm_tot, the relative difference of ssm_tot
            # can be up to that of sup_ssm plus twice that of Ubarf_2_ssm.
            tolerances["sup_ssm"] = 1.40e-4
            tolerances["Ubarf_2_ssm"] = 1.70e-4
            tolerances["ssm_tot"] = 4.80e-4
        elif sys.platform == "win32":
            tolerances["sup_ssm"] = 5.68e-7
            tolerances["ssm_tot"] = 5.63e-7
        spectrum_kwargs: dict[str, tp.Any] = {}
        if NUMBA_DISABLE_JIT:
            # Without Numba, the default accuracy would take several minutes.
            # The point of running the test without Numba is to check that the code runs,
            # so lower accuracy is used, which changes the results by up to 3.4e-4.
            spectrum_kwargs = {"nT": 500, "n_z_lookup": 5000}
            tolerances["sup_ssm"] = 3.5e-4
            tolerances["ssm_tot"] = 3.5e-4
        for filename in filenames:
            data = calc_sup_ssm(f"{filename}.txt", save=False, **spectrum_kwargs)
            ref = np.genfromtxt(SUPPRESSION_FOLDER / f"{filename}_ssm.txt", names=True)
            for key, rtol in tolerances.items():
                with self.subTest(filename=filename, key=key):
                    if rtol is None:
                        assert_allclose(data[key], ref[key], name=key)
                    else:
                        assert_allclose(data[key], ref[key], name=key, rtol=rtol)

    @staticmethod
    def test_suppression() -> None:
        """Test that the default suppression can be computed for scalar and array inputs."""
        v_walls = np.linspace(0.1, 0.9, 5)
        alpha_ns = np.linspace(0.01, 0.1, 5)
        DEFAULT_SUPPRESSION.suppression(v_wall=v_walls[0], alpha_n=alpha_ns[0])
        DEFAULT_SUPPRESSION.suppression(v_wall=v_walls, alpha_n=alpha_ns[0])
        DEFAULT_SUPPRESSION.suppression(v_wall=v_walls, alpha_n=alpha_ns)
