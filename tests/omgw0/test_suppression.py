"""Tests for the suppression module"""

import os
import sys
import unittest

import numpy as np
from pandas.io.parsers import read_csv

from pttools.omgw0.suppression.suppression_ssm_data.remove_hybrids import SUPPRESSION_FOLDER, remove_hybrids
from pttools.omgw0.suppression.suppression_ssm_data.suppression_ssm_calculator import calc_sup_ssm
import pttools.ssmtools.const as ssm_const
from tests.utils.assertions import assert_allclose


class SuppressionTest(unittest.TestCase):
    """Tests for the suppression module"""
    @staticmethod
    def test_remove_hybrids():
        """Compare the data created by remove_hybrids to the reference"""
        path = remove_hybrids(suffix="test")
        settings = {
            "sep": " ",
            "engine": "c"
        }
        data = read_csv(path, **settings)
        os.remove(path)
        ref = read_csv(os.path.join(SUPPRESSION_FOLDER, "suppression_no_hybrids.txt"), **settings)
        assert_allclose(data.values, ref.values)

    @staticmethod
    def test_ssm_calculator():
        """Compare the data created by calc_sup_ssm to the reference"""
        filenames = ["suppression_2", "suppression_no_hybrids"]
        tolerances = {
            "vw_sim": None,
            "alpha_sim": None,
            "sup_ssm": 5.30e-7,
            "Ubarf_2_ssm": None,
            "ssm_tot": 5.30e-7
        }
        if sys.platform == "darwin":
            tolerances["sup_ssm"] = 2.66e-5
            tolerances["Ubarf_2_ssm"] = 1.33e-5
        for filename in filenames:
            data = calc_sup_ssm(
                f"{filename}.txt",
                save=False,
                npt=(ssm_const.NXIDEFAULT, 200, 320)
            )
            with np.load(os.path.join(SUPPRESSION_FOLDER, f"{filename}_ssm.npz")) as ref:
                for key, rtol in tolerances.items():
                    if rtol is None:
                        assert_allclose(data[key], ref[key], name=key)
                    else:
                        assert_allclose(data[key], ref[key], name=key, rtol=rtol)
