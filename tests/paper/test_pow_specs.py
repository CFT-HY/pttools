"""Test power spectra predictions of SSM with different nucleation models"""

import logging
import os
import os.path
import unittest

# os.environ["NUMBA_DEBUG_CACHE"] = "1"

import numba
import numpy as np

from pttools.speedup import NUMBA_DISABLE_JIT, NUMBA_INTEGRATE_TOLERANCES
from pttools.utils.assertions import assert_allclose
from tests.paper.ssm_paper_utils import do_all_plot_ps_compare_nuc, save_compare_nuc_data
from tests.utils import TEST_DATA_PATH

logger = logging.getLogger(__name__)


class TestPowSpecs(unittest.TestCase):
    """Unit testing for sound shell model power spectra, both velocity and GW."""
    @staticmethod
    def test_pow_specs():
        pow_specs()


def pow_specs(filename: str = "data_compare_nuc-test.txt"):
    params_list, v2_list, Omgw_list, p_cwg_list, p_ssm_list = do_all_plot_ps_compare_nuc(
        save_id="final3",
        graph_file_type=None,
        lambda_correction=True
    )
    save_compare_nuc_data(
        os.path.join(TEST_DATA_PATH, filename),
        params_list, v2_list, Omgw_list, p_cwg_list, p_ssm_list
    )

    data_article = np.loadtxt(os.path.join(TEST_DATA_PATH, "data_compare_nuc-final3.txt"))
    data_reference = np.loadtxt(os.path.join(TEST_DATA_PATH, "data_compare_nuc-test_reference.txt"))
    data_test = np.loadtxt(os.path.join(TEST_DATA_PATH, filename))

    # The sign of p_cwg does not matter
    data_article[:, 9] = np.abs(data_article[:, 9])
    data_reference[:, 9] = np.abs(data_reference[:, 9])
    data_test[:, 9] = np.abs(data_test[:, 9])

    if NUMBA_DISABLE_JIT:
        logger.info(f"Numba is disabled: NUMBA_DISABLE_JIT = {NUMBA_DISABLE_JIT}")
        # The results differ slightly depending on the library versions
        # rtol = 4.6e-7
        # Changing v_plus and v_minus implementations has changed the results further for pure Python
        # rtol = 9.61e-7
        # Changes continue
        # rtol = 1.14e-6
        # More changes in 2023-07
        rtol = 1.24e-6
    else:
        # Since this was a heavy computation, let's print info on the threading layer used
        logger.info(f"Numba threading layer used: {numba.threading_layer()}")
        # Using Numba changes the results slightly
        # rtol = 9.1e-7
        # The library versions on Kale change the results further
        # rtol = 1.21e-6
        # Working around indexing bugs in envelope() has changed the results further
        rtol = 1.72e-6
    if NUMBA_INTEGRATE_TOLERANCES:
        logger.warning("test_pow_specs tolerances have been loosened for Numba")
        rtol = 0.013
    assert_allclose(data_test, data_reference, rtol=rtol, atol=0)

    # PTtools has been changed since the article has been written,
    # and therefore there are slight differences in the results.
    assert_allclose(data_test, data_article, rtol=0.14, atol=0)


if __name__ == "__main__":
    unittest.main()
