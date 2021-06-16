import logging
import os.path
import unittest

import numba
import numpy as np

from tests.paper import ssm_paper_utils as spu
from tests.test_utils import TEST_DATA_PATH

logger = logging.getLogger(__name__)


class TestPowSpecs(unittest.TestCase):
    """Unit testing for sound shell model power spectra, both velocity and GW."""
    def test_pow_specs(self):
        params_list, v2_list, Omgw_list, p_cwg_list, p_ssm_list = spu.do_all_plot_ps_compare_nuc('final3', None)

        save_id = 'test'
        file = 'data_compare_nuc-' + save_id + '.txt'

        spu.save_compare_nuc_data(
            os.path.join(TEST_DATA_PATH, file),
            params_list, v2_list, Omgw_list, p_cwg_list, p_ssm_list
        )

        data_article = np.loadtxt(os.path.join(TEST_DATA_PATH, "data_compare_nuc-final3.txt"))
        data_reference = np.loadtxt(os.path.join(TEST_DATA_PATH, "data_compare_nuc-test_reference.txt"))
        data_test = np.loadtxt(os.path.join(TEST_DATA_PATH, "data_compare_nuc-test.txt"))

        # The sign of p_cwg does not matter
        data_article[:, 9] = np.abs(data_article[:, 9])
        data_reference[:, 9] = np.abs(data_reference[:, 9])
        data_test[:, 9] = np.abs(data_test[:, 9])

        # The results differ slightly depending on the library versions
        # np.testing.assert_allclose(data_test, data_reference, rtol=4.6e-7, atol=0)
        # Using Numba changes the results slightly
        np.testing.assert_allclose(data_test, data_reference, rtol=9.1e-7, atol=0)

        # PTtools has been changed since the article has been written,
        # and therefore there are slight differences in the results.
        np.testing.assert_allclose(data_test, data_article, rtol=0.14, atol=0)

        # Since this was a heavy computation, let's print info on the threading layer used
        logger.debug(f"Numba threading layer used: {numba.threading_layer()}")
