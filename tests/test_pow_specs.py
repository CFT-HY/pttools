import unittest

import numpy as np

import ssm_paper_utils as spu


class TestPowSpecs(unittest.TestCase):
    """Unit testing for sound shell model power spectra, both velocity and GW."""
    def test_pow_specs(self):
        params_list, v2_list, Omgw_list, p_cwg_list, p_ssm_list = spu.do_all_plot_ps_compare_nuc('final3', 'pdf')

        data_path = "test_data/"
        save_id = 'test'
        file = 'data_compare_nuc-' + save_id + '.txt'

        spu.save_compare_nuc_data(data_path + file, params_list, v2_list, Omgw_list, p_cwg_list, p_ssm_list)

        data_article = np.loadtxt("test_data/data_compare_nuc-final3.txt")
        data_reference = np.loadtxt("test_data/data_compare_nuc-test_reference.txt")
        data_test = np.loadtxt("test_data/data_compare_nuc-test.txt")

        if not np.array_equal(data_reference, data_test):
            np.testing.assert_allclose(data_reference, data_test)
            print("WARNING! There are small differences in the output and reference data.")

        # The sign of p_cwg does not matter
        data_article[:, 9] = np.abs(data_article[:, 9])
        data_test[:, 9] = np.abs(data_test[:, 9])

        # PTtools has been changed since the article has been written,
        # and therefore there are slight differences in the results.
        np.testing.assert_allclose(data_test, data_article, rtol=0.14, atol=0)
