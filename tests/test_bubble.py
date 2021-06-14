import os.path
import unittest

import numpy as np

import ssm_paper_utils as spu
from test_utils import TEST_DATA_PATH


class TestBubble(unittest.TestCase):
    @staticmethod
    def test_bubble():
        _, data = spu.do_all_plot_ps_1bubble(debug=True)
        data_summed = np.sum(data, axis=2)
        file_path = os.path.join(TEST_DATA_PATH, "bubble.txt")

        # Generate new reference data
        # np.savetxt(file_path), data_summed)

        data_ref = np.loadtxt(os.path.join(file_path))
        np.testing.assert_allclose(data_summed, data_ref)
