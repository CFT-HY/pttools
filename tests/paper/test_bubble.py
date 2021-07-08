import os.path
import unittest

import numpy as np

from tests.paper import ssm_paper_utils as spu
from tests import test_utils


class TestBubble(unittest.TestCase):
    @staticmethod
    def test_bubble():
        _, data = spu.do_all_plot_ps_1bubble(debug=True)
        data_summed = np.sum(data, axis=2)
        file_path = os.path.join(test_utils.TEST_DATA_PATH, "bubble.txt")

        # Generate new reference data
        # np.savetxt(file_path), data_summed)

        data_ref = np.loadtxt(os.path.join(file_path))
        test_utils.assert_allclose(data_summed, data_ref)


if __name__ == "__main__":
    unittest.main()
