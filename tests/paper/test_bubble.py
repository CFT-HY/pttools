import logging
import os.path
import unittest

import numpy as np

from pttools import speedup
from tests.paper import ssm_paper_utils as spu
from tests import test_utils

logger = logging.getLogger(__name__)


class TestBubble(unittest.TestCase):
    @staticmethod
    def test_bubble():
        _, data = spu.do_all_plot_ps_1bubble(debug=True)
        data_summed = np.sum(data, axis=2)
        file_path = os.path.join(test_utils.TEST_DATA_PATH, "bubble.txt")

        # Generate new reference data
        # np.savetxt(file_path), data_summed)

        data_ref = np.loadtxt(os.path.join(file_path))
        if speedup.NUMBA_INTEGRATE:
            logger.warning("test_bubble tolerances have been loosened for NumbaLSODA")
        test_utils.assert_allclose(data_summed, data_ref, rtol=(0.012 if speedup.NUMBA_INTEGRATE else 1e-7))


if __name__ == "__main__":
    unittest.main()
