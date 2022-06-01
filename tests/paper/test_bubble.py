import logging
import os.path
import unittest

import matplotlib.pyplot as plt
import numpy as np

from pttools import speedup
from tests.paper import ssm_paper_utils as spu
from tests.utils.const import TEST_FIGURE_PATH
from tests import utils

logger = logging.getLogger(__name__)

FIG_PATH = os.path.join(TEST_FIGURE_PATH, "bubble")


class TestBubble(unittest.TestCase):
    @staticmethod
    def test_bubble():
        figs, fig_ids, data = spu.do_all_plot_ps_1bubble(debug=True)
        for fig, fig_id in zip(figs, fig_ids):
            utils.save_fig_multi(fig, os.path.join(FIG_PATH, f"bubble_{fig_id}"))
            plt.close(fig)
        data_summed = np.sum(data, axis=2)
        file_path = os.path.join(utils.TEST_DATA_PATH, "bubble.txt")

        # Generate new reference data
        # np.savetxt(file_path), data_summed)

        data_ref = np.loadtxt(os.path.join(file_path))
        if speedup.NUMBA_INTEGRATE:
            logger.warning("test_bubble tolerances have been loosened for NumbaLSODA")
        utils.assert_allclose(data_summed, data_ref, rtol=(0.012 if speedup.NUMBA_INTEGRATE else 1e-7))


if __name__ == "__main__":
    unittest.main()
