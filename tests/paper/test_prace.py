import logging
import os.path
import unittest

import numpy as np

from pttools.speedup import NUMBA_INTEGRATE
from tests.paper import ssm_compare as scom
from tests import test_utils

logger = logging.getLogger(__name__)


class TestPrace(unittest.TestCase):
    def test_ps_prace(self):
        fluid_profiles_dir = os.path.join(test_utils.TEST_DATA_PATH, "fluidprofiles")
        if not os.path.isdir(fluid_profiles_dir):
            logger.warning("Fluid profiles not found. Cannot execute PRACE tests.")
            return
        v2_list, Omgw_scaled_list, data = scom.all_generate_ps_prace(save_ids=("test", "test"), show=False, debug=True)
        ref_path = os.path.join(test_utils.TEST_DATA_PATH, "ps_prace.txt")

        test_data = np.concatenate([np.array([v2_list, Omgw_scaled_list]), data])

        # Generate new reference data
        # np.savetxt(ref_path, test_data)

        ref_data = np.loadtxt(ref_path)
        if NUMBA_INTEGRATE:
            logger.warning("test_ps_prace tolerances have been loosened for NumbaLSODA")
        test_utils.assert_allclose(test_data, ref_data, rtol=(1.6e-2 if NUMBA_INTEGRATE else 1e-7))


if __name__ == "__main__":
    unittest.main()
