import os.path
import unittest

import numpy as np

import ssm_compare as scom
from test_utils import TEST_DATA_PATH


class TestPrace(unittest.TestCase):
    def test_ps_prace(self):
        fluid_profiles_dir = os.path.join(TEST_DATA_PATH, "fluidprofiles")
        if not os.path.isdir(fluid_profiles_dir):
            print("Fluid profiles not found. Cannot execute PRACE tests.")
            return
        v2_list, Omgw_scaled_list, data = scom.all_generate_ps_prace(save_ids=["test", "test"], show=False, debug=True)
        ref_path = os.path.join(TEST_DATA_PATH, "ps_prace.txt")

        test_data = np.concatenate([np.array([v2_list, Omgw_scaled_list]), data])
        print(test_data.shape)
        # Generate new reference data
        # np.savetxt(ref_path, test_data)

        ref_data = np.loadtxt(ref_path)
        np.testing.assert_allclose(test_data, ref_data)
