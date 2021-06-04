import os.path
import unittest

import numpy as np

from pttools import bubble
import ssm_paper_utils as spu
from test_utils import TEST_DATA_PATH


class TestShells(unittest.TestCase):
    @staticmethod
    def shell_file_path(name: str) -> str:
        return os.path.join(TEST_DATA_PATH, f"shells_{name}.txt")

    def test_fluid_shell(self):
        _, arrs, scalars = bubble.plot_fluid_shell(v_wall=0.7, alpha_n=0.052, debug=True, draw=False)
        data = np.array([np.nansum(arr) for arr in arrs] + scalars)
        file_path = os.path.join(TEST_DATA_PATH, "shell.txt")

        # Generate new reference data
        # np.savetxt(file_path, data)

        data_ref = np.loadtxt(file_path)
        np.testing.assert_allclose(data, data_ref)

    def test_fluid_shells(self):
        """Based on sound-shell-model/paper/python/fig_1_9_shell_plots.py"""
        vw_weak_list = spu.vw_weak_list
        vw_inter_list = spu.vw_inter_list

        alpha_weak = spu.alpha_weak
        alpha_inter = spu.alpha_inter

        alpha_weak_list = len(vw_weak_list) * [alpha_weak]
        alpha_inter_list = len(vw_inter_list) * [alpha_inter]

        _, data_weak = bubble.plot_fluid_shells(vw_weak_list, alpha_weak_list, debug=True, draw=False)
        _, data_inter = bubble.plot_fluid_shells(vw_inter_list, alpha_inter_list, debug=True, draw=False)

        vw_list_esp = [0.5, 0.7, 0.77]
        alpha_plus_list_esp = [0.263, 0.052, 0.091]
        alpha_n_list_esp = [bubble.find_alpha_n(vw, ap) for vw, ap in zip(vw_list_esp, alpha_plus_list_esp)]

        _, data_esp = bubble.plot_fluid_shells(vw_list_esp, alpha_n_list_esp, multi=True, debug=True, draw=False)

        # Generate new reference data
        # np.savetxt(self.shell_file_path("weak"), data_weak)
        # np.savetxt(self.shell_file_path("inter"), data_inter)
        # np.savetxt(self.shell_file_path("esp"), data_esp)

        ref_weak = np.loadtxt(self.shell_file_path("weak"))
        ref_inter = np.loadtxt(self.shell_file_path("inter"))
        ref_esp = np.loadtxt(self.shell_file_path("esp"))

        np.testing.assert_allclose(data_weak, ref_weak)
        np.testing.assert_allclose(data_inter, ref_inter)
        np.testing.assert_allclose(data_esp, ref_esp)
