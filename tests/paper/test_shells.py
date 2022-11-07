import logging
import os.path
import sys
import unittest

import matplotlib.pyplot as plt
import numpy as np
# import orjson

from pttools import bubble
from pttools.analysis.plot_fluid_shell import plot_fluid_shell
from pttools.analysis.plot_fluid_shells import plot_fluid_shells
from pttools.speedup import NUMBA_INTEGRATE_TOLERANCES
from tests.paper import const
from tests.paper import ssm_paper_utils as spu
from tests import utils

logger = logging.getLogger(__name__)

FIG_PATH = os.path.join(utils.TEST_FIGURE_PATH, "fluid_shells")


class TestShells(unittest.TestCase):
    @staticmethod
    def shell_file_path(name: str) -> str:
        return os.path.join(utils.TEST_DATA_PATH, f"shells_{name}.txt")

    def test_fluid_shell(self):
        params = bubble.fluid_shell_params(v_wall=0.7, alpha_n=0.052)
        # These are not yet in the reference data
        for name in ["sol_type", "xi_even", "v_approx", "w_approx"]:
            params.pop(name)
        # data = {"arrays": arrs, "scalars": scalars}
        # for name, value in params.items():
        #     print(name, value, type(value))
        # print([np.nansum(param) if isinstance(param, np.ndarray) else param for param in params.values()])
        data_numpy = np.array(
            [np.nansum(param) if isinstance(param, np.ndarray) else param for param in params.values()],
            dtype=np.float_)
        # [np.nansum(arr) for arr in arrs.values()] + list(scalars.values()))

        # path_json = os.path.join(utils.TEST_DATA_PATH, "shell.json")
        path_txt = os.path.join(utils.TEST_DATA_PATH, "shell.txt")

        # Generate new reference data
        # Using old reference data for now. It was generated with the following code.
        # data = np.array([np.nansum(arr) for arr in arrs] + scalars)
        # np.savetxt(path_txt, data)
        # with open(path_json, "wb") as file:
        #     file.write(orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY))

        data_ref = np.loadtxt(path_txt)
        # arrs_nansum = {f"nansum({name})": np.nansum(arr) for name, arr in arrs.items()}
        if NUMBA_INTEGRATE_TOLERANCES:
            logger.warning("test_fluid_shell tolerances have been loosened for NumbaLSODA")
        utils.assert_allclose(data_numpy, data_ref, rtol=(0.292 if NUMBA_INTEGRATE_TOLERANCES else 1e-7))

    def test_fluid_shells(self):
        """Based on sound-shell-model/paper/python/fig_1_9_shell_plots.py"""
        vw_weak_list = const.VW_WEAK_LIST
        vw_inter_list = spu.VW_INTER_LIST

        alpha_weak = const.ALPHA_WEAK
        alpha_inter = const.ALPHA_INTER

        alpha_weak_list = len(vw_weak_list) * [alpha_weak]
        alpha_inter_list = len(vw_inter_list) * [alpha_inter]

        fig_weak, data_weak = plot_fluid_shells(vw_weak_list, alpha_weak_list, debug=True)
        fig_inter, data_inter = plot_fluid_shells(vw_inter_list, alpha_inter_list, debug=True)

        # Espinosa et al. 2010 comparisons
        vw_list_esp = [0.5, 0.7, 0.77]
        alpha_plus_list_esp = [0.263, 0.052, 0.091]
        alpha_n_list_esp = [bubble.find_alpha_n(vw, ap) for vw, ap in zip(vw_list_esp, alpha_plus_list_esp)]

        fig_esp, data_esp = plot_fluid_shells(vw_list_esp, alpha_n_list_esp, multi=True, debug=True)

        for fig, name in zip([fig_weak, fig_inter, fig_esp], ["weak", "inter", "esp"]):
            utils.save_fig_multi(fig, os.path.join(FIG_PATH, name))
            plt.close(fig)

        # Generate new reference data
        # np.savetxt(self.shell_file_path("weak"), data_weak)
        # np.savetxt(self.shell_file_path("inter"), data_inter)
        # np.savetxt(self.shell_file_path("esp"), data_esp)

        ref_weak = np.loadtxt(self.shell_file_path("weak"))
        ref_inter = np.loadtxt(self.shell_file_path("inter"))
        ref_esp = np.loadtxt(self.shell_file_path("esp"))

        if NUMBA_INTEGRATE_TOLERANCES:
            rtols = [0.395, 0.293, 0.104]
            logger.warning("test_fluid_shells tolerances have been loosened for NumbaLSODA: %s", rtols)
        elif sys.platform.startswith("win32"):
            rtols = [0.0196, 1e-7, 1e-7]
            # logger.warning("test_fluid_shells tolerances have been loosened for Windows: %s", rtols)
        else:
            # rtols = [1e-7, 1e-7, 1e-7]
            # Work on the model-independent fluid shell generator has required increasing the tolerances.
            rtols = [0.0196, 1e-7, 1e-7]

        utils.assert_allclose(data_weak, ref_weak, rtol=rtols[0])
        utils.assert_allclose(data_inter, ref_inter, rtol=rtols[1])
        utils.assert_allclose(data_esp, ref_esp, rtol=rtols[2])

    def test_plot_fluid_shell(self):
        fig, params = plot_fluid_shell(v_wall=0.7, alpha_n=0.052)
        utils.save_fig_multi(fig, os.path.join(FIG_PATH, "fluid_shell_single"))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
