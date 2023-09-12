r"""
ConstCSModel 3D
===============

Plot $\xi, v, w$ in 3D for the constant sound speed model
"""

import os.path

import numpy as np

from examples.utils import FIG_DIR
from pttools.analysis.plot_fluid_shell_3d import BubblePlot3D
from pttools.bubble.boundary import SolutionType
from pttools.bubble.bubble import Bubble
from pttools.bubble import quantities
# from pttools.logging import setup_logging
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel


# setup_logging()

def main():
    bag = BagModel(a_s=1.1, a_b=1, V_s=1)
    # css = 1/np.sqrt(3) - 0.05
    # csb = 1/np.sqrt(3) - 0.1
    # print(f"css={css}, csb={csb}")
    csb = 1/np.sqrt(3) - 0.01
    const_cs = ConstCSModel(a_s=1.5, a_b=1, css2=1/3, csb2=csb**2, V_s=1)
    plot = BubblePlot3D(model=const_cs)

    # v_wall and alpha_n values are from Hindmarsh and Hijazi, 2019.
    bag_def = Bubble(bag, v_wall=0.5, alpha_n=0.578, sol_type=SolutionType.SUB_DEF)
    bag_hybrid = Bubble(bag, v_wall=0.7, alpha_n=0.151, sol_type=SolutionType.HYBRID)
    bag_det = Bubble(bag, v_wall=0.77, alpha_n=0.091, sol_type=SolutionType.DETON)

    for bubble in [bag_def, bag_hybrid, bag_det]:
        plot.add(bubble, color="blue")
        print(bubble.info_str())
        kappa = quantities.get_kappa_bag(v_wall=bubble.v_wall, alpha_n=bubble.alpha_n)
        ubarf2 = quantities.get_ubarf2_new_bag(v_wall=bubble.v_wall, alpha_n=bubble.alpha_n)
        ke_frac = quantities.get_ke_frac_bag(v_wall=bubble.v_wall, alpha_n=bubble.alpha_n)
        print(f"Reference kappa={kappa:.4f}, relative error={(bubble.kappa - kappa)/kappa}")
        print(f"Reference ubarf2={ubarf2:.4f}, relative error={(bubble.ubarf2 - ubarf2)/ubarf2}")
        print(f"Reference ke_frac_bva={ke_frac:.4f}, relative error={(bubble.bva_kinetic_energy_fraction - ke_frac)/ke_frac}")

    const_cs_def = Bubble(const_cs, v_wall=0.5, alpha_n=0.578, sol_type=SolutionType.SUB_DEF)
    const_cs_hybrid = Bubble(const_cs, v_wall=0.7, alpha_n=0.151, sol_type=SolutionType.HYBRID)
    # These values had to be modified for a solution to exist
    const_cs_det = Bubble(const_cs, v_wall=0.8, alpha_n=0.1, sol_type=SolutionType.DETON)

    for bubble in [const_cs_def, const_cs_hybrid, const_cs_det]:
        plot.add(bubble, color="red")
        print(bubble.info_str())
    return plot


plot = main()
plot.save(os.path.join(FIG_DIR, "plot_const_cs_xi_v_w"))
if __name__ == "__main__" and "__file__" in globals():
    plot.show()
plot.fig()
