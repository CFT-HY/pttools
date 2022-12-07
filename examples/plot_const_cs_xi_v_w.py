"""
ConstCSModel
============

Constant sound speed model
"""

import os.path

import numpy as np

from examples.utils import FIG_DIR
from pttools.analysis.plot_fluid_shell_3d import BubblePlot3D
from pttools.bubble.boundary import SolutionType
from pttools.bubble.bubble import Bubble
# from pttools.logging import setup_logging
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel


# setup_logging()

bag = BagModel(a_s=1.1, a_b=1, V_s=1)
# css = 1/np.sqrt(3) - 0.05
# csb = 1/np.sqrt(3) - 0.1
# print(f"css={css}, csb={csb}")
csb = 1/np.sqrt(3) - 0.01
const_cs = ConstCSModel(a_s=1.5, a_b=1, css2=1/3, csb2=csb**2, V_s=1)
plot = BubblePlot3D(model=const_cs)

plot.add(Bubble(bag, v_wall=0.4, alpha_n=0.1, sol_type=SolutionType.SUB_DEF), color="blue")
plot.add(Bubble(bag, v_wall=0.7, alpha_n=0.1, sol_type=SolutionType.HYBRID), color="blue")
plot.add(Bubble(bag, v_wall=0.8, alpha_n=0.1, sol_type=SolutionType.DETON), color="blue")

plot.add(Bubble(const_cs, v_wall=0.45, alpha_n=0.6, sol_type=SolutionType.SUB_DEF), color="red")
plot.add(Bubble(const_cs, v_wall=0.7, alpha_n=0.4, sol_type=SolutionType.HYBRID), color="red")
plot.add(Bubble(const_cs, v_wall=0.95, alpha_n=0.30, sol_type=SolutionType.DETON), color="red")

plot.save(os.path.join(FIG_DIR, "plot_const_cs_xi_v_w"))
if __name__ == "__main__" and "__file__" in globals():
    plot.show()
plot.fig()
