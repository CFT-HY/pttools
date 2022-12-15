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
from pttools.bubble import quantities
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

# v_wall and alpha_n values are from Hindmarsh and Hijazi, 2019.
bag_def = Bubble(bag, v_wall=0.5, alpha_n=0.578, sol_type=SolutionType.SUB_DEF)
bag_hybrid = Bubble(bag, v_wall=0.7, alpha_n=0.151, sol_type=SolutionType.HYBRID)
bag_det = Bubble(bag, v_wall=0.77, alpha_n=0.091, sol_type=SolutionType.DETON)

for bubble in [bag_def, bag_hybrid, bag_det]:
    plot.add(bubble, color="blue")
    print(
        f"{bubble.label_unicode}: w0/wn={bubble.w[0] / bubble.wn}, "
        f"Ubarf2={bubble.ubarf2}, K={bubble.kinetic_energy_fraction}, kappa={bubble.kappa}, omega={bubble.omega}, "
        f"Trace anomaly={bubble.trace_anomaly}"
    )
    kappa = quantities.get_kappa(v_wall=bubble.v_wall, alpha_n=bubble.alpha_n)
    print(f"Reference kappa={kappa}")

plot.add(Bubble(const_cs, v_wall=0.5, alpha_n=0.578, sol_type=SolutionType.SUB_DEF), color="red")
plot.add(Bubble(const_cs, v_wall=0.7, alpha_n=0.151, sol_type=SolutionType.HYBRID), color="red")
# These values had to be modified for a solution to exist
plot.add(Bubble(const_cs, v_wall=0.8, alpha_n=0.1, sol_type=SolutionType.DETON), color="red")

plot.save(os.path.join(FIG_DIR, "plot_const_cs_xi_v_w"))
if __name__ == "__main__" and "__file__" in globals():
    plot.show()
plot.fig()
