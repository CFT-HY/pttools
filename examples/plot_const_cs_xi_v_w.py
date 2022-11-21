"""
ConstCSModel
====================

Constant sound speed model
"""

import numpy as np

from pttools.analysis.plot_fluid_shell_3d import BubblePlot3D
from pttools.bubble.boundary import SolutionType
from pttools.bubble.bubble import Bubble
# from pttools.logging import setup_logging
from pttools.models.const_cs import ConstCSModel


def main():
    bag = ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=1/3, V_s=1)
    # css = 1/np.sqrt(3) - 0.05
    # csb = 1/np.sqrt(3) - 0.1
    # print(f"css={css}, csb={csb}")
    csb = 1/np.sqrt(3) - 0.05
    const_cs = ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=csb**2, V_s=1)
    plot = BubblePlot3D(model=const_cs)

    plot.add(Bubble(bag, v_wall=0.85, alpha_n=0.05, sol_type=SolutionType.DETON), color="blue")
    plot.add(Bubble(bag, v_wall=0.5, alpha_n=0.6, sol_type=SolutionType.SUB_DEF), color="blue")

    plot.add(Bubble(const_cs, v_wall=0.95, alpha_n=0.15, sol_type=SolutionType.DETON), color="red")
    plot.add(Bubble(const_cs, v_wall=0.4, alpha_n=0.6, sol_type=SolutionType.SUB_DEF), color="red")

    return plot.create_fig()


# setup_logging()
main()
