"""
ConstCSModel
============

Plot various quantities for the constant sound speed model
"""

import numpy as np

from examples import utils
from pttools.models.const_cs import ConstCSModel
from pttools.analysis.plot_model import ModelPlot


def main() -> ModelPlot:
    csb = 1 / np.sqrt(3) - 0.01
    const_cs = ConstCSModel(a_s=1.5, a_b=1, css2=1/3, csb2=csb**2, V_s=1)
    return ModelPlot(const_cs, t_log=False, y_log=False)


if __name__ == "__main__" and "__file__" in globals():
    plot = main()
    utils.save_and_show(plot.fig, "const_cs.png")
