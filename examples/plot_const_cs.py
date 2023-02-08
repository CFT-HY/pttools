"""
ConstCSModel
============

Plot various quantities for the constant sound speed model
"""

import matplotlib.pyplot as plt

import numpy as np

from pttools.models.const_cs import ConstCSModel
from pttools.analysis.plot_model import ModelPlot


def main():
    csb = 1 / np.sqrt(3) - 0.01
    const_cs = ConstCSModel(a_s=1.5, a_b=1, css2=1/3, csb2=csb**2, V_s=1)
    ModelPlot(const_cs, t_log=False, y_log=False)


if __name__ == "__main__" and "__file__" in globals():
    main()
    plt.show()
