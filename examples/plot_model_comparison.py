"""
Comparison of BagModel and ConstCSModel
=======================================
"""

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
from pttools.analysis.plot_models import ModelsPlot
from pttools import models


def main():
    model_bag = models.BagModel(a_s=1.1, a_b=1, V_s=1)
    # model_const_cs_like_bag = models.ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=1/3, V_s=1)
    model_thermo_bag = models.FullModel(thermo=models.ConstCSThermoModel(a_s=1.1, a_b=1, V_s=1, css2=1/3.3, csb2=1/3))
    model_sm = models.FullModel(thermo=models.StandardModel(), allow_invalid=True)

    plot = ModelsPlot(temp=np.logspace(1, 3, 100))

    plot.add(model_bag, phase=models.Phase.SYMMETRIC)
    plot.add(model_thermo_bag, phase=models.Phase.SYMMETRIC, ls="--")
    plot.add(model_sm, phase=models.Phase.SYMMETRIC, ls=":")
    plot.process()
    return plot


if __name__ == "__main__":
    plot = main()
    utils.save_and_show(plot.fig, "model_comparison.png")
