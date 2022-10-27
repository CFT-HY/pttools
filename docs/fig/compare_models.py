import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.plot_models import ModelsPlot
from pttools import models


def main():
    model_bag = models.BagModel(a_s=1.1, a_b=1, V_s=1)
    # model_const_cs_like_bag = models.ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=1/3, V_s=1)
    model_thermo_bag = models.FullModel(thermo=models.ConstCSThermoModel(a_s=1.1, a_b=1, V_s=1, css2=1/3.3, csb2=1/3))
    model_sm = models.FullModel(thermo=models.StandardModel())

    comp = ModelsPlot(temp=np.logspace(1, 3, 100))

    comp.add(model_bag, phase=models.Phase.SYMMETRIC)
    comp.add(model_thermo_bag, phase=models.Phase.SYMMETRIC, ls="--")
    comp.add(model_sm, phase=models.Phase.SYMMETRIC, ls=":")
    comp.process()


if __name__ == "__main__":
    main()
    plt.show()
