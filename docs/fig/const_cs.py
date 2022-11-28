import matplotlib.pyplot as plt

import numpy as np

from pttools.models.const_cs import ConstCSModel
from pttools.analysis.plot_model import ModelPlot


if __name__ == "__main__":
    csb = 1 / np.sqrt(3) - 0.01
    const_cs = ConstCSModel(a_s=1.5, a_b=1, css2=1/3, csb2=csb**2, V_s=1)
    ModelPlot(const_cs, t_log=False, y_log=False)

    plt.show()
