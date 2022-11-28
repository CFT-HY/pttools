import matplotlib.pyplot as plt

from pttools.analysis.plot_model import ModelPlot
from pttools.models.full import FullModel
from pttools.models.sm import StandardModel


if __name__ == "__main__":
    thermo = StandardModel(V_s=2, g_mult_s=1 + 1e-9)
    ModelPlot(FullModel(thermo))

    plt.show()
