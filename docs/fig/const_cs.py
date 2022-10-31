import matplotlib.pyplot as plt

from pttools.models.const_cs import ConstCSModel
from pttools.analysis.plot_model import ModelPlot


if __name__ == "__main__":
    ModelPlot(ConstCSModel(a_s=1.1, a_b=1, css2=0.4**2, csb2=1/3, V_s=1))
    plt.show()
