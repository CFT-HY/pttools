import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.plot_thermomodels import ThermoModelsPlot
from pttools.bubble.boundary import Phase
from pttools.models.sm import StandardModel


def main():
    thermo = StandardModel()
    temp = np.logspace(thermo.GEFF_DATA_LOG_TEMP[0], thermo.GEFF_DATA_LOG_TEMP[-1], 100)

    comp = ThermoModelsPlot(temp)
    comp.add(thermo, Phase.SYMMETRIC)
    # comp.add(thermo, Phase.BROKEN)
    comp.process()


if __name__ == "__main__":
    main()
    plt.show()
