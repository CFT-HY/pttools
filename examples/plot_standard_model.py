"""
Standard Model
==============

Figures for the Standard Model
"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.g_cs2 import plot_g_cs2

from pttools.analysis.plot_model import ModelPlot
from pttools.analysis.plot_thermomodels import ThermoModelsPlot
from pttools.bubble.boundary import Phase
from pttools.models.full import FullModel
from pttools.models.sm import StandardModel

# %%
# g_eff
# -----
thermo = StandardModel()
fig = plot_g_cs2(thermo)

# %%
# Thermodynamics
# --------------

# thermo = StandardModel(V_s=1.3, g_mult_s=1.3)
temp = np.logspace(thermo.GEFF_DATA_LOG_TEMP[0], thermo.GEFF_DATA_LOG_TEMP[-1], 100)

plot = ThermoModelsPlot(temp)
plot.add(thermo, Phase.SYMMETRIC)
# plot.add(thermo, Phase.BROKEN)
plot.process()

# %%
# FullModel based on StandardModel
# --------------------------------
thermo2 = StandardModel(V_s=2, g_mult_s=1 + 1e-9)
plot2 = ModelPlot(FullModel(thermo2))

if __name__ == "__main__":
    plt.show()
