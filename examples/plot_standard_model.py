"""
Standard Model
==============

Figures for the Standard Model
"""

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
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
# thermo2 = StandardModel(V_s=5e12, g_mult_s=1 + 1e-9)
thermo2 = StandardModel(V_s=5e15, g_mult_s=1 + 1e-5)
model2 = FullModel(thermo2)
plot2 = ModelPlot(model2)
print(model2.t_crit, model2.t_max, model2.alpha_n_min, model2.w_crit)

if __name__ == "__main__":
    utils.save_and_show(fig, "standard_model.png")
