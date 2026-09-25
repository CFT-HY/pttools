"""
Standard Model
==============

Figures for the Standard Model
"""

from matplotlib.figure import Figure
import numpy as np

from examples.utils import save_and_show_figs
from pttools.analysis.g_cs2 import plot_g_cs2
from pttools.analysis.plot_model import ModelPlot
from pttools.analysis.plot_thermomodels import ThermoModelsPlot
from pttools.bubble.phase import Phase
from pttools.models.full import FullModel
from pttools.models.sm import StandardModel
import pttools.type_hints as th

# %%
# g_eff
# -----
# The T_min parameter is for silencing log spam
thermo: StandardModel = StandardModel(silence_temp=True)
fig: Figure = plot_g_cs2(thermo)

# %%
# Thermodynamics
# --------------

# thermo = StandardModel(V_s=1.3, g_mult_s=1.3)
temp: th.FloatArr1D = np.logspace(thermo.GEFF_DATA_LOG_TEMP[0], thermo.GEFF_DATA_LOG_TEMP[-1], 100)

plot: ThermoModelsPlot = ThermoModelsPlot(temp)
plot.add(thermo, Phase.SYMMETRIC)
# plot.add(thermo, Phase.BROKEN)
plot.process()

# %%
# FullModel based on StandardModel
# --------------------------------
# thermo2 = StandardModel(V_s=5e12, g_mult_s=1 + 1e-9)
thermo2: StandardModel = StandardModel(V_s=5e15, g_mult_s=1 + 1e-5, silence_temp=True)
model2: FullModel = FullModel(thermo2)
plot2: ModelPlot = ModelPlot(model2)
print(model2.T_crit, model2.T_max, model2.alpha_n_min, model2.w_crit)


if __name__ == "__main__":
    save_and_show_figs({
        "standard_model": fig,
        "standard_model_thermo": plot2.fig
    })
