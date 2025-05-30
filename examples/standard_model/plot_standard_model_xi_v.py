r"""
Standard Model xi, v
====================

Example that the Standard Model can be used with the bubble solver
"""

import matplotlib.pyplot as plt

from pttools.bubble.boundary import Phase
from pttools.bubble.bubble import Bubble
from pttools.logging import setup_logging
from pttools.models.full import FullModel
from pttools.models.sm import StandardModel

from examples import utils

setup_logging()

sm = StandardModel(V_s=5e12, g_mult_s=1 + 1e-9)
model = FullModel(sm, T_crit_guess=100e3)
wn = model.wn(alpha_n=0.1)
tn = model.temp(wn, Phase.SYMMETRIC)
print(wn, tn)
bubble = Bubble(model, v_wall=0.3, alpha_n=0.05)


fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot()

ax.plot(bubble.xi, bubble.v)
ax.set_xlabel(r"$\xi$")
ax.set_ylabel("$v$")

utils.save_and_show(fig, "standard_model_xi_v.png")
