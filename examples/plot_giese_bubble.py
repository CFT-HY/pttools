r"""
Giese bubble
============

Plot a single bubble using parameters from the code of :giese_2021:`\ `
"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from examples.utils import save
from pttools.bubble import Bubble
from pttools.models import ConstCSModel

logger = logging.getLogger(__name__)

model = ConstCSModel(css2=1/3, csb2=1/3, a_s=5, a_b=1, V_s=1)
alpha_thetabar_ns = np.array([0.01, 0.03, 0.1, 0.3, 1, 3])
colors = ["b", "y", "r", "g", "purple", "grey"]
v_wall = 0.8

fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot()

for alpha_tbn, color in zip(alpha_thetabar_ns, colors):
    try:
        bubble = Bubble(model=model, v_wall=v_wall, alpha_n=alpha_tbn, theta_bar=True, allow_invalid=True)
        bubble.solve()
        ax.plot(bubble.xi, bubble.v, color=color)
    except (RuntimeError, ValueError) as e:
        logger.exception("ERROR:", exc_info=e)


save(fig, "giese_bubble")
