"""
Bag model
=========

Simple plots for the bag model
"""

import matplotlib.pyplot as plt

from pttools.models.bag import BagModel
# from pttools.bubble.fluid import fluid_shell_generic
from pttools.bubble import props
from pttools.bubble.bubble import Bubble

model = BagModel(g_s=123, g_b=120, V_s=0.9)

bubble = Bubble(model, v_wall=0.05, alpha_n=0.85)
bubble.solve()
print(bubble.sol_type)

fig = plt.figure()
ax1, ax2, ax3, ax4 = fig.subplots(4, 1)

phase = props.find_phase(bubble.xi, bubble.v_wall)
theta = model.theta(bubble.w, phase)

ax1.plot(bubble.xi, bubble.v)
ax2.plot(bubble.xi, bubble.w - bubble.w[-1])
ax3.plot(bubble.xi, phase)
ax4.plot(bubble.xi, theta - theta[-1])

print(bubble.kappa)
print(bubble.omega)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlim(0, 1)
ax1.set_ylim(0, 1)

plt.show()
