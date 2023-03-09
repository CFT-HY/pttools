"""
Bag model
=========

Simple plots for the bag model
"""

import matplotlib.pyplot as plt

from pttools.logging import setup_logging
from pttools.bubble.fluid_bag import fluid_shell
from pttools.models.bag import BagModel
from pttools.bubble import props
from pttools.bubble.bubble import Bubble

setup_logging()

model = BagModel(g_s=123, g_b=120, V_s=0.9)

v_wall = 0.85
alpha_n = 0.05

print("Solving with old solver")
v, w, xi = fluid_shell(v_wall, alpha_n)
# print(v, w, xi)

print("Solving with new solver")
bubble = Bubble(model, v_wall=v_wall, alpha_n=alpha_n)
bubble.solve()
print(bubble.sol_type)

fig: plt.Figure = plt.figure()
ax1, ax2, ax3, ax4 = fig.subplots(4, 1)

phase = props.find_phase(bubble.xi, bubble.v_wall)
theta = model.theta(bubble.w, phase)

ax1.plot(bubble.xi, bubble.v, label="new")
ax1.plot(xi, v, label="old", ls=":")
ax1.set_ylabel("v")
ax1.legend()

ax2.plot(bubble.xi, bubble.w, label="new")
ax2.plot(xi, w * bubble.wn, label="old", ls=":")
ax2.set_ylabel("$w$")
ax2.legend()

ax3.plot(bubble.xi, phase)
ax3.set_ylabel(r"$\phi$")

ax4.plot(bubble.xi, theta - theta[-1])
ax4.set_ylabel(r"$\theta - \theta_n$")

print("Kappa:", bubble.kappa)
print("Omega:", bubble.omega)

for ax in [ax1, ax2, ax3, ax4]:
    ax.set_xlabel(r"$\xi$")
    ax.set_xlim(0, 1)
ax1.set_ylim(0, 1)
fig.tight_layout()

plt.show()
