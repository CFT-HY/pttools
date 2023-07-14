"""
Bag model
=========

Simple plots for the bag model
"""

import os.path

import matplotlib.pyplot as plt

from examples import utils
from pttools.logging import setup_logging
from pttools.bubble.fluid_bag import fluid_shell_bag
from pttools.bubble.fluid_reference import ref
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel
from pttools.bubble import props
from pttools.bubble.bubble import Bubble
from pttools.bubble.boundary import SolutionType

setup_logging()

model = BagModel(g_s=123, g_b=120, V_s=0.9)
# model = BagModel(a_s=13.488, a_b=13.159, V_s=0.900, V_b=0.000)
# model = ConstCSModel(css2=0.323, csb2=0.322, g_s=123, g_b=120, V_s=0.9)

v_wall = 0.85
alpha_n = 0.05

print("Solving with old solver")
v, w, xi = fluid_shell_bag(v_wall, alpha_n)
# print(v, w, xi)

print("Solving with new solver")
bubble = Bubble(model, v_wall=v_wall, alpha_n=alpha_n)
bubble.solve()
print(bubble.sol_type)
ref_data = ref().get(v_wall, alpha_n, bubble.sol_type)
print(
    f"vp={ref_data[0]}, vm={ref_data[1]}, "
    f"vp_tilde={ref_data[2]}, vm_tilde={ref_data[3]}, "
    f"wp={ref_data[4]}, wm={ref_data[5]}"
)

fig: plt.Figure = plt.figure()
ax1, ax2, ax3, ax4 = fig.subplots(4, 1, sharex=True)

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

utils.save_and_show(fig, "bag.png")
