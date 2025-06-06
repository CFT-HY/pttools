"""
wn by w_center
==============

For debugging the solver
"""

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
from pttools.bubble import fluid
from pttools.bubble.boundary import SolutionType
from pttools.bubble.chapman_jouguet import v_chapman_jouguet
from pttools.bubble.fluid_reference import ref
# from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel


model = ConstCSModel(css2=1/3 - 0.01, csb2=1/3 - 0.011, g_s=123, g_b=120, V_s=0.9)
# model = BagModel(g_s=123, g_b=120, V_s=0.9)

v_wall = 0.40454545454545454
alpha_n = 0.2534507678410117


vp_bag, vm_bag, vp_tilde_bag, vm_tilde_bag, wp_bag, wm_bag = ref().get(v_wall, alpha_n, SolutionType.SUB_DEF)
wn = model.wn(alpha_n)
wm_bag *= wn
wp_bag *= wn

w_center = np.linspace(0.9*wm_bag, 1.1*wm_bag)
wn_est = np.empty_like(w_center)
for i, w_center_i in enumerate(w_center):
    v_cj = v_chapman_jouguet(model, alpha_n=alpha_n, wn=wn)
    v, w, xi, vp, vm, vp_tilde, vm_tilde, xi_sh, vm_sh, vm_tilde_sh, wp, wn_estimate, wm_sh = \
        fluid.sound_shell_deflagration(model, v_wall=v_wall, wn=wn, w_center=w_center_i, cs_n=model.css2, v_cj=v_cj, vp_guess=vp_bag, wp_guess=wp_bag)
    wn_est[i] = wn_estimate


fig: plt.Figure = plt.figure()
ax: plt.Axes = fig.add_subplot()
ax.plot(w_center, wn_est)
ax.axhline(wn)
ax.set_xlabel("$w_{center}$")
ax.set_ylabel("w_n")

utils.save_and_show(fig, "w_by_w.png")
