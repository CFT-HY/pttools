"""
Entropy (old reference)
=======================

Created on Fri Jul  2 18:20:37 2021

@author: hindmars

Requires input data as a npz file.
"""

# import sys
# sys.path.append('../pttools/')

import pttools.bubble as b
import matplotlib.pyplot as plt
import numpy as np

n_alpha = 10
n_vw = 10

# g_bro = eos.G_BRO_DEFAULT*0.5
# g_sym = eos.G_SYM_DEFAULT

g_bro = 120
g_sym = 123


file_name = 's_change_gbro{:3.0f}_g_sym{:3.0f}_nalpha_{}_nvw_{}.npz'.format(g_bro, g_sym, n_alpha, n_vw)
d = np.load(file_name)

ds_arr = d['arr_0']
vw_arr = d['arr_1']
alpha_arr = d['arr_2']

fig, ax = plt.subplots()

min_level = -0.3
max_level = 0.4
diff_level = 0.05

n_min = int(min_level/diff_level)
n_max = int(max_level/diff_level)

levels = np.linspace(n_min, n_max, n_max - n_min + 1, endpoint=True )*diff_level

cmap_neg = plt.cm.get_cmap("Blues")
cmap_pos = plt.cm.get_cmap("Reds")

cols = list( cmap_neg((levels[levels <0]-diff_level)/(min_level-diff_level)) ) + list(cmap_pos((levels[levels >= 0]+diff_level)/(max_level+diff_level)))

cs = ax.contourf(vw_arr, alpha_arr, ds_arr, levels, colors=cols)

cbar = fig.colorbar(cs)
cbar.ax.set_ylabel(r'$\Delta S/S$')

# ax.plot(b.min_speed_deton(alpha_arr), alpha_arr, 'k--', label=r'$v_{\rm J}$')
ax.plot(b.v_chapman_jouguet_bag(alpha_arr), alpha_arr, 'k--', label=r'$v_{\rm J}$')
# ax.plot(vw_arr, b.alpha_n_max(vw_arr), 'k', label=r'$\alpha_{\rm max}$', linewidth=2)

ax.grid()
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xlabel(r'$v_{\rm w}$')
ax.set_ylabel(r'$\alpha$')
plt.legend()

plt.show()
