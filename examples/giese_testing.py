"""
Giese testing
=============

Test comparison with Giese code and data
"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.models import ConstCSModel
from examples.utils import save_and_show

model = ConstCSModel(css2=1/3, csb2=1/4, a_s=5, a_b=1, V_s=1)
atbs = np.linspace(0.01, 3, 30)
alpha_ns = np.zeros_like(atbs)
for i, alpha_theta_bar_n in enumerate(atbs):
    try:
        alpha_ns[i] = model.alpha_n_from_alpha_theta_bar_n(alpha_theta_bar_n=alpha_theta_bar_n)
    except RuntimeError:
        pass

print(alpha_ns)
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(atbs[alpha_ns < atbs], alpha_ns[alpha_ns < atbs])
ax.plot(atbs[alpha_ns >= atbs], alpha_ns[alpha_ns >= atbs], color="red")

ax.set_xlabel(r"$\alpha_{\bar{\theta}_n}$")
ax.set_ylabel(r"$\alpha_n$")
ax.grid()

save_and_show(fig, "giese_testing.png")
# plt.show()
