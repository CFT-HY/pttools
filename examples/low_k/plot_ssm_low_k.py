"""low-k approximation"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pttools.ssmtools.low_k import pow_gw_approximation, pow_gw_junction, power_spectrum_integration_low, power_spectrum_integration_int

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.omgw0 import Spectrum


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

font_size = 20
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams.update({'lines.linewidth': 1.5})
mpl.rcParams.update({'axes.linewidth': 1.})
mpl.rcParams.update({'axes.labelsize': font_size+1})
mpl.rcParams.update({'xtick.labelsize': font_size})
mpl.rcParams.update({'ytick.labelsize': font_size})
# but make legend smaller
mpl.rcParams.update({'legend.fontsize': 16})




"Choose study parameters"

npt = 2000 # numebr of points in the spectrum

# fig, ax = plt.subplots(1,1, figsize =(12, 7))
fig, ax = plt.subplots(1,1, figsize =(9, 5))

"Choose fixed parameters"
omega = 1/3
cs = np.sqrt(omega)
nu = (1- 3*cs**2)/(1+ 3*cs**2)
vw = 0.8
alpha = 0.01
kp = 200*np.pi # kp*eta_star
HR = 2*np.pi/kp * 2/(1+3*omega)
Lf = 2*np.pi/kp

# INFO
# Lf = R_star
# HR = H_star * R_star = r_star
# H_star =
# kp = kappa_peak * eta_star
# eta_star is set to 1 = initial time of the acoustic phase (this is a normalization choice)
# R_* = 2pi/kp (2pi is arbitrary)
# When plotting over z, we have the peak at around z = 1
# 2/(1+3*omega) comes from the Friedmann equation
# Use the existing PTtools definition for omega

# omega = ?

# r_star = H_star R_star
# eta_star is set to 1
# -> k_peak * eta_star = k_peak = 2pi / r_star * 2 / (1 + 3*omega)
# -> Lf = 2pi / k_peak
# -> Ht
# -> eta_end, tau-star, tau_end

# Mark y = Lorenzo z
# Mark z = Lorenzo x

model = BagModel(alpha_n_min=alpha)
bubble = Bubble(model, v_wall=vw, alpha_n=alpha)
ubarf2 = bubble.ubarf2

Ht = HR/bubble.ubarf
eta_star = 1
eta_end = eta_star + Ht
tau_star = eta_star/Lf
tau_end = eta_end/Lf
# kp = 2*np.pi /HR

z = np.logspace(-3, 3, npt) # z = kR_*




eps = 1.e-10
xmax = max(z) * (0.5 * (1. + cs) / cs) + eps
xmin = min(z) * (0.5 * (1. - cs) / cs) - eps
x = np.logspace(np.log10(xmin), np.log10(xmax), npt)
# x = np.logspace(-2., 9, 2000)

x_inf = np.logspace(-3, 3, npt)

spectrum_x = Spectrum(bubble, y=x, r_star=HR)
spectrum_inf = Spectrum(bubble, y=x_inf, r_star=HR)
spectrum_z = Spectrum(bubble, y=z, r_star=HR)


# sd_v = spec_den_v(x, params, npt, filename, skip, method, de_method, z_st_thresh)
# sd_gw, y = spec_den_gw_scaled(x, sd_v, z)

# Pgw_approx = z**3/2/np.pi**2 * HR * Ht * Pgw_approximation(z, spectrum_x.spec_den_v, cs=cs, tau_star=tau_star, tau_end=tau_end)

#The above is equal to the following
# Pv_exp = spec_den_v_bag(x, params_v)
# Pv_inf = spec_den_v_bag(x_inf, params_v)
Pv_exp = spectrum_x.spec_den_v
Pv_inf = spectrum_inf.spec_den_v
# Pgw_high = HR * Ht * power_gw_scaled_bag(z, (vw, alpha))
Pgw_high = HR * Ht * spectrum_z.pow_gw
# Pgw_low = 4/3* z**3/2/np.pi**2 * HR * Ht * power_spectrum_integration_low(x_inf, Pv_inf, z, cs=cs, tau_star=tau_star, tau_end=tau_end)
# Pgw_int = 4/3* z**3/2/np.pi**2 * HR * Ht * power_spectrum_integration_int(x, Pv_exp, z, cs=cs, tau_star=tau_star)
# Pgw_approx_2 = pow_gw_junction(z, Pgw_low, Pgw_int, Pgw_high, cs=cs, tau_star=tau_star, tau_end=tau_end)


# Pgw_exp = HR* Ht * ssmtools.power_gw_scaled(z, params_v)
# ax.plot(z, Pgw_high, color = 'blue', linestyle = '--', label = r'high-freq appox: PTtools')
ax.plot(z, HR * Ht * spectrum_z.pow_gw_ssm, color = 'blue', linestyle = '--', label = r'high-freq appox: PTtools')

# ax.plot(z, Pgw_peak, color = 'red', linestyle = '--', label = r'Lorenzo integrator')
# ax.plot(z, Pgw_low, color = 'orange', linestyle = '--', label = r'low-freq approx')
# ax.plot(z, Pgw_int, color = 'cyan', linestyle = '--', label = r'intermediate-freq approx')
# ax.plot(z, Pgw_approx_2, color = 'k', linestyle = '-', label = r'full approx')
ax.plot(z, HR * Ht * spectrum_z.pow_gw_low, color = 'orange', linestyle = '--', label = r'low-freq approx')
ax.plot(z, HR * Ht * spectrum_z.pow_gw_int, color = 'cyan', linestyle = '--', label = r'intermediate-freq approx')
ax.plot(z, HR * Ht * spectrum_z.pow_gw, color = 'k', linestyle = '-', label = r'full approx')
# ax.set_title(r'$\alpha = {:5.2f}, \; \xi_w = {:5.1f}, \; \eta_{{\rm end}} = {:5.3f}, \; H_*R_* = {:5.2f}, \; {{\bar U}} = {:5.3f}, \; H_*\tau_v = {:5.3f}$'.format(alpha, vw, eta_end, HR, bubble.ubarf, Ht), fontsize=font_size-1)
ax.set_title(r'$\alpha = {:5.2f}, \; \xi_w = {:5.1f}, \; \eta_{{\rm end}} = {:5.3f}, \; H_*R_* = {:5.2f}, \; {{\bar U}} = {:5.3f}, \; H_*\tau_v = {:5.3f}$'.format(alpha, vw, eta_end, HR, bubble.ubarf, Ht), fontsize=font_size-2)


"Axes customization"

ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(linestyle ='dotted')
ax.set_ylim(bottom = 1e-19, top = 1e-10)
ax.set_xlim(left = 1e-3, right = 2e+2)
# ax.legend(loc = 'upper left')
ax.set_ylabel(r'$\mathcal{P}_{{\rm gw}}(kR_*) $', labelpad = 10)
ax.legend(loc = 'upper left')
# fig.legend(bbox_to_anchor=(1., 0.5), loc='center right')
# fig.subplots_adjust(bottom=None, top=None, left=None, right=0.75, hspace=None, wspace=None)


# fig.savefig('/Users/giombilo/Desktop/ssm_simpson.pdf')
fig.tight_layout()
fig.savefig("low_k.svg")
plt.show()
