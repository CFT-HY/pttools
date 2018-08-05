from __future__ import absolute_import, division, print_function
# Eikr only


import sys
import bubble
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import fsolve
import EIKR_Toolbox as Eos


def df_dtau(z, t):
    # Returns differentials in parametric form, suitable for odeint
    v = z[0]
    xi = z[1]
    w = z[2]

    f1 = 2 * v * (Eos.cs(w) ** 2) * (1 - (v ** 2)) * (1 - (xi * v))  # dv/dt
    f2 = xi * (((xi - v) ** 2) - (Eos.cs(w) ** 2) * ((1 - (xi * v)) ** 2))  # dxi/dt
    f3 = ((2 * v * w / (xi * (xi - v))) * f2 + ((w / (1 - v ** 2)) * (((1 - v * xi) / (xi - v))
                                                                      + ((xi - v) / (1 - xi * v)))) * f1)  # dw/dt

    return [f1, f2, f3]


def find_wall_frame_sym_vars(wminus, vminus_wall):
    # Make guess at eps_plus using standard Weak conditions
    eps_plus = Eos.epsilon(Eos.Tn, 0)
    eps_minus = Eos.epsilon_w(wminus)

    Q = wminus * (bubble.gamma2(vminus_wall)) * vminus_wall
    E = (wminus * (bubble.gamma2(vminus_wall)) * (vminus_wall ** 2) + 0.25*wminus - eps_minus)
    a = 3 * Q / 4
    b = -(E + eps_plus)
    c = Q / 4

    vplus_wall = (-b - ((b ** 2) - 4 * a * c) ** 0.5) / (2 * a)
    wplus = Q * (1 - vplus_wall ** 2) / vplus_wall

    def em_eqns(x, Q_minus, E_minus):
        v = x[0]
        w = x[1]
        Q_plus = w * bubble.gamma2(v) * v
        mom_con = Q_plus - Q_minus
        en_con = E_minus - Q_plus * v - Eos.p_w(w, 0)
        return [mom_con, en_con]

    if type(vplus_wall) is np.ndarray:
        vplus_wall[np.where(isinstance(vplus_wall, complex))] = np.nan
        len_v = len(vplus_wall)
        # print('starting fsolve for wall speed')
        for i in range(0, len_v):
            if not np.isnan(vplus_wall[i]):
                est = [vplus_wall[i], wplus[i]]
                fluid_wall = fsolve(em_eqns, est, args=(Q[i], E[i]))
                vplus_wall[i] = fluid_wall[0]
                wplus[i] = fluid_wall[1]

    else:
        if type(vplus_wall) is complex:
            vplus_wall = np.nan
        else:
            est = [vplus_wall, wplus]
            fluid_wall = fsolve(em_eqns, est, args=(Q, E))
            vplus_wall = fluid_wall[0]
            wplus = fluid_wall[1]

    vplus = bubble.lorentz(xi_w, vplus_wall)
    return wplus, vplus


def fluid_shell_param(v0, w0, xi0, N=1000):
    # Integrates parametric fluid equations from an initial condition
    t = np.linspace(0., -50., N)
    if isinstance(xi0, np.ndarray):
        soln = odeint(df_dtau, (v0[0], xi0[0], w0[0]), t)
    else:
        soln = odeint(df_dtau, (v0, xi0, w0), t)
    v = soln[:, 0]
    xi = soln[:, 1]
    w = soln[:, 2]

    return v, w, xi


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('usage: %s <xi_wall> <w_minus> \n' % sys.argv[0])
        sys.exit(1)
    xi_w = float(sys.argv[1])
    if not 0. < xi_w < 1.:
        while not 0. < xi_w < 1.:
            print('Error: xi_wall must be satisfy 0 < xi_wall < 1')
            xi_w = input('Enter xi_wall ')
    w_minus = float(sys.argv[2])

    w_plus, v_plus = find_wall_frame_sym_vars(w_minus, xi_w)
    print(w_plus, v_plus)
    v_array, w_array, xi_array = fluid_shell_param(v_plus, w_plus, xi_w)
