from __future__ import absolute_import, division, print_function
# Eikr only


import sys
import bubble
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, newton
import EIKR_Toolbox as Eos
from new_general_shooting_solution import fluid_minus, exit_speed_wall, root_estimate


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

    return wplus, vplus_wall


def find_xi_shock(v, w, xi):
    # Finds shock by working through known xi, w, v values solving v-cs^2, and searching for a sign change in
    # the values
    values = np.zeros(len(xi))
    for i in range(0, len(xi)):
        # print(i)
        values[-i-1] = bubble.lorentz(xi[-i-1], v[-i-1])*xi[-i-1] - Eos.cs2_w(w[-i-1], 0)
        # print(v[-i-1], xi[-i-1], Eos.cs2_w(w[-i-1], 0), w[-i-1], values[-i-1])
        if values[-i-1]*values[-i] < 0:
            shock_index = i+1
            break
    # print(values)
    print(shock_index)
    return bubble.lorentz(xi[-shock_index], v[-shock_index]), w[-shock_index], xi[-shock_index]


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

    w_plus, v_plus_wall = find_wall_frame_sym_vars(w_minus, xi_w)
    v_plus = bubble.lorentz(xi_w, v_plus_wall)
    v_array, w_array, xi_array = bubble.fluid_shell_param(v_plus, w_plus, xi_w, direction=-1)
    v_s_minus, w_s_minus, xi_s = find_xi_shock(v_array, w_array, xi_array)
    v_s_plus = 1./(3. * v_s_minus)
    w_s_plus = w_s_minus * bubble.gamma2(v_s_minus)*v_s_minus/(bubble.gamma2(v_s_plus)*v_s_plus)
    print(w_s_plus)

