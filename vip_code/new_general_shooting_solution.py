#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jacky
Modified by Danny
"""
from __future__ import absolute_import, division, print_function

import sys
import bubble
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import fsolve
import Mechanics_Toolbox as Mech
#import EIKR_Toolbox as Eos


def set_params_sim(name, new_value=None):
    global wn, N
    if name == 'default':
        wn = 1.
        N = 1000
        set_eos('Bag')
    elif name == 'wn':
        wn = new_value
    elif name == 'N':
        N = new_value
    else:
        sys.exit('set_params_sim: params name not recognised')


def set_eos(eos_name):
    global Eos
    if eos_name == 'Bag':
        import Bag_Toolbox as Eos
    elif eos_name == 'Eikr':
        import EIKR_Toolbox as Eos
    else:
        print('set_eos: eos_name not recognised, eos unchanged')    


def cs(w, phi=None):
    return Eos.cs_w(w, phi)


def pressure(w, phi=None):
    return Eos.p_w(w, phi)


def df_dtau(z, t, c_s=cs):
    # Returns differentials in parametric form, suitable for odeint
    v = z[0]
    xi = z[1]
    w = z[2]

    f1 = 2 * v * (c_s(w) ** 2) * (1 - (v ** 2)) * (1 - (xi * v))  # dv/dt
    f2 = xi * (((xi - v) ** 2) - (c_s(w) ** 2) * ((1 - (xi * v)) ** 2))  # dxi/dt
    f3 = ((2 * v * w / (xi * (xi - v))) * f2 + ((w / (1 - v ** 2)) * (((1 - v * xi) / (xi - v))
                                                                      + ((xi - v) / (1 - xi * v)))) * f1)  # dw/dt

    return [f1, f2, f3]


def fluid_shell_param(v0, w0, xi0, N=1000, c_s=cs):
    # Integrates parametric fluid equations from an initial condition
    t = np.linspace(0., 50., N)
    if isinstance(xi0, np.ndarray):
        soln = odeint(df_dtau, (v0[0], xi0[0], w0[0]), t, args=(c_s,))
    else:
        soln = odeint(df_dtau, (v0, xi0, w0), t, args=(c_s,))
    v = soln[:, 0]
    xi = soln[:, 1]
    w = soln[:, 2]

    return v, w, xi


def fluid_from_xi_sh(xi_sh, w_n=1., N=1000, c_s=cs):
    # Integrates fluid equations back from xi_shock, enthalpy w_n
    v_minus = 0.5 * (3 * (xi_sh ** 2) - 1) / xi_sh
    w_minus = w_n * (9 * (xi_sh ** 2) - 1) / (3 * (1 - (xi_sh ** 2)))
    v, w, xi = fluid_shell_param(v_minus, w_minus, xi_sh, N, c_s)
    # Returns lower (physical) segment of curve v(xi), also w and xi there.
    n_min = np.argmin(xi)
    print('v_minus',v_minus)
    print('n_min',n_min)
    v_ls = v[0:n_min]
    w_ls = w[0:n_min]
    xi_ls = xi[0:n_min]

    return v_ls, w_ls, xi_ls


def fluid_minus(v_plus_wall, w_plus, eos='Bag'):
    # Returns v_minus, w_minus from v_plus and w_plus (wall frame)
    # i.e. solves energy-momentum conservation equations across wall
    # print ("vpluswall", v_plus_wall, "w_plus_plasma", w_plus_plasma)
    # print(eos)
    eps_plus = Eos.epsilon_w(w_plus, phi=0.0)
    eps_minus = 0.0

    #    eps_plus = Eos.call_params()[2]
    #    eps_minus = Eos.call_params()[3]
    Q = w_plus * (bubble.gamma2(v_plus_wall)) * v_plus_wall
    E = (w_plus * (bubble.gamma2(v_plus_wall)) * (v_plus_wall ** 2)
         + w_plus / 4 - eps_plus)
    a = 3 * Q / 4
    b = -(E + eps_minus)
    c = Q / 4
    v_minus_wall = (-b - ((b ** 2) - 4 * a * c) ** (0.5)) / (2 * a)

    w_minus_wall = Q * (1 - v_minus_wall ** 2) / v_minus_wall
    # print('w_minus_plasma', w_minus_plasma)
    v_minus_wall[np.where(isinstance(v_minus_wall, complex))] = np.nan
    if eos != 'Bag':
        def eqns(x0, length):
            # print('x0', x0[177])
            # print('len x0', len(x0))
            v_minus_wall_est = np.array([])
            w_minus_wall_est = np.array([])
            for i in range(0, length):
                v_minus_wall_est = np.append(v_minus_wall_est, x0[i])
            for i in range(length, 2*length):
                w_minus_wall_est = np.append(w_minus_wall_est, x0[i])
            # print('v_minus_wall', v_minus_wall)
            # print('v_minus_est ', v_minus_wall_est)
            # print('w_minus_wall', w_minus_wall)
            # print('w_minus_est ', w_minus_wall_est)
            Q_minus = w_minus_wall_est*bubble.gamma2(v_minus_wall_est)*v_minus_wall_est
            mom_con = Q-Q_minus
            en_con = E - Q_minus * v_minus_wall_est - Eos.p_w(w_minus_wall_est)
            out = np.append(mom_con, en_con)
            # print('out', out)
            return out

        est = np.array([[v_minus_wall], [w_minus_wall]])
        len_v = len(v_minus_wall)
        # len_w = len(w_minus_wall)
        # print('len v', len_v)
        # print('len w', len_w)
        # print(v_minus_wall[88])
        # print('est', est)
        wall = fsolve(eqns, est, args=(len_v))
        # print(wall)
        for i in range(0, len_v):
            v_minus_wall[i] = wall[i]
            w_minus_wall[i] = wall[i+len_v]

        #        T_plus = Eos.T_w(w_plus_plasma, 0)
        #        T_minus = opt.fsolve(Eos.delta_w, T_plus, xi_w)[0]
        #        w_minus_wall = Eos.w_minus(T_minus)
        #
        #        alpha_plus = Eos.alphaplus(T_plus)
        #        v_minus_wall = Mech.v_minus(v_plus_wall, alpha_plus)
        #   use fsolve with bag v_minus_wall, w_minus as initial guess

    return v_minus_wall, w_minus_wall

def fluid_minus_local_from_fluid_plus_plasma(v_plus_plasma, w_plus_plasma,
                                             xi_plus_plasma, eos='Bag'):
    # Finds v_minus, w_minus if wall is at range of positions xi_plus_plasma
    v_plus_wall = bubble.lorentz(xi_plus_plasma, v_plus_plasma)  # this is an array?
    v_minus_local, w_minus_local = fluid_minus(v_plus_wall, w_plus_plasma, eos)

    return v_minus_local, w_minus_local


def exit_speed_wall(xi):
    # Calculates fluid exit speed in wall frame, for wall moving
    # at speed xi. If xi < cs, it is a deflagration, and fluid
    # exits with wall speed.  If xi > cs, fluid must exit with sound speed.
    v_exit = np.zeros(len(xi))
    for i in range(0, len(xi)):
        if xi[i] < 1 / (3 ** (1 / 2)):
            v_exit[i] = xi[i]
        if xi[i] >= 1 / (3 ** (1 / 2)):
            v_exit[i] = 1 / (3 ** (1 / 2))
    return v_exit


def root_estimate(xi_ls, v_minus_local, v_exit):
    for i in range(0, len(xi_ls) - 1):
        if (v_minus_local[i] - v_exit[i] > 0 and v_minus_local[i + 1] - v_exit[i + 1] < 0) \
                or (v_minus_local[i] - v_exit[i] < 0 and v_minus_local[i + 1] - v_exit[i + 1] > 0):
            return (xi_ls[i] + xi_ls[i + 1]) / 2
        if v_minus_local[i] - v_exit[i] == 0:
            return xi_ls[i]


def fluid_at_wall(xi_shock, w_n=1, eos='Bag', N=1000, c_s=cs):
    # Integrate back from shock, return inferred wall speed and fluid variables
    # uses:  fluid_from_xi_sh
    #            fluid_minus_local_from_fluid_plus_plasma
    #            exit_speed_wall(xi_real)
    print('xi_shock, w_n, N, c_s',xi_shock, w_n, N, c_s)
    v_ls, w_ls, xi_ls = fluid_from_xi_sh(xi_shock, w_n, N, c_s)

    v_minus_local, w_minus_local = fluid_minus_local_from_fluid_plus_plasma(v_ls, w_ls,
                                                                            xi_ls, eos)
    v_exit = exit_speed_wall(xi_ls)

    rootguess_xi = root_estimate(xi_ls, v_minus_local, v_exit)
    print('rootguess_xi', rootguess_xi)
    v_minus_local_function = interp1d(xi_ls, v_minus_local)
    v_exit_function = interp1d(xi_ls, v_exit)

    def v_remainder(xi_ls, v_minus_local_function, v_exit_function):
        return v_minus_local_function(xi_ls) - v_exit_function(xi_ls)

    xi_try = fsolve(v_remainder, rootguess_xi, args=(v_minus_local_function, v_exit_function))

    return xi_try


def wall_speed_zero(xi_sh, xi_wall, w_n=1, eos='Bag', N=1000, c_s=cs):
    # Returns difference between wall speed xi_try computed from
    # xi_shock and eos and target wall speed. Suitable for use in root-finder.
    xi_try = fluid_at_wall(xi_sh, w_n, eos, N, c_s)
    return xi_try - xi_wall


def root_find_xi_sh(xi_wall, w_n=1., eos='Bag', N=1000, c_s=cs):
    # invokes root finder on wall_speed_zero to get xi_sh
    x0 = c_s(w_n, 0)
    xi_shock = opt.newton(wall_speed_zero, x0*(1.01), args=(xi_wall, w_n, eos, N, c_s))
    return xi_shock


def plot_graph(xi_wall, eos, w_n=1, N=1000):
    xi_shock = root_find_xi_sh(xi_wall, w_n, eos, N=N)
    v_ls, w_ls, xi_ls = fluid_from_xi_sh(xi_shock, w_n, N)
    v_minus_local, w_minus_local = fluid_minus_local_from_fluid_plus_plasma(v_ls, w_ls,
                                                                            xi_ls, eos)

    plt.plot(xi_ls, w_ls)
    plt.plot(xi_ls, v_ls)
    plt.plot(xi_ls, v_minus_local)
    plt.show()

    w_plot = np.zeros(len(xi_ls))
    v_plot = np.zeros(len(xi_ls))

    a = np.where(xi_ls < xi_wall)[0]
    w_plot[a[0]:] = w_minus_local[a[0]]
    v_plot[a[0]:] = 0

    for i in range(0, len(xi_ls)):
        if xi_ls[i] >= xi_wall and xi_ls[i] < xi_shock:
            w_plot[i] = w_ls[i]
            v_plot[i] = v_ls[i]
        if xi_ls[i] >= xi_shock:
            w_plot[i] = w_n
            v_plot[i] = 0

    plt.figure(1)
    # print(max(w_plot))
    plt.axis([0, 1, 0, max(w_plot)])
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$w(\xi)$')
    plt.plot(xi_ls, w_plot)

    plt.figure(2)
    plt.axis([0, 1, 0, 1])
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$v(\xi)$')
    plt.plot(xi_ls, v_plot)
    plt.show()


def plot_graphs():
    xi_wall = np.arange(0.35, 0.49, 0.04)
    for i in range(0, len(xi_wall)):
        plot_graph(xi_wall[i], eos='Bag')


set_params_sim('default')
fail = False
if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('usage: %s <xi_wall> <model> \n' % sys.argv[0])
        sys.exit(1)
    xi_w = float(sys.argv[1])
    if not 0. < xi_w < 1.:
        while not 0. < xi_w < 1.:
            print('Error: xi_wall must be satisfy 0 < xi_wall < 1')
            xi_w = input('Enter xi_wall ')
    state_eqn = str(sys.argv[2])
    while True:
        if state_eqn == 'Bag' or state_eqn == 'Test':
            import Bag_Toolbox as Eos
            break
        elif state_eqn == 'Eikr':
            import EIKR_Toolbox as Eos
            break
        else:
            print('Error: Unrecognised input')
            state_eqn = raw_input('Enter equation of state model: Bag, Eikr, or Test ')
    print('Would you like to edit simulation parameters? Y/N')
    print('Defaults: wn=1, N=1000')
    param_edit = raw_input()
    while True:
        if param_edit == 'Y':
            param = raw_input('Select parameter: wn/N')
            value = raw_input('Enter new value')
            set_params_sim(param, value)
            param_edit = raw_input('Do you want to change another simulation parameter? Y/N')
            if param_edit == 'N':
                break
            elif param_edit == 'Y':
                print('')
            else:
                print('Input not recognised.')
                param_edit = raw_input('Would you like to edit simulation parameters? Y/N')
        elif param_edit == 'N':
            break
        else:
            print('Input not recognised.')
            param_edit = raw_input('Would you like to edit simulation parameters? Y/N')
    print('Would you like to edit model parameters? Y/N')
    if state_eqn == 'Bag':
        print('Defaults:')
        Eos.print_params()
        param_edit = raw_input()
        while True:
            if param_edit == 'Y':
                param = raw_input('Select parameter')
                value = raw_input('Enter new value')
                set_params_sim(param, value)
                param_edit = raw_input('Do you want to change another model parameter? Y/N')
                if param_edit == 'N':
                    break
                elif param_edit == 'Y':
                    print('')
                else:
                    print('Input not recognised.')
                    param_edit = raw_input('Would you like to edit model parameters? Y/N')
            elif param_edit == 'N':
                break
            else:
                print('Input not recognised.')
                param_edit = raw_input('Would you like to edit model parameters? Y/N')
    elif state_eqn == 'Eikr':
        print('Defaults:')
        Eos.print_params()
        param_edit = raw_input()
        while True:
            if param_edit == 'Y':
                param = raw_input('Select parameter')
                value = raw_input('Enter new value')
                set_params_sim(param, value)
                param_edit = raw_input('Do you want to change another model parameter? Y/N')
                if param_edit == 'N':
                    break
                elif param_edit == 'Y':
                    print('')
                else:
                    print('Input not recognised.')
                    param_edit = raw_input('Would you like to edit model parameters? Y/N')
            elif param_edit == 'N':
                break
            else:
                print('Input not recognised.')
                param_edit = raw_input('Would you like to edit model parameters? Y/N')
    else:
        print('An unknown error has occurred, using default Bag settings (ignore if using Test)')

    plot_graph(xi_w, state_eqn)
