#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jacky
Modified by Danny
"""
import sys
import bubble
import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.optimize import fsolve


def cs(w, phi):
    return Eos.cs_w(w, phi)


def pressure(w, phi):
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
    v_ls = v[0:n_min]
    w_ls = w[0:n_min]
    xi_ls = xi[0:n_min]

    return v_ls, w_ls, xi_ls


def fluid_minus(v_plus_wall, w_plus_plasma, eos='Bag', p=pressure, params=[0, 0.1]):
    # Returns v_minus, w_minus from v_plus and w_plus (wall frame)
    # i.e. solves energy-momentum conservation equations across wall
    #print ("vpluswall", v_plus_wall, "w_plus_plasma", w_plus_plasma)
    #print(eos)
    if eos == 'Bag':
        e_plus = params[1]
        e_minus = params[0]
        Q = w_plus_plasma * (bubble.gamma2(v_plus_wall))
        E = (w_plus_plasma * (bubble.gamma2(v_plus_wall)) * (v_plus_wall ** 2)
             + w_plus_plasma / 4 - e_plus)
        a = 3 * Q * v_plus_wall / 4
        b = -(E + e_minus)
        c = Q * v_plus_wall / 4
        v_minus_wall = (-b - ((b ** 2) - 4 * a * c) ** (0.5)) / (2 * a)
        #print('v_minus_wall', v_minus_wall)
        w_minus_plasma = Q * v_plus_wall * (1 - v_minus_wall ** 2) / v_minus_wall
        #print('w_minus_plasma', w_minus_plasma)
        w_minus_wall = w_minus_plasma
        v_minus_wall[np.where(isinstance(v_minus_wall, complex))] = np.nan

        return v_minus_wall, w_minus_wall

    if eos != 'Bag':

        v_minus_wall_estimate, w_minus_wall_estimate = (fluid_minus(v_plus_wall,
                                                                    w_plus_plasma, 'Bag', p, params))
        vw_minus_wall_estimate = np.stack((v_minus_wall_estimate, w_minus_wall_estimate))
        e_plus = params[1]
        e_minus = params[0]
        p_plus = p(w_plus_plasma, e_plus)
        Q = w_plus_plasma * (bubble.gamma2(v_plus_wall))
        E = w_plus_plasma * (bubble.gamma2(v_plus_wall)) * (v_plus_wall ** 2) + p_plus
        v_minus_wall = np.zeros(len(v_plus_wall))
        w_minus_wall = np.zeros(len(v_plus_wall))
        for i in range(len(v_plus_wall)):
            def equations(x):
                v_minus_wall = x[0]
                w_minus_wall = x[1]
                return_array = np.zeros_like(x)
                return_array[0] = (w_minus_wall * (bubble.gamma2(v_minus_wall)) * v_minus_wall
                                   - Q[i] * v_plus_wall[i])
                return_array[1] = (w_minus_wall * (bubble.gamma2(v_minus_wall))
                                   * (v_minus_wall ** 2) + p(w_minus_wall, e_minus) - E[i])
                return return_array

            v_minus_wall[i], w_minus_wall[i] = (fsolve(equations,
                                                       vw_minus_wall_estimate[:, i]))

        return v_minus_wall, w_minus_wall


def fluid_minus_local_from_fluid_plus_plasma(v_plus_plasma, w_plus_plasma,
                                             xi_plus_plasma, eos='Bag', p=pressure, params=[0, 0.1]):
    # Finds v_minus, w_minus if wall is at range of positions xi_plus_plasma
    v_plus_wall = bubble.lorentz(xi_plus_plasma, v_plus_plasma)  # this is an array?
    v_minus_local, w_minus_local = fluid_minus(v_plus_wall, w_plus_plasma, eos, p, params)

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


def fluid_at_wall(xi_shock, w_n=1, eos='Bag', p=pressure, params=[0, 0.1], N=1000, c_s=cs):
    # Integrate back from shock, return inferred wall speed and fluid variables
    # uses:  fluid_from_xi_sh
    #            fluid_minus_local_from_fluid_plus_plasma
    #            exit_speed_wall(xi_real)
    v_ls, w_ls, xi_ls = fluid_from_xi_sh(xi_shock, w_n, N, c_s)

    v_minus_local, w_minus_local = fluid_minus_local_from_fluid_plus_plasma(v_ls, w_ls,
                                                                            xi_ls, eos, p, params)
    v_exit = exit_speed_wall(xi_ls)

    rootguess_xi = root_estimate(xi_ls, v_minus_local, v_exit)
    print('rootguess_xi', rootguess_xi)
    v_minus_local_function = interp1d(xi_ls, v_minus_local)
    v_exit_function = interp1d(xi_ls, v_exit)

    def v_remainder(xi_ls, v_minus_local_function, v_exit_function):
        return v_minus_local_function(xi_ls) - v_exit_function(xi_ls)

    xi_try = fsolve(v_remainder, rootguess_xi, args=(v_minus_local_function, v_exit_function))

    return xi_try


def wall_speed_zero(xi_sh, xi_wall, w_n=1, eos='Bag', p=pressure, params=[0, 0.1],
                    N=1000, c_s=cs):
    # Returns difference between wall speed xi_try computed from
    # xi_shock and eos and target wall speed. Suitable for use in root-finder.
    xi_try = fluid_at_wall(xi_sh, w_n, eos, p, params, N, c_s)
    return xi_try - xi_wall


def root_find_xi_sh(xi_wall, w_n=1., eos='Bag', p=pressure, params=[0, 0.1], N=1000, c_s=cs):
    # invokes root finder on wall_speed_zero to get xi_sh
    x0 = c_s(w_n)
    xi_shock = opt.newton(wall_speed_zero, x0, args=(xi_wall, w_n, eos, p, params, N, c_s))
    return xi_shock


def plot_graph(xi_wall, eos, w_n=1., params=[0, 0.1], N=1000):
    xi_shock = root_find_xi_sh(xi_wall, w_n, eos, params=params, N=N)
    v_ls, w_ls, xi_ls = fluid_from_xi_sh(xi_shock, w_n, N)
    v_minus_local, w_minus_local = fluid_minus_local_from_fluid_plus_plasma(v_ls, w_ls,
                                                                            xi_ls, eos, params)

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
    #print(max(w_plot))
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
        plot_graph(xi_wall[i])


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write('usage: %s <xi_wall> <model> \n' % sys.argv[0])
        sys.exit(1)
    xi_w = float(sys.argv[1])
    if not 0. < xi_w < 1.:
        while not 0. < xi_w < 1.:
            print 'Error: xi_wall must be satisfy 0 < xi_wall < 1'
            v_wall = input('Enter xi_wall ')
    state_eqn = str(sys.argv[2])
    while True:
        if state_eqn == 'Bag':
            import Bag_Toolbox as Eos
            break
        elif state_eqn == 'Eikr':
            import EIKR_Toolbox as Eos
            break
        else:
            print 'Error: Unrecognised input'
            state_eqn = raw_input('Enter equation of state model: Bag, Eikr, or Test ')
    print('Would you like to edit other parameters? Y/N')
    if state_eqn == 'Bag' or state_eqn == 'Test':
        print('Defaults: w_n=1, e-=0, e+=0.1, N=1000')
        param_edit = raw_input()
        while True:
            if param_edit == 'Y':
                print('Leave values blank to use defaults')
                in1 = raw_input('w_n = ')
                if in1 == '':
                    wn = 1.0
                else:
                    wn = float(in1)
                in2 = raw_input('e- = ')
                if in2 == '':
                    eminus = 0.
                else:
                    eminus = float(in2)
                in3 = raw_input('e+ = ')
                if in3 == '':
                    eplus = 0.1
                else:
                    eplus = float(in3)
                in4 = raw_input('N = ')
                if in4 == '':
                    N = 1000
                else:
                    N = int(in4)
                secondary_params = [eminus, eplus]
                break
            elif param_edit == 'N':
                wn = 1.
                secondary_params = [0., 0.1]
                N = 1000
                break
            else:
                print('Input not recognised.')
                param_edit = raw_input('Would you like to edit parameters? Y/N')
    elif state_eqn == 'Eikr':
        print('Defaults: w_n=1, phi-=None, phi+=0, N=1000')
        param_edit = raw_input()
        while True:
            if param_edit == 'Y':
                print('Leave values blank to use defaults')
                in1 = raw_input('w_n = ')
                if in1 == '':
                    wn = 1.0
                else:
                    wn = float(in1)
                in2 = raw_input('phi- = ')
                if in2 == '':
                    phiminus = None
                else:
                    phiminus = float(in2)
                in3 = raw_input('phi+ = ')
                if in3 == '':
                    phiplus = 0.
                else:
                    phiplus = float(in3)
                in4 = raw_input('N = ')
                if in4 == '':
                    N = 1000
                else:
                    N = int(in4)
                secondary_params = [phiminus, phiplus]
                break
            elif param_edit == 'N':
                wn = 1.
                secondary_params = [None, 0.1]
                N = 1000
                break
            else:
                print('Input not recognised.')
                param_edit = raw_input('Would you like to edit parameters? Y/N')
    else:
        print('An unknown error has occurred, using default Bag settings')
        fail = True
    if fail == True:
        plot_graph(xi_w, state_eqn)
    else:
        plot_graph(xi_w, state_eqn, wn, secondary_params, N)
