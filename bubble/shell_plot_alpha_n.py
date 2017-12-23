#!/usr/bin/env python
#
# Program for calculating and plotting scaling velocity profile around expanding Higgs-phase bubble
# as a function of wall speed and alpha_n ((4/3)trace anomaly over symmetric phase enthalpy density)
# See Espinosa et al 2010
#
# Mudhahir Al-Ajmi and Mark Hindmarsh 2015-17

from __future__ import absolute_import, division, print_function

import sys, os
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import bubble as b


def main():
    if not len(sys.argv) in [3, 4]:
        sys.stderr.write('usage: %s <v_wall> <alpha_n> [N_points]\n' % sys.argv[0])
        sys.exit(1)
            
    v_wall = float(sys.argv[1])
    alpha_n = float(sys.argv[2])

    if len(sys.argv) < 4:
        Np = 5000
    else:
        Np = int(sys.argv[3])

    alpha_p = b.find_alpha_plus(v_wall, alpha_n, Np)

    if np.isnan(alpha_p):
        sys.stderr.write('shell_plot_alpha_n: error: no solution for '
                         'v_wall={}, alpha_n={}\n'.format(v_wall,alpha_n) )
        sys.stderr.write('shell_plot_alpha_n: Max alpha_n for '
                         'v_wall={} is alpha_n={}\n'.format(v_wall,b.alphaNMaxDeflagration(v_wall)) )
        sys.stderr.write('shell_plot_alpha_n: Min v_wall for '
                         'alpha_n={} is v_wall={}\n'.format(alpha_n,'[function to be written]') )
        sys.exit(2)

    wall_type = b.identify_wall_type(v_wall, alpha_n)

    xi, v_sh, n_wall, n_cs = b.xvariables(Np, v_wall)

    vfp_w, vfm_w, vfp_p, vfm_p = b.fluid_speeds_at_wall(alpha_p, wall_type, v_wall)

    print('shell_plot_alpha_n: alpha_plus(alpha_n={}) = {}\n'.format(alpha_n, alpha_p))
    print("Wall speed, type: {}, {}".format(v_wall, wall_type))

    print("Fluid velocity just ahead of the wall in wall frame (v+):  ", vfp_w)
    print("Fluid velocity just behind of the wall in wall frame (v-):  ", vfm_w)
    print("Fluid velocity just ahead of the wall in plasma frame:", vfp_p)
    print("Fluid velocity just behind of the wall in plasma frame:", vfm_p)

    # Now ready to solve for fluid profile
    v_f, enthalp = b.fluid_shell(v_wall, alpha_p, wall_type,  Np)

    # with v_f can find shock position and get enthalpy ratio
    n_sh = b.find_shock_index(v_f[:],xi,v_wall,wall_type)
    r = enthalp[n_wall] / enthalp[n_wall - 1]

    print("xi_v_wall= ", xi[n_wall])
    print("xi_0     = ", b.xi_zero(v_wall,v_f[n_wall]))

    print("npts     = ", Np)
    print("n_wall   = ", n_wall)
    print("n_sh     = ", n_sh)
    print("v_shock  = ", xi[n_sh])

    print("w_+/w_n  = ", enthalp[n_wall])
    print("w_sh/w_n = ", enthalp[n_sh-1])
    print("w_+/w_sh = ", enthalp[n_wall]/enthalp[n_sh-1])
    print("w_-/w_n  = ", enthalp[n_wall-1])
    print("w_+/w_-  = ", r)

    Ubarf2 = b.Ubarf_squared(v_f, enthalp, xi, v_wall)
    wbar = b.mean_enthalpy_change(v_f, enthalp, xi, v_wall)
    K, _ = b.get_ke_frac(v_wall, alpha_n)
    print("d wbar   = ", wbar)
    print("Ubarf2   = ", Ubarf2)
    print("kappa    = ", Ubarf2/(0.75*alpha_n))
    print("K        = ", K)

    high_alpha_p_plot = 0.25  # Above which plot high-alpha approximation
    low_alpha_p_plot = 0.025  # Below which plot  low-alpha approximation

    if alpha_p > high_alpha_p_plot:
        v_f_approx = b.v_approx_high_alpha(xi[n_wall:], v_wall, v_f[n_wall])
        w_approx = b.w_approx_high_alpha(xi[n_wall:], v_wall, v_f[n_wall], enthalp[n_wall])
        print('{} type, approx speed at wall:    {}'.format(wall_type, v_f_approx[0]))
        print('{} type, approx enthalpy at wall: {}'.format(wall_type, w_approx[0]))
        print('Approx enthalpy just behind shock ', w_approx[n_sh-n_wall-1])

    if alpha_p < low_alpha_p_plot and not wall_type == 'Hybrid':
        v_f_approx = b.v_approx_low_alpha(xi, v_wall, alpha_p)
        # w_approx = b.w_approx_high_alpha(xi[n_wall:],v_wall,v_f[n_wall],enthalp[n_wall])
        print('Low alpha, approx speed at n_wall+1 ', v_f_approx[n_wall+1])
        print('Low alpha, approx speed at n_wall   ', v_f_approx[n_wall])
        # print('Hybrid type, approx enthal at wall ', w_approx[0])
        # print('Approx enthalpy just behind shock ', w_approx[n_sh-n_wall-1])

    # Plot
    # setup latex plotting
    # Needs AGG or PS or PDF backends, doesn't seem to work.
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')

    yscale_v = max(v_f)
    yscale_enth = max(enthalp)

    # xscale_min = xi[n_wall]*0.5
    # xscale_max = xi[n_sh]*1.1
    xscale_min = 0.0
    xscale_max = 1.0


    if v_wall > b.cs0:
        leg_lr = 'left'
    else:
        leg_lr = 'right'

    if alpha_p < 0.05:
        leg_ud = 'lower'
    else:
        leg_ud = 'upper'

    leg_pos_v = 'upper' + ' ' + leg_lr
    leg_pos_w = leg_ud + ' ' + leg_lr

    # plt.figure("Fluid velocity")
    plt.figure(figsize=(8, 8))
    plt.subplot(2,1,1)

    plt.title(r'%s: $\xi_{\rm w} =  %g$, $\alpha_+ =  %g$, $\alpha_n =  %g$, $r = %g$'
              % (wall_type, v_wall, alpha_p, alpha_n, r))
    plt.plot(xi, v_f,'b',label=r'$v(\xi)$')
    plt.plot(xi[n_cs:],v_sh[n_cs:],'r--',label=r'$v_{\rm sh}(\xi)$')
    if alpha_p > high_alpha_p_plot:
        plt.plot(xi[n_wall:], v_f_approx,'b--',label=r'$v$ high $\alpha$ approx')
        plt.plot(xi, xi,'k--',label=r'$v_{\rm max}$')

    if alpha_p < low_alpha_p_plot and not wall_type == 'Hybrid':
        plt.plot(xi, v_f_approx, 'b--', label=r'$v$ low $\alpha$ approx')


    plt.legend(loc=leg_pos_v)

    plt.ylabel(r'$v(\xi)$',size = 18)
    plt.xlabel(r'$\xi$',size = 18)
    plt.axis([xscale_min, xscale_max, 0.0, yscale_v*1.2])

    # plt.figure("Enthalpy")
    plt.subplot(2,1,2)
    # plt.title(r'%s: $\xi_{\rm w} =  %g$, $\alpha_+ =  %g$, $\alpha_n =  %g$, $r = %g$'
    #           % (wall_type, v_wall, alpha_p, alpha_n, r))
    plt.plot(xi[:],enthalp[:],'b',label=r'$w(\xi)$')
    plt.plot(xi[n_cs:],b.w_shock(xi[n_cs:]),'r--',label=r'$w_{\rm sh}(\xi)$')

    if alpha_p > high_alpha_p_plot:
        plt.plot(xi[n_wall:], w_approx, 'b--', label=r'$w$ high $\alpha$ approx')

    plt.legend(loc=leg_pos_w)

    plt.ylabel(r'$w(\xi)/w_{\rm n}$',size = 18)
    plt.xlabel(r'$\xi$',size = 18)
    plt.axis([xscale_min, xscale_max, 0.0, yscale_enth*1.2])

    plt.savefig('shell_plot_vw_{}_alphan_{}.pdf'.format(v_wall,alpha_n))
    plt.show()


if __name__ == '__main__':
    main()
