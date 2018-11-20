#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Programme for calculating and plotting kappa estimated from bubble fluid profile
# Comparison with Espinosa et al 2010
#

import sys
import numpy as np
import matplotlib.pyplot as plt
import bubble as b
import sigfig as sf


def main():
    if not len(sys.argv) in [5,6,7]:
        sys.stderr.write('usage: %s <vw_min> <vw_max> <log10_alpha_n_min> <log10_alpha_n_max> [n_vw] [n_a]\n' % sys.argv[0])
        sys.exit(1)
            
    vw_min = float(sys.argv[1])
    vw_max = float(sys.argv[2])
    log10_alpha_n_min = float(sys.argv[3])
    log10_alpha_n_max = float(sys.argv[4])

    n_vw = 500 # Default number of vw points
    if len(sys.argv) >= 6:
        n_vw = float(sys.argv[5])
    n_a = 3 # Default number of alpha_n points
    if len(sys.argv) == 7:
        n_a = float(sys.argv[6])

    if vw_min > 1.0 or vw_max > 1.0:
        print("error: wall speed > 1. not possible")
        sys.exit(1)

    if vw_min > vw_max:
        print("error: vw_min must be less than vw_max")
        sys.exit(1)


    vw_arr = np.linspace(vw_min,vw_max,n_vw)
#    gamma_arr = 1./np.sqrt(1 - vw_arr**2)
    alpha_n_arr = np.logspace(log10_alpha_n_min,log10_alpha_n_max,n_a)

    # Plot
#    setup latex plotting
    plt.rc('text', usetex=True)
    plt.rc('font', family='sanserif')
    plt.rc('font', size=16)
    plt.rc('lines', linewidth=2)
    # plt.rc('axes', linewidth=1.5)

    # colour_list = ['b','r','g','c','m']

    ita = np.nditer(alpha_n_arr,op_flags=['readwrite'])

    def_index = (vw_arr < b.cs0)

    for alpha_n in ita:
        alpha_n[...] = sf.round_sig(alpha_n,1)
        xcol = (np.log10(alpha_n)- log10_alpha_n_min)/(log10_alpha_n_max - log10_alpha_n_min)
        col = (xcol,0,1.-xcol)
        vJouguet = b.min_speed_deton(alpha_n)

        y_data = b.get_kappa(vw_arr,alpha_n,verbosity=2)
        x_data = vw_arr

        hyb_index = np.logical_and(vw_arr > b.cs0, vw_arr < vJouguet)
        det_index = (vw_arr > vJouguet)

        print('Plotting', alpha_n)
        plt.semilogy(x_data[def_index], y_data[def_index], color=col,
                     label=r'$\alpha_{{\rm n}} = {}$'.format(alpha_n))
        plt.semilogy(x_data[hyb_index], y_data[hyb_index], color=col, linestyle='--')
        plt.semilogy(x_data[det_index], y_data[det_index], color=col)
        # plt.plot(x_data[def_index], y_data[def_index], color=col,
        #              label=r'$\alpha_{{\rm n}} = {0:.4}$'.format(alpha_n))
        # plt.plot(x_data[hyb_index], y_data[hyb_index], color=col, linestyle='--')
        # plt.plot(x_data[det_index], y_data[det_index], color=col)


    plt.xlabel(r'$v_w$',fontsize=18)
    plt.ylabel(r'$\kappa$',fontsize=18)
    # plt.ylim([-2,np.log10(2)])
    plt.ylim([1e-2, 2.])
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles[::-1],labels[::-1],loc='upper left',fontsize=12)
    plt.grid()
    plt.tight_layout()

    plt.savefig('kappa_vw_alpha_n_new.pdf')

    plt.show()

if __name__ == '__main__':
    main()
