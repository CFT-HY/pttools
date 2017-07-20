#!/usr/bin/env python

import sys
import numpy as np
from scipy import integrate as itg
from scipy import optimize as opt
import matplotlib.pyplot as plt

import Bag_Toolbox as eos_bag
import EIKR_Toolbox as eos_eikr

def main():
    # Plot thermodynamic parameters, comparing two equations of state

    t = np.linspace(0.0,1.0,100)

    # Plot sound speed squared
    plt.figure()

    plt.plot(t,eos_bag.cs2(t),label='Bag')

    plt.plot(t,eos_eikr.cs2(t),label='EIKR')

    plt.ylim((0.,1.))

    plt.xlabel(r'$T/T_c$',fontsize=20)
    plt.ylabel(r'$c_s^2$',fontsize=20)
    plt.legend()

    # Plot EIKR borken phase vev
    plt.figure()

    plt.plot(t,eos_eikr.phi_broken(t),label='EIKR')

    plt.xlabel(r'$T/T_c$',fontsize=20)
    plt.ylabel(r'$\phi_b(T)$',fontsize=20)

    # Plot EIKR broken phase specific heat
    plt.figure()

    plt.plot(t,eos_eikr.de_dT(t),label='EIKR')

    plt.xlabel(r'$T/T_c$',fontsize=20)
    plt.ylabel(r'$de/dT$',fontsize=20)

    # Plot EIKR broken phase g_eff and h_eff
    plt.figure()

    geff = eos_eikr.e(t)/t**4/(np.pi**2/30)
    heff = eos_eikr.s(t)/t**3/(2*np.pi**2/45)

    plt.plot(t,geff,label=r'EIKR $g_{\rm eff}$')
    plt.plot(t,heff,label=r'EIKR $h_{\rm eff}$')

    plt.xlabel(r'$T/T_c$',fontsize=20)
    plt.ylabel(r'$g_{\rm eff}$, $h_{\rm eff}$',fontsize=20)
    plt.legend()

    plt.ylim((-100.,150.))

    plt.show()


if __name__ == '__main__':
    main()
