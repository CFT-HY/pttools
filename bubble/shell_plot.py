#!/usr/bin/env python
#
# Programme for calculating and plotting scaling velocity profile around expanding Higgs-phase bubble.
# See Espinosa et al 2010
#
# Mudhahir Al-Ajmi and Mark Hindmarsh 2015-17
#


import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import bubble as b


def main():
    if not len(sys.argv) in [3, 4]:
        sys.stderr.write('usage: %s <v_wall> <alpha_n> [N_points]\n' % sys.argv[0])
        sys.exit(1)
            
    v_wall = float(sys.argv[1])
    alpha_n = float(sys.argv[2])

    if len(sys.argv) < 4:
        Np = b.NPDEFAULT
    else:
        Np = int(sys.argv[3])

    wall_type = b.identify_wall_type(v_wall, alpha_n)

    v, w, xi = b.fluid_shell(v_wall, alpha_n, Np)
    xi_even = np.linspace(1/Np,1-1/Np,Np)
    v_sh = b.v_shock(xi_even)

    n_wall = b.find_v_index(xi, v_wall)
    n_cs = np.int(np.floor(b.cs0*Np))

    r = w[n_wall]/w[n_wall-1]
    alpha_plus = alpha_n*w[-1]/w[n_wall]


    ubarf2 = b.Ubarf_squared(v, w, xi, v_wall)
    # Kinetic energy fraction of total
    ke_frac = ubarf2/(0.75*(1 + alpha_n))
    # Efficiency of turning Higgs potential into kinetic energy
    kappa = ubarf2/(0.75*alpha_n)
    # and efficiency of turning Higgs potential into thermal energy
    dw = 0.75 * b.mean_enthalpy_change(v, w, xi, v_wall)/(0.75 * alpha_n * w[-1])
    


    if wall_type == 'Hybrid':
        v_approx = b.v_approx_hybrid(xi[n_wall:], v_wall, v[n_wall])

    # Plot
    # Comment out for latex plotting if possible
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')
    mpl.rcParams.update({'font.size': 14})
    yscale_v = max(v)*1.2
    xscale_max = min(xi[-2]*1.1,1.0)
    yscale_enth = max(w)*1.2
    xscale_min = xi[n_wall]*0.5

    plt.figure(figsize=(8, 8))

# First velocity
    plt.subplot(2,1,1)

    plt.title(r'{}: $\xi_{{\rm w}} =  {}$, $\alpha_{{\rm n}} =  {}$, $\alpha_+ =  {:5.3f}$, $r =  {:5.3f}$, $\xi_{{\rm sh}} =  {}$'.format(
        wall_type, v_wall, alpha_n, alpha_plus, r, xi[-2]),size=14)
    plt.plot(xi, v, 'b', label=r'$v(\xi)$')
    if not wall_type == 'Detonation':
        plt.plot(xi_even[n_cs:], v_sh[n_cs:], 'r--', label=r'$v_{\rm sh}(\xi_{\rm sh})$')
    if wall_type == 'Hybrid':
        plt.plot(xi[n_wall:], v_approx, 'b--', label='linear approx')
    
    plt.legend(loc='upper left')

    plt.ylabel(r'$v(\xi)$', size=16)
    plt.xlabel(r'$\xi$', size=16)
    plt.axis([xscale_min, xscale_max, 0.0, yscale_v])

# Then enthalpy
    plt.subplot(2,1,2)

    plt.title(r'$K =  {:5.3f}$, $\kappa =  {:5.3f}$, $\omega  =  {:5.3f}$'.format(ke_frac, kappa, dw),size=14)
    plt.plot(xi[:], w[:], 'b', label=r'$w(\xi)$')

    plt.legend(loc='upper left')
    plt.ylabel(r'$w(\xi)$', size=16)
    plt.xlabel(r'$\xi$', size=16)
    plt.axis([xscale_min, xscale_max, 0.0, yscale_enth])

    plt.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()
