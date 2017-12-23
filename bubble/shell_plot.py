#!/usr/bin/env python
#
# Programme for calculating and plotting scaling velocity profile around expanding Higgs-phase bubble.
# See Espinosa et al 2010
#
# Mudhahir Al-Ajmi and Mark Hindmarsh 2015-17
#


import sys, os
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import bubble as b


def main():
    if not len(sys.argv) in [3, 4]:
        sys.stderr.write('usage: %s <v_wall> <alpha_plus> <wall_type> [N_points]\n' % sys.argv[0])
        sys.exit(1)
            
    v_wall = float(sys.argv[1])
    alpha_plus = float(sys.argv[2])
    wall_type = sys.argv[3]

    if len(sys.argv) < 5:
        Np = b.NPDEFAULT
    else:
        Np = int(sys.argv[4])

    if (v_wall < b.cs) and not (wall_type == 'Deflagration'):
        sys.stderr.write('warning: v_wall < cs, changing wall_type to Deflagration\n\n')
        wall_type = 'Deflagration'

    if (v_wall > b.cs) and (wall_type == 'Deflagration'):
        sys.stderr.write('error: Deflagration requies v_wall < cs\n')
        sys.exit(3)

    if (wall_type == 'Hybrid') and (v_wall > b.max_speed_deflag(alpha_plus)):
        sys.stderr.write('error: v_wall too large for deflagration with alpha_plus = {}\n'.format(alpha_plus))
        sys.exit(4)

    if (wall_type == 'Detonation') and (v_wall < b.min_speed_deton(alpha_plus)):
        sys.stderr.write('warning: v_wall too small for detonation with alpha_plus = {}\n'.format(alpha_plus))
        sys.stderr.write('warning: changing wall_type to Hybrid\n\n')
        wall_type = 'Hybrid'

    if (wall_type != 'Detonation') and (alpha_plus > 1/3.):
        sys.stderr.write('error: Hybrid or Deflagration requires alpha_plus < 1/3\n\n')
        sys.exit(5)

    _, v_sh, n_wall, ncs = b.xvariables(Np, v_wall)
    vfp_w, vfm_w, vfp_p, vfm_p = b.fluid_speeds_at_wall(alpha_plus, wall_type, v_wall)

    print("Fluid velocity just ahead of the wall in wall frame (v+): ", vfp_w)
    print("Fluid velocity just behind the wall in wall frame (v-):   ", vfm_w)
    print("Fluid velocity just ahead of the wall in plasma frame:    ", vfp_p)
    print("Fluid velocity just behind the wall in plasma frame:      ", vfm_p)

    v_f, enthalp, xi = b.fluid_shell(v_wall, alpha_plus, wall_type, Np)

    # print "Fluid velocity divisions (dxi):", dxi
    dxi = 1./Np
    print("v_just_behind(xi[n_wall+1],dxi) = ", b.v_just_behind(xi[n_wall+1], vfm_p, dxi))
    print("Wall speed, type:", v_wall, ",", wall_type)

    alpha_n = alpha_plus * enthalp[n_wall] / enthalp[-1]
    r = enthalp[n_wall]/enthalp[n_wall-1]

    if wall_type == 'Hybrid':
        v_f_approx = b.v_approx_hybrid(xi[n_wall:], v_wall, v_f[n_wall])
        print('Hybrid type, approx speed at wall ', v_f_approx[0])

    print("xi_v_wall = ", xi[n_wall])
    print("n_wall =    ", n_wall)

#    print "Shock speed:", xi[nShock]

    yscale_v = max(v_f)
        
#    enthalp = b.enthalpy(v_wall, alpha_plus, wall_type, Np, v_f)
    print("v[n_wall] = ", v_f[n_wall])
    print("w[n_wall] = ", enthalp[n_wall])
    print("alpha_n   = ", alpha_n)
    print("r         = ", r)

    count = Np - 1
    while enthalp[count] == enthalp[count-1]:
        xscale_max = xi[count]*1.1
        count = count-1

    yscale_enth = max(enthalp)
    xscale_min = xi[n_wall]*0.5

    Ncs = np.int(np.floor(b.cs*Np))

    # Plot
    # setup latex plotting
    #plt.rc('text', usetex=True)
    #plt.rc('font', family='serif')

    plt.figure(figsize=(8, 8))
    plt.subplot(2,1,1)

    # plt.figure("Fluid velocity")

    plt.title(r'{}: $\xi_{{\rm w}} =  {}$, $\alpha_+ =  {}$, $\alpha_{{\rm n}} =  {}$, $r =  {}$'.format(
        wall_type, v_wall, alpha_plus, alpha_n, r))
    plt.plot(xi, v_f, 'b', label=r'$v(\xi)$')
    plt.plot(xi[Ncs:], v_sh[Ncs:], 'r--', label=r'$v_{\rm sh}(\xi_{\rm sh})$')
    if wall_type == 'Hybrid':
        plt.plot(xi[n_wall:], v_f_approx, 'b--', label='linear approx')
    
    plt.legend(loc='upper left')

    plt.ylabel(r'$v(\xi)$', size=20)
    plt.xlabel(r'$\xi$', size=20)
    plt.axis([xscale_min, xscale_max, 0.0, yscale_v*1.2])

    # plt.figure("Enthalpy")
    plt.subplot(2,1,2)
    # plt.title(r'%s: $\xi_{\rm w} =  %g$, $\alpha_+ =  %g$, ' % (wall_type, v_wall, alpha_plus))
    plt.plot(xi[:], enthalp[:], 'b', label=r'$w(\xi)$')
#       plt.plot(x[Ncs:],v_sh[Ncs:],'r--',label=r'$v_{\rm sh}(\xi_{\rm sh})$')
    plt.legend(loc='upper left')

    plt.ylabel(r'$w(\xi)$', size=20)
    plt.xlabel(r'$\xi$', size=20)
    plt.axis([xscale_min, xscale_max, 0.0, yscale_enth*1.2])

    plt.show()

if __name__ == '__main__':
    main()
