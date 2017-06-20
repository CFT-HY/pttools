#!/usr/bin/env python
#
# Programme for calculating and plotting scaling velocity profile around expanding Higgs-phase bubble.
# See Espinosa et al 2010
#
# Mudhahir Al-Ajmi and Mark Hindmarsh 2015-16
#
# Change to show how to use git


import sys, os
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import bubble as b

def vHybrid_approx(xi_,vWall_,v_xiWall_):
    xi0 = (v_xiWall_ + 2*vWall_)/3.
    A2 = 3*(2*xi0 -1)/(1 - xi0**2)
    dv = (xi_ - xi0)
    return xi_- 3*dv - A2*dv**2



def main():
    if not len(sys.argv) in [3,4]:
        sys.stderr.write('usage: %s <vWall> <alpha_plus> <wallType> [N_points]\n' % sys.argv[0])
        sys.exit(1)
            
    vWall = float(sys.argv[1])
    Alpha_f = float(sys.argv[2])
    wallType = sys.argv[3]

    if len(sys.argv) < 5:
        Np = 5000
    else:
        Np = int(sys.argv[4])

    if (vWall < b.cs) and not (wallType == 'Deflagration'):
        sys.stderr.write('warning: v_wall < cs, changing wallType to Deflagration\n\n')
        wallType = 'Deflagration'

    if (vWall > b.cs) and (wallType == 'Deflagration'):
        sys.stderr.write('error: Deflagration requies v_wall < cs\n')
        sys.exit(3)

    if (wallType == 'Hybrid') and (vWall > b.max_speed_deflag(Alpha_f)):
        sys.stderr.write('error: vWall too large for deflagration with alpha_plus = {}\n'.format(Alpha_f))
        sys.exit(4)

    if (wallType == 'Detonation') and (vWall < b.min_speed_deton(Alpha_f)):
        sys.stderr.write('warning: vWall too small for detonation with alpha_plus = {}\n'.format(Alpha_f))
        sys.stderr.write('warning: changing wallType to Hybrid\n\n')
        wallType = 'Hybrid'

    if (not (wallType=='Detonation') ) and (Alpha_f > 1/3.):
        sys.stderr.write('error: Hybrid or Deflagration requires alpha_plus < 1/3\n\n')
        sys.exit(5)


    dxi, xi, v_sh, nWall, ncs, xi_ahead, xi_behind = b.xvariables(Np, vWall)
   
    vfp_w, vfm_w, vfp_p, vfm_p = b.wallVariables(Alpha_f, wallType, vWall)

    print "Fluid velocity just ahead of the wall in wall frame (v+):  ", vfp_w
    print "Fluid velocity just behind of the wall in wall frame (v-):  ", vfm_w
    print "Fluid velocity just ahead of the wall in plasma frame:", vfp_p
    print "Fluid velocity just behind of the wall in plasma frame:", vfm_p

    print "Fluid velocity divisions (dxi):", dxi
    print 'v_just_behind(xi[nWall]= ', b.v_just_behind(xi[nWall+1],vfm_p,dxi)
    print "Wall speed, type:", vWall, "," ,wallType

    print "Integrating ahead because", vWall, "<", b.max_speed_deflag(Alpha_f), "(Max deflagration speed)"

#    v_f = b.velocity(vWall, Alpha_f, wallType,  Np)

    v_f,enthalp = b.fluid_shell(vWall, Alpha_f, wallType,  Np)

    if wallType == 'Hybrid':
        v_f_approx = vHybrid_approx(xi[nWall:],vWall,v_f[nWall])
        print 'Hybrid type, speed at wall ', v_f_approx[0]

    print "xi_vWall = ", xi[nWall]
    print "nWall =    ", nWall

#    print "Shock speed:", xi[nShock]

    yscale_v = max(v_f)[0]
        
#    enthalp = b.enthalpy(vWall, Alpha_f, wallType, Np, v_f)
    print "w[nWall]= ", enthalp[nWall]

    count = Np
    while (enthalp[count] == enthalp[count-1]):
        xscale_max = xi[count]*1.1
        count = count-1

    yscale_enth = max(enthalp)[0]
    xscale_min = xi[nWall]*0.5

    Ncs = np.int(np.floor(b.cs*Np))

        # Plot
        # setup latex plotting
        #plt.rc('text', usetex=True)
        #plt.rc('font', family='serif')

    plt.figure("Fluid velocity")
    plt.title(r'%s: $\xi_{\rm w} =  %g$, $\alpha_+ =  %g$, ' % (wallType, vWall, Alpha_f))
    plt.plot(xi, v_f,'b',label=r'$v(\xi)$')
    plt.plot(xi[Ncs:],v_sh[Ncs:],'r--',label=r'$v_{\rm sh}(\xi_{\rm sh})$')
    if wallType == 'Hybrid':
        plt.plot(xi[nWall:], v_f_approx,'b--',label='linear approx')
    
    plt.legend(loc='upper left')

    plt.ylabel(r'$v(\xi)$',size = 20)
    plt.xlabel(r'$\xi$',size = 20)
    plt.axis([xscale_min, xscale_max, 0.0, yscale_v*1.2])

    plt.figure("Enthalpy")
    plt.title(r'%s: $\xi_{\rm w} =  %g$, $\alpha_+ =  %g$, ' % (wallType, vWall, Alpha_f))
    plt.plot(xi[:],enthalp[:],'b',label=r'$w(\xi)$')
#       plt.plot(x[Ncs:],v_sh[Ncs:],'r--',label=r'$v_{\rm sh}(\xi_{\rm sh})$')
    plt.legend(loc='upper left')

    plt.ylabel(r'$w(\xi)$',size = 20)
    plt.xlabel(r'$\xi$',size = 20)
    plt.axis([xscale_min, xscale_max, 0.0, yscale_enth*1.2])

    plt.show()

if __name__ == '__main__':
    main()
