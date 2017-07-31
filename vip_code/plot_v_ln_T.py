#!/usr/bin/env python
# A program to generate plots of v and T/T_c against xi.

import sys
import numpy as np
from scipy import integrate as itg
from scipy import optimize as opt
import matplotlib.pyplot as plt
import Mechanics_Toolbox as Mech


def min_speed_deton(al_p, cs):
    # Minimum speed for a detonation
    return (cs/(1 + al_p))*(1 + np.sqrt(al_p*(2. + 3.*al_p)))


def identify_type(vw, al_p, tm=0):
    # Identifies wall type from provided parameters
    cs = Eos.cs(tm)
    if vw < cs:
        walltype = 'Def'
        vm = vw
        vp = Mech.v_plus(vm, al_p)
    elif vw > cs:
        if vw < min_speed_deton(al_p, cs):
            walltype = 'Hyb'
            vm = cs
            vp = Mech.v_plus(vm, al_p)
        else:
            walltype = 'Det'
            vp = vw
            print 'vp = ', vp
            vm = Mech.v_minus(vp, al_p)
    # Should consider case where vWall==cs
    print 'Using wall type ', walltype
    print 'v- = ', vm
    print 'v+ = ', vp
    print 'cs = ', cs
    return walltype, vp, vm


def v_shock(xis, cs):
    # Fluid velocity at a shock at xis.  No shock for xis < cs, so returns zero
    return np.maximum(0., (xis**2-cs**2)/(xis*(1-cs**2)))


def dy_dxi(y, xi):
    # Coupled ODEs
    v, t = y
    g = Mech.gamma(v)
    dv_dxi = (2 * v / xi) * 1 / (g*(1 - v * xi) * ((Mech.mu(xi, v) ** 2 / Eos.cs2(t)) - 1))
    dt_dxi = t*g**2 * Mech.mu(xi, v) * dv_dxi
    return [dv_dxi, dt_dxi]


def vShock(xis, tm):
    # Fluid velocity at a shock at xis.  No shock for xis < cs, so returns zero
    return np.maximum(0.,(xis**2 - Eos.cs2(tm))/(xis*(1 - Eos.cs2(tm))) )


def xvariables(Npts, v_w, tm):
    dxi = 1./Npts
    xi = np.linspace(0.01, 1., num=(Npts+1))
    v_sh = np.zeros(Npts+1)
    nWall = np.int(np.floor(v_w/dxi))
    ncs = np.int(np.floor(Eos.cs(tm)/dxi))
    v_sh[ncs:] = vShock(xi[ncs:], tm)
    return dxi, xi, v_sh, nWall, ncs


def v_just_behind(x, v, dx):
    # Fluid velocity one extra space step behind wall, arranged so that dv_dxi_deton guaranteed positive
    dv = np.sqrt(4.*dx*v*(1-v*v)*(x-v)/(1-x*x))
    return v-dv


def shell_prop(xiw, vp, vm, tm, tp, walltype, points):
    # Integrates coupled ODEs for v and T, and corrects non-physical behaviour
    dxi, xi, v_sh, nWall, ncs = xvariables(points, xiw, tm)
    vmp = Mech.mu(xiw, vm)
    vpp = Mech.mu(xiw, vp)
    y_m = v_just_behind(xiw, vmp, dxi), tm
    y_p = vpp, tp
    v_arr = np.zeros_like(xi)
    T_arr = np.zeros_like(xi)
    range_m = np.where(xi < xiw)
    range_p = np.where(xi > xiw)
    xi_m = xi[range_m]
    xi_p = xi[range_p]
    xi_m_rev = xi_m[::-1]
    if xiw > Eos.cs(tm):
        sols_m = itg.odeint(dy_dxi, y_m, xi_m_rev)
        v_arr[range_m] = sols_m[::-1, 0]
        T_arr[range_m] = sols_m[::-1, 1]
    sols_p = itg.odeint(dy_dxi, y_p, xi_p)

    v_arr[range_p] = sols_p[:, 0]
    T_arr[range_p] = sols_p[:, 1]

    print walltype, 'Correcting beyond shock'
    if not (walltype == "Det"):
        for n in range(nWall, points):
            if v_arr[n] < v_sh[n]:
                nShock = n
                break
        tps = Eos.tps_from_wps(T_arr[nShock-1], Mech.mu(xi[nShock-1], v_arr[nShock-1]), xi[nShock-1])
        T_arr[nShock:] = tps
    else:
        nShock = nWall
    v_arr[nShock:] = 0.0

    for n in range(0, len(xi)):
        if xi[n] >= Eos.cs(T_arr[n]):
            ncs = n
            break

    if not (walltype == 'Def'):
        print walltype, 'Correcting up to speed of sound'
        v_arr[0:ncs] = 0.
        T_arr[0:ncs] = T_arr[ncs]

    if walltype == 'Def':
        print walltype, 'Correcting up to wall'
        T_arr[0:nWall] = tm
    return v_arr, T_arr, xi


def main():
    # Error-check input parameters
    state_eqn = str(sys.argv[3])
    while True:
        if state_eqn == 'Bag':
            import Bag_Toolbox as Eos
            break
        elif state_eqn == 'Eikr':
            import EIKR_Toolbox as Eos
            break
        else:
            print 'Error: Unrecognised input'
            state_eqn = raw_input('Enter equation of state model: Bag or Eikr ')

    v_wall = float(sys.argv[1])
    if not 0. < v_wall < 1.:
        while not 0 < v_wall < 1:
            print 'Error: v_wall must be satisfy 0 < v_wall < 1'
            v_wall = input('Enter v_wall ')

    if len(sys.argv) == 5:
        npts = int(sys.argv[5])
    else:
        npts = 5000
        print 'npts unspecified, defaulting to npts = 5000'

    # select meaning of secondary_parameter based on model used
    if state_eqn == 'Bag':
        alpha_plus = float(sys.argv[2])
        if not 0. < alpha_plus < 1./3.: # if def
            while not 0 < alpha_plus < 1./3.:
                print 'Error: alpha_plus must be satisfy 0 < alpha_plus < 1/3'
                alpha_plus = input('Enter alpha_plus')
        t_plus = Eos.t_plus(alpha_plus)
    else:
        t_plus = float(sys.argv[2])
        if not 0. < t_plus < 1:
            while not 0. < t_plus < 1.:
                print 't_plus must satisfy 0 < t_plus < 1'
                t_plus = input('Enter t_plus ')
        alpha_plus = Eos.alphaplus(t_plus)
        print 'alpha_plus = ', alpha_plus

    # identify derived quantities, order dependent on state_eqn
    if state_eqn == 'Bag':
        print 'Using Bag procedure'
        wall_type, v_plus, v_minus = identify_type(v_wall, alpha_plus)
        t_minus = Eos.t_minus(t_plus, v_plus, v_minus)
    else:
        print 'Using EIKR procedure'
        print 'tplus = ', t_plus
        t_minus = opt.fsolve(Eos.delta_w, t_plus, v_wall)[0]
        print 'tminus=', t_minus
        sys.exit(1)
        wall_type, v_plus, v_minus = identify_type(v_wall, alpha_plus, t_minus)

    vs, Ts, xis = shell_prop(v_wall, v_plus, v_minus, t_minus, t_plus, wall_type, npts)

    # Normalise outputs
    Ts_norm = Ts/Ts[-1]
    w = Eos.w_minus(Ts)
    w_norm = w/w[-1]

    # Plot outputs
    plt.figure()
    plt.title('Velocity')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$v$')
    plt.plot(xis, vs)

    plt.figure()
    plt.title('Temperature')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$T$')
    plt.plot(xis, Ts_norm)

    plt.figure()
    plt.title('Enthalpy')
    plt.xlabel(r'$\xi$')
    plt.ylabel(r'$w/w_{\rm n}$')
    plt.plot(xis, w_norm)

    plt.show()


if __name__ == '__main__':
    if not len(sys.argv) in [4, 5]:
        sys.stderr.write('usage: %s <v_wall> <secondary_parameter> <state_eqn> [npts]\n' % sys.argv[0])
        sys.exit(1)
    state_eqn = str(sys.argv[3])
    if state_eqn == 'Bag':
        import Bag_Toolbox as Eos
    elif state_eqn == 'Eikr':
        import EIKR_Toolbox as Eos
    main()
