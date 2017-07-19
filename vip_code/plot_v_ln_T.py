# A program to generate plots of v and ln(T/T_c) against xi.
# May in future split into two files, one containing the program itself, and a toolbox-type file.

import sys
import numpy as np
from scipy import integrate as itg
from scipy import optimize as opt
import matplotlib.pyplot as plt


def min_speed_deton(al_p, cs):
    # Minimum speed for a detonation
    print 'min_speed_deton: al_p ', al_p
    return (cs/(1 + al_p))*(1 + np.sqrt(al_p*(2. + 3.*al_p)))


# def max_speed_deflag(al_p, c):
#     # Maximum speed for a deflagration
#     vm=cs
#     return 1/(3*vPlus(vm, al_p, 'Deflagration'))


def identify_type(vw, al_p, eos, tm=0):
    # vw = wall velocity, al_p is alpha plus, cs is speed of sound (varies dependent and EoS used).
    cs = Eos.cs(tm)
    if vw < cs:
        walltype = 'Def'
        vm = vw
        vp = vplus(vm, al_p)
    elif vw > cs:
        if vw < min_speed_deton(al_p, cs):
            walltype = 'Hyb'
            vm = cs
            vp = vplus(vm, al_p)
        else:
            walltype = 'Det'
            vp = vw
            vm = vminus(vp, al_p)
    # Should consider case where vWall==cs
    print 'Using wall type ', walltype
    print 'v- = ', vm
    print 'v+ = ', vp
    print 'cs = ', cs
    return walltype, vp, vm


def vplus(vm, al_p):
    if vm <= 1./np.sqrt(3.):
        vp = (1./(1.+al_p))*(((vm/2.)+(1./(6.*vm)))-np.sqrt(((vm/2.)+(1./(6.*vm)))**2+(al_p**2)+((2./3.)*al_p)-(1./3.)))
    elif vm > 1./np.sqrt(3.):
        vp = (1./(1.+al_p))*(((vm/2.)+(1./(6.*vm)))+np.sqrt(((vm/2.)+(1./(6.*vm)))**2+(al_p**2)+((2./3.)*al_p)-(1./3.)))
    return vp


def vminus(vp, al_p):
    if vp < 1./np.sqrt(3.):
        vm = (((1+al_p)/2.)*vp+((1.-3.*al_p)/(6*vp)))-np.sqrt(((((1.+al_p)*vp)/2.)+((1.-3.*al_p)/(6.*vp)))**2-(1./3.))
    elif vp > 1./np.sqrt(3.):
        vm = (((1+al_p)/2.)*vp+((1.-3.*al_p)/(6*vp)))+np.sqrt(((((1.+al_p)*vp)/2.)+((1.-3.*al_p)/(6.*vp)))**2-(1./3.))
    # Edge case?
    return vm


def mu(vw, v):
    return (vw-v)/(1.-vw*v)


def dv_dxi_deflag(vp, x, cs):
    # differential equation: dv/dxi  for deflagrations
    if vp < v_shock(x, cs):  # Can happen if you try to integrate beyond shock
        val = 0.  # Stops solution blowing up
    else:
        val = (2./x)*vp*(1.-vp**2)*(1./(1-x*vp))*(1./(mu(x, vp)**2/cs**2 - 1))
    return val


def dv_dxi_deton(vp, x, cs):
    # differential equation: dv/dxi  for detonations and hybrids (integrate backwards from wall)
    val = (2./x)*vp*(1.-vp**2)*(1./(1-x*vp))*(1./(mu(x, vp)**2/cs**2 - 1))
    return val


def v_shock(xis, cs):
    # Fluid velocity at a shock at xis.  No shock for xis < cs, so returns zero
    return np.maximum(0., (xis**2-cs**2)/(xis*(1-cs**2)))


# def xi_stop(wallType, vw):
#     if wallType == 'Def':
#         xs = vw # wrong side, want xi shock
#     else:
#         xs = csMinus() # T dep, don't know final T
#     return xs


def gamma(v):
    return np.sqrt(1./(1-v**2))


def dy_dxi(y, xi):
    v, t = y
    g = gamma(v)
    print 'gamma', g
    print 'v', v
    print 'xi', xi
    dv_dxi = (2 * v / xi) * 1 / (g*(1 - v * xi) * ((mu(xi, v) ** 2 / Eos.cs2(t)) - 1))
    dlog_tm_dxi = g**2 * mu(xi, v) * dv_dxi
    return [dv_dxi, dlog_tm_dxi]


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
    return xi, v_sh, nWall, ncs


def shell_prop(xiw, vp, vm, tm, tp, walltype, points):
    xi, v_sh, nWall, ncs = xvariables(points, xiw, tm)
    vmp = mu(xiw, vm)
    vpp = mu(vp, xiw)
    y_m = vmp, tm
    y_p = vpp, tp
    v_arr = np.zeros_like(xi)
    psi_arr = np.zeros_like(xi)
    range_m = np.where(xi < xiw)
    range_p = np.where(xi > xiw)
    xi_m = np.transpose(xi[range_m])
    xi_p = xi[range_p]
    if xiw > Eos.cs(tm):
        sols_m = itg.odeint(dy_dxi, y_m, xi_m)
        print 'm', sols_m
        v_arr[range_m] = np.transpose(sols_m[:, 0])
        psi_arr[range_m] = sols_m[:, 1]
    sols_p = itg.odeint(dy_dxi, y_p, xi_p)
    print 'p', sols_p
    v_arr[range_p] = sols_p[:, 0]
    # if not (walltype == "Detonation"):
    #     for n in range(nWall, points):
    #         if v_arr[n] < v_sh[n]:
    #             nShock = n
    #             break
    # else:
    #     nShock = nWall

    # # Set fluid velocity to zero in front of the shock (integration isn't correct in front of shock)
    # v_arr[nShock:] = 0.0

    psi_arr[range_p] = sols_p[:, 1]

    # # Also need to set w to constant behind place where v -> 0
    # n_cs = np.max(np.where(v_arr > 0.))
    # psi_arr[n_cs + 1:] = psi_arr[n_cs]
    # # # and v to zero behind v -> 0 place
    # v_arr[n_cs + 1:] = 0
    return v_arr, psi_arr, xi


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
        print'eval alpha plus'
        alpha_plus = Eos.alphaplus(t_plus)
        print 'eval finished'

    # identify derived quantities, order dependent on state_eqn
    if state_eqn == 'Bag':
        wall_type, v_plus, v_minus = identify_type(v_wall, alpha_plus, state_eqn)
        t_minus = Eos.t_minus(t_plus, v_plus, v_minus)
    else:
        t_minus = opt.fsolve(Eos.delta_w, t_plus, v_wall)[0]
        print 'tminus=', t_minus
        wall_type, v_plus, v_minus = identify_type(v_wall, alpha_plus, state_eqn, t_minus)

    vs, psis, xis = shell_prop(v_wall, v_plus, v_minus, t_minus, t_plus, wall_type, npts)

    plt.figure()
    plt.title('Velocity')
    plt.xlabel('$/xi$')
    plt.ylabel('v')
    plt.plot(xis, vs)

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
