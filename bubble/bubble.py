#!/usr/bin/env python
#
# Functions for calculating fluid profile around expanding Higgs-phase bubble.
# See Espinosa et al 2010
#
# Mudhahir Al-Ajmi and Mark Hindmarsh 2015-18
# Contributions from: 
# Danny Bail 2016-18
# Jacky Lindsay and Mike Soughton 2017-18
#
# Planned changes:
# - more pythonic function names (no capitals) # Now mostly done MBH 8.12.17
# - rationalise wallVariables and find_wall_index (mostly done MBH 11.18)
# - parametric form of differential equations
# - allow general equation of state (so integrate with V, T together instead of v, w separately)

from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import scipy.integrate as int
import scipy.optimize as opt


# Should think about true cs
cs0 = 1/np.sqrt(3)  # ideal speed of sound
cs0_2 = 1./3  # ideal speed of sound squared
cs = cs0
eps = np.nextafter(0, 1)  # smallest float

# Default number of entries in xi array
NPDEFAULT = 5000
# Integration limit for parametric form of fluid equations
t_end = 50.

def cs_fun(w):
    # Speed of sound function
    # to be adapted to more realistic equations of state, e.g. with interpolation
    return cs0


def cs_w(w):
    # Speed of sound function, another label
    # to be adapted to more realistic equations of state, e.g. with interpolation
    return cs0


def cs2_w(w):
    # Speed of sound squared function
    # to be adapted to more realistic equations of state, e.g. with interpolation
    return cs0_2

# Relativity helper functions

def lorentz(xi, v):
    # Lorentz Transformation of fluid speed v between local moving frame and plasma frame.
    return (xi - v)/(1 - v*xi)


def gamma2(v):
    # Calculate square of Lorentz gamma
    return 1./(1. - v**2)


def gamma(v):
    # Calculate Lorentz gamma
    return np.sqrt(gamma2(v))


# Boundary conditions at bubble wall

def v_plus(vm, ap, wall_type):
    # Wall frame speed ahead of the wall
    X = vm + 1./(3*vm)
    if wall_type == 'Detonation':
        b = 1.
    else:
        b = -1.
    return_value = (0.5/(1+ap))*(X + b*np.sqrt(X**2 + 4.*ap**2 + (8./3.)*ap - (4./3.)))

    if isinstance(return_value, np.ndarray):
        return_value[np.where(isinstance(return_value,complex))] = np.nan
    else:
        if isinstance(return_value,complex):
            return_value = np.nan
            
    return return_value


def v_minus(vp, ap, wall_type='Detonation'):
    #    Wall frame speed behind the wall
    vp2 = vp**2
    Y = vp2 + 1./3.
    Z = (Y - ap*(1. - vp2))
    X = (4./3.)*vp2

    if wall_type=='Detonation':
        b = +1.
    else:
        b = -1.
    
    return_value = (0.5/vp)*(Z + b*np.sqrt(Z**2 - X))

    if isinstance(return_value, np.ndarray):
        return_value[np.where(isinstance(return_value,complex))] = np.nan
    else:
        if isinstance(return_value,complex):
            return_value = np.nan
            
    return return_value

    
def fluid_speeds_at_wall(alpha_p, wall_type, v_wall):
    # Sets up boundary conditions at the wall
    # (was: wallVariables)
    # Returns fluid speed vf just behind (m) and just ahead (p) of wall,
    # in wall (_w) and plasma (_p) frames.
    if v_wall <= 1:
        # print( "max_speed_deflag(alpha_p)= ", max_speed_deflag(alpha_p))
        #     if v_wall < max_speed_deflag(alpha_p) and v_wall <= cs and alpha_p <= 1/3.:
        if wall_type == 'Deflagration':
            vfm_w = v_wall                             # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = lorentz(v_wall, vfm_w)             # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(v_wall, alpha_p, wall_type) # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = lorentz(v_wall, vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
        elif wall_type == 'Hybrid':
            vfm_w = cs                                 # Fluid velocity just behind the wall in plasma frame (hybrid)
            vfm_p = lorentz(v_wall, vfm_w)             # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(cs, alpha_p, wall_type)     # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = lorentz(v_wall, vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
        elif wall_type == 'Detonation':
            vfm_w = v_minus(v_wall, alpha_p)           # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = lorentz(v_wall, vfm_w)             # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_wall                             # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = lorentz(v_wall, vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
        else:
            print("fluid_speeds_at_wall: error: wall_type wrong or unset")
            sys.exit(1)
    else:
        print("fluid_speeds_at_wall: error: v_wall > 1")

    return vfp_w, vfm_w, vfp_p, vfm_p


def v_just_behind(x, v, dx):
    # Fluid velocity one extra space step behind wall, arranged so that dv_dxi_deton 
    # Doesn't blow up and is guaranteed positive
    dv = np.sqrt(4. * dx * v * (1 - v * v) * (x - v) / (1 - x * x))
    # print( "v_just_behind: ",x,v,dx)
    return v - dv


def enthalpy_ratio(v_m, v_p):
    # ratio of enthalpies behind and ahead of a shock or transition front
    # From momentum conservation
    # w_-/w_+
    return gamma2(v_m)*v_m/(gamma2(v_p)*v_p)


# Velocity and enthalpy at shock

def v_shock(xi):
    # Fluid velocity at a shock at xi.  No shock for xi < cs, so returns zero
    v_sh = (3*xi**2 - 1)/(2*xi)

    if isinstance(v_sh, np.ndarray):
        v_sh[np.where(xi < cs0)] = 0.0
    else:
        if xi < cs0:
            v_sh = 0.0

    return v_sh


def w_shock(xi, w_n=1.):
    # Fluid enthalpy at a shock at xi.  No shock for xi < cs, so returns nan
    w_sh = w_n * (9*xi**2 - 1)/(3*(1-xi**2))

    if isinstance(w_sh, np.ndarray):
        w_sh[np.where(xi < cs0)] = np.nan
    else:
        if xi < cs0:
            w_sh = np.nan

    return w_sh


# RHS of differential equations

def dfluid_dxi(vel_en, xi, direction, cs2_fun=cs2_w):
    #  returns [dv/dxi, dw/dxi] for integrating ODEs 
    v = vel_en[0]
    w = vel_en[1]
    mu = lorentz(xi, v)
    ga2 = gamma2(v)
    cs2 = cs2_fun(w)

    if direction=='ahead' and v < v_shock(xi):  
        dv_dxi = 0.   # Stops solution blowing up when integrating beyond shock
    else:
        dv_dxi = (2./xi)*(v/(1-xi*v))/(ga2*(mu**2/cs2 - 1))
    dw_dxi = w * ga2*mu*(1/cs2 + 1)*dv_dxi
    return [dv_dxi, dw_dxi]

    
def fluid_integrate(vel_en0, xi_array, direction, cs2_fun=cs2_w):
    # integrates fluid equations in non-parametric form, returning tuple of 1D arrays
    # no error checking as yet
        
    soln = int.odeint(dfluid_dxi, vel_en0, xi_array, 
                      args=(direction, cs2_fun))
    return soln[:,0], soln[:,1], xi_array


# And now the parametric form (Jacky Lindsay and Mike Soughton MPhys project 2017-18)
def df_dtau(y, t, cs2_fun=cs2_w):
    # Returns differentials in parametric form, suitable for odeint
    v  = y[0]
    w  = y[1]
    xi = y[2]
    cs2 = cs2_fun(w)

    dxi_dt = xi * (((xi - v) ** 2) - (cs2) * ((1 - (xi * v)) ** 2))  # dxi/dt
    dv_dt  = 2 * v * (cs2) * (1 - (v ** 2)) * (1 - (xi * v))  # dv/dt
    dw_dt  = ((2 * v * w / (xi * (xi - v))) * dxi_dt + 
              ((w / (1 - v ** 2)) * (((1 - v * xi) / (xi - v)) + ((xi - v) / (1 - xi * v)))) * dv_dt)  # dw/dt

    return [dv_dt, dw_dt, dxi_dt]

    
def fluid_integrate_param(v0, w0, xi0, direction=+1., N=NPDEFAULT, cs2_fun=cs2_w):
    # Integrates parametric fluid equations from an initial condition
    t = np.linspace(0., direction*t_end, N)
    if isinstance(xi0, np.ndarray):
        soln = int.odeint(df_dtau, (v0[0], w0[0], xi0[0]), t, args=(cs2_fun, ))
    else:
        soln = int.odeint(df_dtau, (v0, w0, xi0), t, args=(cs2_fun, ))
    v = soln[:, 0]
    w  = soln[:, 1]
    xi  = soln[:, 2]

    return v, w, xi

# Useful quantities for deciding type of transition

def min_speed_deton(al_p):
    # Minimum speed for a detonation (Jouguet speed)
    # Equivalent to v_plus(cs0,al_p)
    return (cs0/(1 + al_p))*(1 + np.sqrt(al_p*(2. + 3.*al_p)))


def max_speed_deflag(al_p):
    # Maximum speed for a deflagration: speed where wall and shock are coincident
    # May be greater than 1, meaning that hybrids exist for all wall speeds.
    # alpha_plus < 1/3, but alpha_n unbounded above
    return 1/(3*v_plus(cs0, al_p, 'Deflagration'))


def identify_wall_type(vw, al_p):
    # Where both detonation and hybrid are possible, gives detonation
    # vw = wall velocity, al_p = alpha plus (= alpha_n for detonation)

    if vw < cs0:
        wall_type = 'Deflagration'
    else:
        if vw < min_speed_deton(al_p):
            wall_type = 'Hybrid'
        else:
            wall_type = 'Detonation'

    return wall_type

# Housekeeping functions for setting up integration

def make_xi_array(Npts=NPDEFAULT):
    # Creates array with xi values where v, and w are to be evaluated
    dxi = 1./Npts
    xi = np.linspace(0., 1.-dxi, Npts)
    return xi


def xvariables(Npts, v_w):
    # Makes xi array, array with v_shock, and returns index of v_wall and n_cs
#    dxi = 1./Npts
    xi = make_xi_array(Npts)
    v_sh = np.zeros(Npts)
    n_wall = find_v_index(xi, v_w)
    ncs = find_v_index(xi, cs) - 1 # Just behind
    v_sh[ncs:] = v_shock(xi[ncs:])
    return xi, v_sh, n_wall, ncs


def find_wall_index_npt(Npts, v_w):
    # Was: derived parameters
    dxi = 1./Npts
    n_wall = np.int(np.ceil(v_w/dxi))
    return n_wall


def find_v_index(xi, v):
    # Finds array index of xi just above v
    n = 0
    it = np.nditer(xi, flags=['c_index'])
    for x in it:
        if x >= v:
            n = it.index
            break
    return n


def find_shock_index(v_f, xi, v_wall, wall_type):
    # Finds array index of shock from where fluid velocity goes below v_shock
    # For detonation, returns wall position.
    n_shock = 0

    if not (wall_type == "Detonation"):
        it = np.nditer([v_f, xi], flags=['c_index'])
        for v, x in it:
            if x > v_wall:
                if v <= v_shock(x):
                    n_shock = it.index
                    break
    else:
        n_shock = find_v_index(xi, v_wall)

    return n_shock

# Main function for integrating fluid equations and deriving v, w
# for complete range 0 < xi < 1
    
def fluid_shell(v_w, al_p, wall_type="Calculate", Npts=NPDEFAULT):
    # Returns np 1D arrays with fluid velocity and enthalpy, for
    # given wall speed v_f and alpha_plus al_p
    # Enthalpy normalised to w_n = 1.0 far in front (only possible for constant cs)
    # Consider replacing Npts with xi array

    if wall_type == "Calculate":
        wall_type = identify_wall_type(v_w, al_p)
    # In case wall_type given wrongly (check this)
    if v_w > cs0 and wall_type == 'Deflagration':
        wall_type == 'Hybrid'

    xi, v_sh, n_wall, ncs = xvariables(Npts, v_w)   # initiating x-axis variables

    xi_ahead = xi[n_wall:]
    xi_behind = xi[n_wall-1:ncs:-1]  # Could be empty
    v_fluid = np.zeros_like(xi)
    w_fluid = np.zeros_like(xi)

    vfp_w, vfm_w, vfp_p, vfm_p = fluid_speeds_at_wall(al_p, wall_type, v_w)

    wp = 1.0 # Nominal value - will be rescaled later
    wm = wp/enthalpy_ratio(vfm_w, vfp_w) # enthalpy just behind wall
    w_fluid[0:n_wall] = wm # Supersonic walls will overwrite for n>ncs
    
    if wall_type == 'Hybrid':
        # Look out for divergent slope just behind wall
        dxi = v_w - xi[n_wall-1]
        if dxi < 0.:
            print("error: fluid_shell_new: dxi = v_w - xi[n_wall-1] is negative")
            sys.exit(1)
        v_behind_init = v_just_behind(xi[n_wall-1], vfm_p, dxi)
    else:
        v_behind_init = vfm_p

    v_fluid[n_wall:], w_fluid[n_wall:], _ = \
        fluid_integrate([vfp_p, wp], xi_ahead, 'ahead')
        
    v_fluid[n_wall-1:ncs:-1], w_fluid[n_wall-1:ncs:-1], _ = \
        fluid_integrate([v_behind_init, wm], xi_behind, 'behind')

    # Now reconstruct solution away from integration region

    # First in front of the shock
    if not wall_type == "Detonation":
        n_shock = find_shock_index(v_fluid, xi, v_w, wall_type)
        vfp_s = xi[n_shock]                  # Shock wall speed
        vfm_s = 1/(3*vfp_s)                  # Fluid velocity just behind the shock wall in the shock wall frame
        w_fluid[n_shock:] = w_fluid[n_shock-1]*enthalpy_ratio(vfm_s, vfp_s)
        v_fluid[n_shock:] = 0.0 # integration isn't correct in front of shock

    # Second behind wall or cs, whichever smaller
    # Hybrid/detonation integration extends back to trailing edge at n_cs+1
    # w is constant behind that
    # Deflagration: n_wall-1 is just behind wall
    # Hybrid, Detonation: n_cs+1 is just ahead of sound speed
    n_trail = min(n_wall-1,ncs+1)
    w_fluid[0:n_trail] = w_fluid[n_trail]

    # Finally, normalise enthalpy to w_n = 1 far in front
    w_fluid *= 1./w_fluid[-1]
    
    return v_fluid, w_fluid, xi

    
# Functions for alpha_n (strength parameter at nucleation temp) and 
# alpha_plus (strength parameter just in front of wall)
    
def find_alpha_n(v_wall, a_p, wall_type="Calculate", Np=NPDEFAULT):
    # Calculates alpha_N ([(3/4) difference in trace anomaly]/enthalpy) from alpha_plus (a_p)
    if wall_type == "Calculate":
        wall_type = identify_wall_type(v_wall, a_p)
    _, w, xi = fluid_shell(v_wall, a_p, wall_type, Np)
    n_wall = find_v_index(xi, v_wall)

    return a_p*w[n_wall]/w[-1]
    

def find_alpha_n_from_w_xi(w, xi, v_wall, alpha_p):
    # Calculates alpha_N ([(3/4) difference in trace anomaly]/enthalpy) from alpha_plus (ap_)
    # Assuming one has solution arrays w, xi
    n_wall = find_v_index(xi, v_wall)
    return alpha_p*w[n_wall]/w[-1]
    
    
def alpha_n_max_hybrid(v_wall, Np=NPDEFAULT):
    # Calculates maximum alpha_n for given v_wall, for Hybrid
    wall_type = identify_wall_type(v_wall, 1./3)
    if wall_type == "Deflagration":
        sys.stderr.write('alpha_n_max_hybrid: error: called with v_wall < cs\n')
        sys.stderr.write('use alpha_n_max_deflagration instead\n')
        sys.exit(6)

    # Might have been returned as "Detonation, which takes precedence over Hybrid
    wall_type = "Hybrid"
    ap = 1./3 - 1e-3
    _, w, xi = fluid_shell(v_wall, ap, wall_type, Np)
    n_wall = find_v_index(xi, v_wall)

    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    # Need n_wall+1, as w is an integral of v, and lags by 1 step
    return w[n_wall+1]*(1./3)


def alpha_n_max_deflagration(v_wall, Np=NPDEFAULT):
    # Calculates maximum alpha_n (relative trace anomaly outside bubble) for given v_wall, for deflagration
    # Works for hybrids, as they are supersonic deflagrations

    # wall_type_ = identify_wall_type(v_wall_, 1./3)
    if v_wall > cs0:
        wall_type = "Hybrid"
    else:
        wall_type = "Deflagration"

    ap = 1./3 - 1e-3
    _, w, xi = fluid_shell(v_wall, ap, wall_type, Np)
    n_wall = find_v_index(xi, v_wall)

    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    # Need n_wall+1, as w is an integral of v, and lags by 1 step
    return w[n_wall+1]*(1./3)


def alpha_plus_max_detonation(v_wall):
    # Maximum allowed value of alpha_+ for a detonation with wall speed v_wall
    # Comes from inverting v_w > v_Jouguet
    it = np.nditer([None,v_wall],[],[['writeonly','allocate'],['readonly']])
    for bb, vw in it:
        a = 3*(1-vw**2)
        if vw < cs0:
            b = 0.0
        else:
            b = (1-np.sqrt(3)*vw)**2
        bb[...] = b/a

    if isinstance(v_wall,np.ndarray):
        return it.operands[0]
    else:
        return type(v_wall)(it.operands[0])


def alpha_n_max_detonation(v_wall):
    # Maximum allowed value of alpha_n for a detonation with wall speed v_wall
    # Same as alpha_plus_max_detonation, because alpha_n = alpha_+
    return alpha_plus_max_detonation(v_wall)


def alpha_plus_min_hybrid(v_wall):
    # Minimum allowed value of alpha_+ for a hybrid with wall speed v_wall
    # Condition from coincidence of wall and shock
    b = (1-np.sqrt(3)*v_wall)**2
    c = 9*v_wall**2 - 1
    # for bb, vw in np.nditer([b,v_wall_]):
    if isinstance(b, np.ndarray):
        b[np.where(v_wall < 1./np.sqrt(3))] = 0.0
    else:
        if v_wall < cs0:
            b = 0.0
    return b/c


def alpha_n_min_hybrid(v_wall):
    # Minimum alpha_n for a hybrid is equal to maximum alpha_n for a detonation
    return alpha_n_max_detonation(v_wall)


def find_alpha_plus(v_wall, alpha_n_given, Np=NPDEFAULT):
    # Calculate alpha_plus from a given alpha_n and v_wall
    # Arguments must both be scalars

    if alpha_n_given < alpha_n_max_detonation(v_wall):
        # Must be detonation
        a_solution = alpha_n_given
    else:
        if alpha_n_given < alpha_n_max_deflagration(v_wall):
            if v_wall <= cs0:
                wall_type = "Deflagration"
            else:
                wall_type = "Hybrid"
            def func(x): return find_alpha_n(v_wall, x,wall_type, Np) - alpha_n_given
            a_initial_guess = alpha_plus_initial_guess(v_wall, alpha_n_given)
            a_solution = opt.fsolve(func, a_initial_guess, xtol=1e-4)[0]
        else:
            a_solution = np.nan

    return a_solution


def alpha_plus_initial_guess(v_wall, alpha_n_given):
    # Calculating initial guess for alpha_plus from alpha_n.
    # Linear approx between alpha_n_min and alpha_n_max

    if alpha_n_given < 0.05:
        a_guess = alpha_n_given
    else:
        alpha_plus_min = alpha_plus_min_hybrid(v_wall)
        alpha_plus_max = 1. / 3

        alpha_n_min = alpha_n_min_hybrid(v_wall)
        alpha_n_max = alpha_n_max_deflagration(v_wall)

        slope = (alpha_plus_max - alpha_plus_min) / (alpha_n_max - alpha_n_min)

        a_guess = alpha_plus_min + slope * (alpha_n_given - alpha_n_min)

    return a_guess


# Functions for calculating approximate solutions
    
def xi_zero(v_wall, v_xiWall):
    # Used in approximate solution near v(xi) = xi: defined as solution to v(xi0) = xi0
    xi0 = (v_xiWall + 2*v_wall)/3.
    return xi0


def v_approx_high_alpha(xi, v_wall, v_xiWall):
    # Approximate solution for v(xi) near v(xi) = xi
    xi0 = xi_zero(v_wall, v_xiWall)
    A2 = 3*(2*xi0 - 1)/(1 - xi0**2)
    dv = (xi - xi0)
    return xi0 - 2*dv - A2*dv**2


def v_approx_hybrid(xi, v_wall, v_xiWall):
    # Approximate solution for v(xi) near v(xi) = xi (same as v_approx_high_alpha)
    xi0 = (v_xiWall + 2*v_wall)/3.
    A2 = 3*(2*xi0 - 1)/(1 - xi0**2)
    dv = (xi - xi0)
    return xi - 3*dv - A2*dv**2


def w_approx_high_alpha(xi, v_wall, v_xi_wall, w_xi_wall):
    # Approximate solution for w(xi) near v(xi) = xi
    xi0 = xi_zero(v_wall, v_xi_wall)
    return w_xi_wall*np.exp(-12*(xi - xi0)**2/(1 - xi0**2)**2)


def v_approx_low_alpha(xi, v_wall, alpha):
    # Approximate solution for v(xi) at low alpha_+ = alpha_n

    def v_approx_fun(x, v_w):
        # Shape of linearised solution for v(xi)
        return (v_w/x)**2 * (cs0**2 - x**2)/(cs0**2 - v_w**2)

    v_app = np.zeros_like(xi)
    v_max = 3 * alpha * v_wall/abs(3*v_wall**2 - 1)
    shell = np.where(np.logical_and(xi > min(v_wall, cs0), xi < max(v_wall, cs0)))
    v_app[shell] = v_max * v_approx_fun(xi[shell], v_wall)

    return v_app


def w_approx_low_alpha(xi, v_wall, alpha):
    # Approximate solution for w(xi) at low alpha_+ = alpha_n

    v_max = 3 * alpha * v_wall/abs(3*v_wall**2 - 1)
    gaws2 = 1./(1. - 3*v_wall**2)
    w_app = np.exp(8*v_max * gaws2 * v_wall * ((v_wall/cs0) - 1))

    w_app[np.where(xi > max(v_wall, cs0))] = 1.0
    w_app[np.where(xi < min(v_wall, cs0))] = 0.0

    return w_app

# Functions for calculating quantities derived from solutions

def mean_enthalpy_change(v, w, xi, v_wall):
    # Compute mean change in enthalphy in bubble relative to outside value
    integral = np.trapz(xi**2 * (w - w[-1]), xi)
    return 3 * integral / v_wall**3


def Ubarf_squared(v, w, xi, v_wall):
    # Compute mean square (4-)velocity of fluid in bubble
    integrand = xi**2 * w * v**2 * gamma2(v)
    integral = np.trapz(integrand, xi)
    return 3 * integral / (w[-1]*v_wall**3)


def get_ke_frac(v_wall, alpha_n):
    # Determine kinetic energy fraction (of total energy)
    # Bag equation of state only so far.
    # Note e_n = (3./4) w_n (1+alpha_n) (assuming no trace anomaly in broken phase)
    ubar2, wall_type_list = get_ubarf2(v_wall, alpha_n)

    return ubar2/(0.75*(1 + alpha_n)), wall_type_list


def get_ubarf2(v_wall, alpha_n, npts=NPDEFAULT):
    # Get mean square fluid velocity. Array input for vw allowed.
    # NB was called get_ubarf2_arr
    it = np.nditer([v_wall, None])
    wall_type_list = []
    for vw, Ubarf2 in it:
        alpha_p = find_alpha_plus(vw, alpha_n)
        wall_type_list.append(identify_wall_type(vw, alpha_p))

        if not np.isnan(alpha_p):

            # Now ready to solve for fluid profile
            v_f, w, xi = fluid_shell(vw, alpha_p, wall_type_list[-1], npts)

            Ubarf2[...] = Ubarf_squared(v_f, w, xi, vw)
        else:
            Ubarf2[...] = np.nan

    Ubar2_arr = it.operands[1]

    return Ubar2_arr, wall_type_list


def get_kappa(v_wall, alpha_n, verbosity=0):
    # Calculates efficiency factor kappa from wall velocity array
    # NB was called get_kappa_arr
    it = np.nditer([v_wall, None])
    wall_type_list = []
    for vw, kappa in it:
        wall_type_list.append(identify_wall_type(vw, alpha_n))
        if wall_type_list[-1] == 'Hybrid' and vw > 0.9:
            npts = 100000
            if verbosity > 1:
                print('Choosing {} points'.format(npts))
        else:
            npts = NPDEFAULT
        alpha_p = find_alpha_plus(vw, alpha_n, npts)
        
        if not np.isnan(alpha_p):
            # Now ready to solve for fluid profile
            v_f, w, xi = fluid_shell(vw, alpha_p, wall_type_list[-1],npts)

            kappa[...] = (4./3)*Ubarf_squared(v_f, w, xi, vw)/alpha_n
        else:
            kappa[...] = np.nan
        if verbosity > 0:
            print("v_w, alpha_n, alpha_p, kappa: ", vw, alpha_n, alpha_p, kappa)

    kappa_arr = it.operands[1]

    return kappa_arr


##########
#
# Functions due for retirement
#
##########

def dv_dxi_deflag(v1, x):
    #  differential equation: dv/dxi  for deflagrations
    if v1 < v_shock(x):  # Can happen if you try to integrate beyond shock
        val = 0.         # Stops solution blowing up
    else:
        val = (2./x)*v1*(1.-v1**2)*(1./(1-x*v1))*(1./(lorentz(x, v1)**2/cs**2 - 1))
    return val


def dv_dxi_deton(v1, x):
    #  differential equation: dv/dxi  for detonations and hybrids (integrate backwards from wall)
    val = (2./x)*v1*(1.-v1**2)*(1./(1-x*v1))*(1./(lorentz(x, v1)**2/cs**2 - 1))
    return val


def ln_enthalpy_integrand(v1, x):
    # Calculating enthalpy
    lf = gamma2(v1)
    lb = lorentz(x, v1)
    return (1.+1./cs**2)*lf*lb


def velocity(v_w, al_p, wall_type, Npts):
    # Integrates dv_dxi away from wall at xi = v_w and returns fluid velocity for 0<xi<1)

    v_fluid = np.zeros([Npts, 1])  # 2D array to accommodate output of odeint
    vfp_w, vfm_w, vfp_p, vfm_p = fluid_speeds_at_wall(al_p, wall_type, v_w)
    xi, v_sh, n_wall, ncs = xvariables(Npts, v_w)   # initiating x-axis variables

    # print( "velocity: ", v_w, xi[n_wall - 1], vfm_p, wall_type

    xi_ahead = xi[n_wall:]
    xi_behind = xi[n_wall-1:ncs:-1]  # Could be empty

    if wall_type == 'Hybrid':
        # dxi = xi[n_wall] - xi[n_wall-1]
        dxi = v_w - xi[n_wall-1]
        if dxi < 0.:
            print("velocity: dxi = v_w - xi[n_wall-1] i negative")
            sys.exit(1)
        v_behind_init = v_just_behind(xi[n_wall-1], vfm_p, dxi)
    else:
        v_behind_init = vfm_p

    #   Calculating the fluid velocity
    v_fluid[n_wall:] = int.odeint(dv_dxi_deflag, vfp_p, xi_ahead)  # ,mxstep=5000000)
    v_fluid[n_wall-1:ncs:-1] = int.odeint(dv_dxi_deton, v_behind_init, xi_behind)  # ,mxstep=5000000)

    n_shock = find_shock_index(v_fluid[:,0], xi, v_w, wall_type)

    # Set fluid velocity to zero in front of the shock (integration isn't correct in front of shock)

    v_fluid[n_shock:] = 0.0

    return v_fluid[:, 0], xi


def enthalpy(v_wall, alpha_p, wall_type, Np, v_f):
    # Integrates dw_dxi away from wall at xi = v_w and returns enthalpy for 0<xi<1)
    # Integrates using known v_f(xi) and trapezium rule

    w = np.zeros_like(v_f)
    vfp_w, vfm_w, vfp_p, vfm_p = fluid_speeds_at_wall(alpha_p, wall_type, v_wall)
    xi, v_sh, n_wall, ncs = xvariables(Np, v_wall)   # initiating x-axis variables

    n_shock = find_shock_index(v_f[:], xi, v_wall, wall_type)

    # alp_s=0                               # in the plasma, no difference in EOS across shock
    if wall_type == "Detonation":
        vfp_s = xi[n_shock]                   # Shock wall speed (same as wall speed)
        vfm_s = vfp_s                        # Same as saying there is no separate shock
    else:
        vfp_s = xi[n_shock]                   # Shock wall speed
        vfm_s = 1/(3*vfp_s)                  # Fluid velocity just behing the shock wall in the shock wall frame

    # print( "n_shock:", n_shock)
    # print( "n_wall:", n_wall)
    # print( "Shock speeds (v+,v_-):", vfp_s,vfm_s)

    rw = enthalpy_ratio(vfm_w, vfp_w)
    rs = enthalpy_ratio(vfm_s, vfp_s)

    # print( "rw= ", rw, enthalpy_ratio(vfm_w,vfp_w))
    # print( "rs= ", rs, enthalpy_ratio(vfm_s,vfp_s))

    ln_en_integ = ln_enthalpy_integrand(v_f[:n_shock], xi[:n_shock])

    en_exp = int.cumtrapz(ln_en_integ, v_f[:n_shock], initial=0.0)
    w[:n_shock] = np.exp(en_exp)

    # if wall_type == "Deflagration":
    #     w[:n_wall-1,0] *= w[n_wall,0]/(w[n_wall-1,0]*rw)
    #     w[n_shock:,0] = shock_Enthalpy/(rs)
    # if wall_type == "Hybrid":
    #     # w[n_shock:,0] = shock_Enthalpy/(rs)
    #     w[:n_wall-1,0] *= w[n_wall,0]/(w[n_wall-1,0]*rw)

    if wall_type == "Detonation":
        w[n_wall:] = rw*w[n_wall-1]
    else:
        w[n_wall:n_shock] *= rw*w[n_wall-1]/w[n_wall]
        w[n_shock:] = rs*w[n_shock-1]

    # Doesn't work if n_shock = n_wall+1

    # Normalise to w_n, i.e. enthalpy at large distance
    w[:] *= 1. / w[-1]

    return w, xi


def dfluid_dxi_deton(vel_en, xi, cs2_fun=cs2_w):
    #  returns [dv/dxi, dw/dxi] for integrating ODEs (detonation)
    v = vel_en[0]
    w = vel_en[1]
    mu = lorentz(xi, v)
    ga2 = gamma2(v)
    cs2 = cs2_fun(w)

    dv_dxi = (2./xi)*(v/(1-xi*v))/(ga2*(mu**2/cs2 - 1))
    dw_dxi = w * ga2*mu*(1/cs2 + 1)*dv_dxi
    return [dv_dxi, dw_dxi]


def dfluid_dxi_deflag(vel_en, xi, cs2_fun=cs2_w):
    #  returns [dv/dxi, dw/dxi] for integrating ODEs (deflagration)
    v = vel_en[0]
    w = vel_en[1]
    mu = lorentz(xi, v)
    ga2 = gamma2(v)
    cs2 = cs2_fun(w)

    if v < v_shock(xi):  # Can happen if you try to integrate beyond shock
        dv_dxi = 0.         # Stops solution blowing up
    else:
        dv_dxi = (2./xi)*(v/(1-xi*v))/(ga2*(mu**2/cs2 - 1))
    dw_dxi = w * ga2*mu*(1/cs2 + 1)*dv_dxi
    return [dv_dxi, dw_dxi]


def fluid_shell_old(v_w, al_p, wall_type="Calculate", Npts=NPDEFAULT):
    # Returns np 1D arrays with fluid velocity and enthalpy, for
    # given wall speed v_f and alpha_plus al_p
    # Consider replacing Npts with xi array

    if wall_type == "Calculate":
        wall_type = identify_wall_type(v_w, al_p)
    # In case wall_type given wrongly (check this)
    if v_w > cs0 and wall_type == 'Deflagration':
        wall_type == 'Hybrid'

    v_fluid, xi = velocity(v_w, al_p, wall_type, Npts)
    w_fluid, _ = enthalpy(v_w, al_p, wall_type, Npts, v_fluid)

#    Now replaced by integrating equations as a pair, as in new fluid_shell 
#    Eventually will be superceded by fluid_shall_parametric
    
    return v_fluid, w_fluid, xi


