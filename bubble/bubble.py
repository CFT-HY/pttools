r"""
Functions for calculating fluid profile around expanding Higgs-phase bubble.

Finds, analyses and plots self-similar functions $v$ (radial fluid velocity) 
and $w$ (fluid enthalpy) as functions of the scaled variable $\xi = r/t$. 
Main inputs are wall speed $v_w$ and global transition strength parameter 
$\alpha_n$.

See Espinosa et al 2010, Hindmarsh & Hijazi 2019.

Authors: Mark Hindmarsh 2015-20, with Mudhahir Al-Ajmi, and contributions from: 
 Danny Bail (Sussex MPhys RP projects 2016-18); Jacky Lindsay and Mike Soughton (MPhys project 2017-18)

Changes planned at 06.20:
- allow general equation of state (so integrate with V, T together instead of v, w separately)
   Idea to introduce eos as a class. Need a new interface which uses eos variables rather than alpha.
- Include bubble nucleation calculations of beta (from V(T,phi))
- Now comments are docstrings, think about sphinx
- Complete checks for physical (v_wall, alpha_n)

Changes 06.20:
- Small improvements to docstrints.
- Start introducing checks for physical (v_wall, alpha_n): check_wall_speed, check_physical_parameters


"""

# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

import sys
import numpy as np
import scipy.integrate as spi
import scipy.optimize as opt
import matplotlib as mpl
import matplotlib.pyplot as plt

# Get decent-size plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

font_size = 18
mpl.rcParams.update({'font.size': font_size})
mpl.rcParams.update({'lines.linewidth': 1.5})
mpl.rcParams.update({'axes.linewidth': 2.0})
mpl.rcParams.update({'axes.labelsize': font_size})
mpl.rcParams.update({'xtick.labelsize': font_size})
mpl.rcParams.update({'ytick.labelsize': font_size})
# but make legend smaller
mpl.rcParams.update({'legend.fontsize': 14})

# smallest float
eps = np.nextafter(0, 1)  

# Default and maximum number of entries in xi array
N_XI_DEFAULT = 5000
N_XI_MAX = 1000000
# How accurate is alpha_plus(alpha_n)
find_alpha_plus_tol=1e-6
# Integration limit for parametric form of fluid equations
T_END_DEFAULT = 50.
dxi_small = 1./N_XI_DEFAULT

# Some functions useful for the bag equation of state.

cs0 = 1/np.sqrt(3)  # ideal speed of sound
cs0_2 = 1./3  # ideal speed of sound squared


#def cs_w(w):
#    # Speed of sound function, another label
#    # to be adapted to more realistic equations of state, e.g. with interpolation
#    return cs0
#
#
#def cs2_w(w):
#    # Speed of sound squared function
#    # to be adapted to more realistic equations of state, e.g. with interpolation
#    return cs0_2
#
#
def cs2_bag(w):
    """
    Speed of sound squared in Bag model, equal to 1/3 independent of enthalpy $w$
    """
    # Should return same type/shape as w
    return cs0_2


def theta_bag(w, phase, alpha_n):
    """
    Trace anomaly $\theta = (e - 3p)/4$ in Bag model.
    """
    return alpha_n * (0.75 * w[-1]) * (1 - phase) 


def p(w, phase, theta_s, theta_b=0.):
    """
     Pressure as a function of enthalpy, assuming bag model.
     phase: phase indicator (see below).
     theta = (e - 3p)/4 (trace anomaly or "vacuum energy")
     _s = symmetric phase, ahead of bubble (phase = 0)
     _b = broken phase, behind bubble (phase = 1)
     enthalpy, theta and phase can be arrays (same shape)
    """
    theta = theta_b*phase + theta_s*(1.0 - phase)
    return 0.25*w - theta
    

def e(w, phase, theta_s, theta_b=0.):
    """
     Energy density as a function of enthalpy, assuming bag model.
     theta = (e - 3p)/4 ("vacuum energy")
     _s = symmetric phase, ahead of bubble (phase = 0)
     _b = broken phase, behind bubble (phase = 1)
     enthalpy and phase can be arrays (same shape)
    """
    return w - p(w,phase,theta_s,theta_b)


def w(e, phase, theta_s, theta_b=0):
    """
     Enthalpy as a function of energy density, assuming bag model.
     theta = (e - 3p)/4 ("vacuum energy")
     _s = symmetric phase, ahead of bubble (phase = 0)
     _b = broken phase, behind bubble (phase = 1)
     enthalpy and phase can be arrays (same shape)
    """
#     Actually, theta is often known only from alpha_n and w, so should
#     think about an fsolve?
    theta = theta_b*phase + theta_s*(1.0 - phase)
    return (4/3)*(e - theta)


def phase(xi,v_w):
    """
     Returns array indicating phase of system.  
     in symmetric phase (xi>v_w), phase = 0 
     in broken phase (xi<v_w), phase = 1
    """
    ph = np.zeros_like(xi)
    ph[np.where(xi < v_w)] = 1.0
    return ph


# Relativity helper functions

def lorentz(xi, v):
    """
     Lorentz transformation of fluid speed v between moving frame and plasma frame.
    """
    return (xi - v)/(1 - v*xi)


def gamma2(v):
    """
     Square of Lorentz gamma
    """
    return 1./(1. - v**2)


def gamma(v):
    """
     Lorentz gamma
    """
    return np.sqrt(gamma2(v))


# Boundary conditions at bubble wall 

def v_plus(vm, ap, wall_type):
    """
     Wall frame fluid speed v_plus ahead of the wall, as a function of 
     vm = v_minus - fluid speed v_plus behind the wall
     ap = alpha_plus - strength parameter at wall
     wall_type - Detonation, Deflagration, Hybrid
    """
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
    """
     Wall frame fluid speed v_minus behind the wall, as a function of 
     vp = v_plus - fluid speed v_plus behind the wall
     ap = alpha_plus - strength parameter at wall
     wall_type - Detonation, Deflagration, Hybrid
    """
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

    
def fluid_speeds_at_wall(v_wall, alpha_p, wall_type):
    """
     Solves fluid speed boundary conditions at the wall, returning 
         vfp_w, vfm_w, vfp_p, vfm_p
     Fluid speed vf? just behind (?=m) and just ahead (?=p) of wall,
     in wall (_w) and plasma (_p) frames.
    """
    if v_wall <= 1:
        # print( "max_speed_deflag(alpha_p)= ", max_speed_deflag(alpha_p))
        #     if v_wall < max_speed_deflag(alpha_p) and v_wall <= cs and alpha_p <= 1/3.:
        if wall_type == 'Deflagration':
            vfm_w = v_wall                             # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = lorentz(v_wall, vfm_w)             # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(v_wall, alpha_p, wall_type) # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = lorentz(v_wall, vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
        elif wall_type == 'Hybrid':
            vfm_w = cs0                                # Fluid velocity just behind the wall in plasma frame (hybrid)
            vfm_p = lorentz(v_wall, vfm_w)             # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(cs0, alpha_p, wall_type)     # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = lorentz(v_wall, vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
        elif wall_type == 'Detonation':
            vfm_w = v_minus(v_wall, alpha_p)           # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = lorentz(v_wall, vfm_w)             # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_wall                             # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = lorentz(v_wall, vfp_w)             # Fluid velocity just ahead of the wall in plasma frame
        else:
            sys.stderr.write("fluid_speeds_at_wall: error: wall_type wrong or unset")
            sys.exit(1)
    else:
        sys.stderr.write("fluid_speeds_at_wall: error: v_wall > 1")

    return vfp_w, vfm_w, vfp_p, vfm_p


def enthalpy_ratio(v_m, v_p):
    """
     Ratio of enthalpies behind (w_- ) and ahead (w_+) of a shock or 
     transition front, w_-/w_+. Uses conservation of momentum in moving frame.
    """
    return gamma2(v_m)*v_m/(gamma2(v_p)*v_p)


# Fluid differential equations 
# Now in parametric form (Jacky Lindsay and Mike Soughton MPhys project 2017-18)
# RHS is Eq (33) in Espinosa et al (plus dw/dt not written there)
def df_dtau(y, t, cs2_fun=cs2_bag):
    """
     Differentials of fluid variables (v, w, xi) in parametric form, suitable for odeint
    """
    v  = y[0]
    w  = y[1]
    xi = y[2]
    cs2 = cs2_fun(w)
    xiXv = xi*v
    xi_v = xi - v
    v2  = v*v
    
    dxi_dt = xi * ((xi_v)**2 - cs2 * (1 - xiXv)**2)  # dxi/dt
    dv_dt  = 2 * v * cs2 * (1 - v2) * (1 - xiXv)  # dv/dt
    dw_dt  = (w/(1 - v2)) * (xi_v/(1 - xiXv)) * (1/cs2 + 1) * dv_dt

    return [dv_dt, dw_dt, dxi_dt]

    
def fluid_integrate_param(v0, w0, xi0, t_end=T_END_DEFAULT, n_xi=N_XI_DEFAULT, cs2_fun=cs2_bag):
    """
     Integrates parametric fluid equations in df_dtau from an initial condition.
     Positive t_end integrates along curves from (v,w) = (0,cs0) to (1,1).
     Negative t_end integrates towards (0,cs0).
     Returns: v, w, xi, t
    """
    t = np.linspace(0., t_end, n_xi)
    if isinstance(xi0, np.ndarray):
        soln = spi.odeint(df_dtau, (v0[0], w0[0], xi0[0]), t, args=(cs2_fun, ))
    else:
        soln = spi.odeint(df_dtau, (v0, w0, xi0), t, args=(cs2_fun, ))
    v   = soln[:, 0]
    w   = soln[:, 1]
    xi  = soln[:, 2]

    return v, w, xi, t


# Useful quantities for deciding type of transition

def min_speed_deton(alpha):
    """
     Minimum speed for a detonation (Jouguet speed). 
     Equivalent to v_plus(cs0,alpha). 
     Note that alpha_plus = alpha_n for detonation.
    """
    return (cs0/(1 + alpha))*(1 + np.sqrt(alpha*(2. + 3.*alpha)))


def max_speed_deflag(alpha_p):
    """
     Maximum speed for a deflagration: speed where wall and shock are coincident.
     May be greater than 1, meaning that hybrids exist for all wall speeds above cs.
     alpha_plus < 1/3, but alpha_n unbounded above.
    """
    return 1/(3*v_plus(cs0, alpha_p, 'Deflagration'))


def identify_wall_type(v_wall, alpha_n):
    """
     Determines wall type from wall speed and global strength parameter. 
     wall_type = [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    # v_wall = wall velocity, alpha_n = relative trace anomaly at nucleation temp outside shell
    wall_type = 'Error' # Default
    if alpha_n < alpha_n_max_detonation(v_wall):
        # Must be detonation
        wall_type = 'Detonation'
    else:
        if alpha_n < alpha_n_max_deflagration(v_wall):
            if v_wall <= cs0:
                wall_type = "Deflagration"
            else:
                wall_type = "Hybrid"

    return wall_type


def identify_wall_type_alpha_plus(v_wall, alpha_p):
    """
     Determines wall type from wall speed and at-wall strength parameter. 
     wall_type = [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    if v_wall <= cs0:
        wall_type = 'Deflagration'
    else:
        if alpha_p < alpha_plus_max_detonation(v_wall):
            wall_type = 'Detonation'
            if alpha_p > alpha_plus_min_hybrid(v_wall) and alpha_p < 1/3.:
                sys.stderr.write('identify_wall_type_alpha_plus: warning:\n')
                sys.stderr.write('      Hybrid and Detonation both possible for v_wall = {}, alpha_plus = {}\n'.format(v_wall,alpha_p))
                sys.stderr.write('      Choosing detonation.\n')
        else:
            wall_type = 'Hybrid'


    if alpha_p > (1/3.) and not wall_type == 'Detonation':
        sys.stderr.write('identify_wall_type_alpha_plus: error:\n')
        sys.stderr.write('      no solution for v_wall = {}, alpha_plus = {}\n'.format(v_wall,alpha_p))
        wall_type = 'Error'

    return wall_type


# Useful functions for finding properties of solution 

def find_v_index(xi, v_target):
    """
     The first array index of xi where value is just above v_target
    """
    n = 0
    it = np.nditer(xi, flags=['c_index'])
    for x in it:
        if x >= v_target:
            n = it.index
            break
    return n


def v_shock(xi):
    """
     Fluid velocity at a shock at xi.  No shocks exist for xi < cs, so returns zero.
    """
    # Maybe should return a nan?
    v_sh = (3*xi**2 - 1)/(2*xi)

    if isinstance(v_sh, np.ndarray):
        v_sh[np.where(xi < cs0)] = 0.0
    else:
        if xi < cs0:
            v_sh = 0.0

    return v_sh


def w_shock(xi, w_n=1.):
    """
     Fluid enthalpy at a shock at xi.  No shocks exist for xi < cs, so returns nan.
    """
    w_sh = w_n * (9*xi**2 - 1)/(3*(1-xi**2))

    if isinstance(w_sh, np.ndarray):
        w_sh[np.where(xi < cs0)] = np.nan
    else:
        if xi < cs0:
            w_sh = np.nan

    return w_sh


def find_shock_index(v_f, xi, v_wall, wall_type):
    """
     Array index of shock from first point where fluid velocity v_f goes below v_shock
     For detonation, returns wall position.
    """
    check_wall_speed(v_wall)
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


def shock_zoom_last_element(v, w, xi):
    """
     Replaces last element of (v,w,xi) arrays by better estimate of 
     shock position and values of v, w there.
    """
    v_sh = v_shock(xi)
    # First check if last two elements straddle shock
    if v[-1] < v_sh[-1] and v[-2] > v_sh[-2] and xi[-1] > xi[-2]:
        dxi = xi[-1] - xi[-2]
        dv = v[-1] - v[-2]
        dv_sh = v_sh[-1] - v_sh[-2]
        dw_sh = w[-1] - w[-2]
        dxi_sh = dxi * (v[-2] - v_sh[-2])/(dv_sh - dv)
        # now replace final element
        xi[-1] = xi[-2] + dxi_sh
        v[-1]  = v[-2] + (dv_sh/dxi)*dxi_sh
        w[-1]  = w[-2] + (dw_sh/dxi)*dxi_sh
    # If not, do nothing
    return v, w, xi
    

# Main function for integrating fluid equations and deriving v, w
# for complete range 0 < xi < 1

def fluid_shell(v_wall, alpha_n, n_xi=N_XI_DEFAULT):
    """
     Finds fluid shell (v, w, xi) from a given v_wall, alpha_n, which must be scalars.  
     Option to change xi resolution n_xi
    """
    check_physical_params([v_wall,alpha_n])
    wall_type = identify_wall_type(v_wall, alpha_n)
    if wall_type == 'Error':
        sys.stderr.write('fluid_shell: giving up because of identify_wall_type error')
        return np.nan, np.nan, np.nan
    else:
        al_p = find_alpha_plus(v_wall, alpha_n, n_xi)
        if not np.isnan(al_p):
            return fluid_shell_alpha_plus(v_wall, al_p, wall_type, n_xi)
        else:
            return np.nan, np.nan, np.nan
        

def fluid_shell_alpha_plus(v_wall, alpha_plus, wall_type='Calculate', n_xi=N_XI_DEFAULT, w_n=1, cs2_fun=cs2_bag):
    """
     Finds fluid shell (v, w, xi) from a given v_wall, alpha_plus (at-wall strength parameter).  
     Where v=0 (behind and ahead of shell) uses only two points.
     v_wall and alpha_plus must be scalars, and are converted from 1-element arrays if needed.
     Options: 
         wall_type (string) - specify wall type if more than one permitted.
         n_xi (int) - increase resolution
         w_n - specify enthalpy outside fluid shell
         cs2_fun - sound speed squared as a function of enthalpy, default 
    """
    check_wall_speed(v_wall)
    dxi = 1./n_xi
#    dxi = 10*eps

    if isinstance(alpha_plus, np.ndarray):
        al_p = np.asscalar(alpha_plus)
    else:
        al_p = alpha_plus
    if isinstance(v_wall, np.ndarray):
        v_w = np.asscalar(v_wall)
    else:
        v_w = v_wall

    if wall_type == "Calculate":
        wall_type = identify_wall_type_alpha_plus(v_w, al_p)

    if wall_type == 'Error':
        sys.stderr.write('fluid_shell_alpha_plus: giving up because of identify_wall_type error')
        return np.nan, np.nan, np.nan

    # Solve boundary conditions at wall
    vfp_w, vfm_w, vfp_p, vfm_p = fluid_speeds_at_wall(v_w, al_p, wall_type)
    wp = 1.0 # Nominal value - will be rescaled later
    wm = wp/enthalpy_ratio(vfm_w, vfp_w) # enthalpy just behind wall

    # Set up parts outside shell where v=0. Need 2 points only.
    xif = np.linspace(v_wall + dxi,1.0,2)
    vf = np.zeros_like(xif)
    wf = np.ones_like(xif)*wp 

    xib = np.linspace(min(cs2_fun(w_n)**0.5,v_w)-dxi,0.0,2)
    vb = np.zeros_like(xib)
    wb = np.ones_like(xib)*wm 

    # Integrate forward and find shock.
    if not wall_type == 'Detonation':
    # First go
        v,w,xi,t = fluid_integrate_param(vfp_p, wp, v_w, -T_END_DEFAULT, N_XI_DEFAULT, cs2_fun)
        v, w, xi, t = trim_fluid_wall_to_shock(v, w, xi, t, wall_type)
    # Now refine so that there are ~N points between wall and shock.  A bit excessive for thin
    # shocks perhaps, but better safe than sorry. Then improve final point with shock_zoom...
        t_end_refine = t[-1]
        v,w,xi,t = fluid_integrate_param(vfp_p, wp, v_w, t_end_refine, n_xi, cs2_fun)
        v, w, xi, t = trim_fluid_wall_to_shock(v, w, xi, t, wall_type)
        v, w, xi = shock_zoom_last_element(v, w, xi)
    # Now complete to xi = 1
        vf = np.concatenate((v,vf))
    # enthalpy
        vfp_s = xi[-1]        # Fluid velocity just ahead of shock in shock frame = shock speed
        vfm_s = 1/(3*vfp_s)   # Fluid velocity just behind shock in shock frame
        wf = np.ones_like(xif)*w[-1]*enthalpy_ratio(vfm_s, vfp_s)
        wf = np.concatenate((w,wf))
    # xi
        xif[0] = xi[-1]
        xif = np.concatenate((xi,xif))
        
    # Integrate backward to sound speed.
    if not wall_type == 'Deflagration':
    # First go
        v,w,xi,t = fluid_integrate_param(vfm_p, wm, v_w, -T_END_DEFAULT, N_XI_DEFAULT, cs2_fun)
        v, w, xi, t = trim_fluid_wall_to_cs(v, w, xi, t, v_wall, wall_type)
#    # Now refine so that there are ~N points between wall and point closest to cs
#    # For walls just faster than sound, will give very (too?) fine a resolution.
#        t_end_refine = t[-1]
#        v,w,xi,t = fluid_integrate_param(vfm_p, wm, v_w, t_end_refine, n_xi, cs2_fun)
#        v, w, xi, t = trim_fluid_wall_to_cs(v, w, xi, t, v_wall, wall_type)

    # Now complete to xi = 0
        vb = np.concatenate((v,vb))
        wb = np.ones_like(xib)*w[-1]
        wb = np.concatenate((w,wb))
        # Can afford to bring this point all the way to cs2.
        xib[0] = cs2_fun(w[-1])**0.5
        xib = np.concatenate((xi,xib))

    # Now put halves together in right order
# Need to fix this according to python version
#    v  = np.concatenate((np.flip(vb,0),vf))
#    w  = np.concatenate((np.flip(wb,0),wf))
#    w  = w*(w_n/w[-1])
#    xi = np.concatenate((np.flip(xib,0),xif))
    v  = np.concatenate((np.flipud(vb),vf))
    w  = np.concatenate((np.flipud(wb),wf))
    w  = w*(w_n/w[-1])
    xi = np.concatenate((np.flipud(xib),xif))

    return v, w, xi


def trim_fluid_wall_to_cs(v, w, xi, t, v_wall, wall_type, dxi_lim=dxi_small, cs2_fun=cs2_bag):
    """
     Picks out fluid variable arrays (v, w, xi, t) which are definitely behind 
     the wall for deflagration and hybrid.
     Also removes negative fluid speeds and xi <= sound_speed, which might be left by
     an inaccurate integration.
     If the wall is within about 1e-16 of cs, rounding errors are flagged.
    """
    check_wall_speed(v_wall)
    n_start = 0 
        

    n_stop_index = -2
    n_stop = 0
    if not wall_type == 'Deflagration':
        it = np.nditer([v, w, xi], flags=['c_index'])
        for vv, ww, x in it:
            if vv <= 0 or x**2 <= cs2_fun(ww):
                n_stop_index = it.index
                break
    
    if n_stop_index == 0:
        sys.stderr.write('trim_fluid_wall_to_cs: warning: integation gave v < 0 or xi <= cs\n')
        sys.stderr.write('     wall_type: {}, v_wall: {}, xi[0] = {}, v[] = {}\n'.format(
                wall_type, v_wall, xi[0], v[0]))
        sys.stderr.write('     Fluid profile has only one element between vw and cs. Fix implemented by adding one extra point.\n')
        n_stop = 1
    else:
        n_stop = n_stop_index

    if (xi[0] == v_wall) and not (wall_type == "Detonation"):
        n_start = 1
        n_stop += 1
        

    return v[n_start:n_stop], w[n_start:n_stop], xi[n_start:n_stop], t[n_start:n_stop]



def trim_fluid_wall_to_shock(v, w, xi, t, wall_type):
    """
     Trims fluid variable arrays (v, w, xi) so last element is just ahead of shock
    """
    n_shock_index = -2
    n_shock = 0
    if not wall_type == 'Detonation':
        it = np.nditer([v, xi], flags=['c_index'])
        for vv, x in it:
            if vv <= v_shock(x):
                n_shock_index = it.index
                break
    
    if n_shock_index == 0:
        sys.stderr.write('trim_fluid_wall_to_shock: warning: v[0] < v_shock(xi[0]\n')
        sys.stderr.write('     wall_type: {}, xi[0] = {}, v[0] = {}, v_sh(xi[0]) = {}\n'.format(
                wall_type, xi[0], v[0], v_shock(xi[0])))
        sys.stderr.write('     Shock profile has only one element. Fix implemented by adding one extra point.\n')
        n_shock = 1
    else:
        n_shock = n_shock_index

#    print("n_shock",n_shock,v[n_shock],v_shock(xi[n_shock]))
    return v[:n_shock+1], w[:n_shock+1], xi[:n_shock+1], t[:n_shock+1]


# Functions for alpha_n (strength parameter at nucleation temp) and 
# alpha_p(lus) (strength parameter just in front of wall)
    
def find_alpha_n(v_wall, alpha_p, wall_type="Calculate", n_xi=N_XI_DEFAULT):
    """
     Calculates alpha_n from alpha_plus, for given v_wall.
     v_wall can be scalar or iterable.
     alpha_p[lus] must be scalar.
     alpha = ([(3/4) difference in trace anomaly]/enthalpy).
     alpha_n is global strength parameter, alpha_plus the at-wall strength parameter.
     
    """
    check_wall_speed(v_wall)
    if wall_type == "Calculate":
        wall_type = identify_wall_type_alpha_plus(v_wall, alpha_p)
    _, w, xi = fluid_shell_alpha_plus(v_wall, alpha_p, wall_type, n_xi)
    n_wall = find_v_index(xi, v_wall)
    return alpha_p*w[n_wall]/w[-1]


def find_alpha_plus(v_wall, alpha_n_given, n_xi=N_XI_DEFAULT):
    """
     Calculate alpha_plus from a given alpha_n and v_wall.
     v_wall can be scalar or iterable.
     alpha_n_given must be scalar.
     alpha = ([(3/4) difference in trace anomaly]/enthalpy) 
     alpha_n is global strength parameter, alpha_plus the at-wall strength parameter.
    """

    it = np.nditer([None,v_wall],[],[['writeonly','allocate'],['readonly']])
    for ap, vw in it:
        if alpha_n_given < alpha_n_max_detonation(vw):
        # Must be detonation
            wall_type = 'Detonation'
            ap[...] = alpha_n_given
        else:
            if alpha_n_given < alpha_n_max_deflagration(vw):
                if vw <= cs0:
                    wall_type = "Deflagration"
                else:
                    wall_type = "Hybrid"
                def func(x): 
                    return find_alpha_n(vw, x, wall_type, n_xi) - alpha_n_given
                a_initial_guess = alpha_plus_initial_guess(vw, alpha_n_given)
                al_p = opt.fsolve(func, a_initial_guess, xtol=find_alpha_plus_tol)[0]
                ap[...] = al_p
            else:
                ap[...] = np.nan

    if isinstance(v_wall,np.ndarray):
        return it.operands[0]
    else:    
        return type(v_wall)(it.operands[0])


def alpha_plus_initial_guess(v_wall, alpha_n_given):
    """
     Initial guess for root-finding of alpha_plus from alpha_n.
     Linear approx between alpha_n_min and alpha_n_max. 
    """
#     Doesn't do obvious checks like Detonation - needs improving?

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


def find_alpha_n_from_w_xi(w, xi, v_wall, alpha_p):
    """
     Calculates alpha_N ([(3/4) difference in trace anomaly]/enthalpy) from alpha_p[lus] 
     Assumes one has solution arrays w, xi
    """
    n_wall = find_v_index(xi, v_wall)
    return alpha_p*w[n_wall]/w[-1]
    
    
def alpha_n_max_hybrid(v_wall, n_xi=N_XI_DEFAULT):
    """
     Calculates maximum alpha_n for given v_wall, assuming Hybrid fluid shell
    """
    wall_type = identify_wall_type_alpha_plus(v_wall, 1./3)
    if wall_type == "Deflagration":
        sys.stderr.write('alpha_n_max_hybrid: error: called with v_wall < cs\n')
        sys.stderr.write('     use alpha_n_max_deflagration instead\n')
        sys.exit(6)

    # Might have been returned as "Detonation, which takes precedence over Hybrid
    wall_type = "Hybrid"
    ap = 1./3 - 1e-8
    _, w, xi = fluid_shell_alpha_plus(v_wall, ap, wall_type, n_xi)
    n_wall = find_v_index(xi, v_wall)

    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    return w[n_wall]*(1./3)


def alpha_n_max(v_wall, Np=N_XI_DEFAULT):
    """
    alpha_n_max(v_wall, Np=N_XI_DEFAULT)
    
     Calculates maximum alpha_n (relative trace anomaly outside bubble) 
     for given v_wall, which is max alpha_n for (supersonic) deflagration.
    """    
    return alpha_n_max_deflagration(v_wall, Np)


def alpha_n_max_deflagration(v_wall, Np=N_XI_DEFAULT):
    """
     Calculates maximum alpha_n (relative trace anomaly outside bubble) 
     for given v_wall, for deflagration.
     Works also for hybrids, as they are supersonic deflagrations
    """
    check_wall_speed(v_wall)
    # wall_type_ = identify_wall_type(v_wall_, 1./3)
    it = np.nditer([None,v_wall],[],[['writeonly','allocate'],['readonly']])
    for ww, vw in it:
        if vw > cs0:
            wall_type = "Hybrid"
        else:
            wall_type = "Deflagration"

        ap = 1./3 - 1.0e-10 # Warning - this is not safe.  Causes warnings for v low vw
        _, w, xi = fluid_shell_alpha_plus(vw, ap, wall_type, Np)
        n_wall = find_v_index(xi, vw)
        ww[...] = w[n_wall+1]*(1./3)

    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalised to 1 at large xi
    # Need n_wall+1, as w is an integral of v, and lags by 1 step
    if isinstance(v_wall,np.ndarray):
        return it.operands[0]
    else:
        return type(v_wall)(it.operands[0])


def alpha_plus_max_detonation(v_wall):
    """
     Maximum allowed value of alpha_plus for a detonation with wall speed v_wall. 
     Comes from inverting v_w > v_Jouguet
    """
    if v_wall >= 1.0:
        sys.exit('alpha_n_max_detonation: error: unphysical parameter(s)\n\
                 v_wall = {}, require v_wall < 1'.format(v_wall))

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
    """
     Maximum allowed value of alpha_n for a detonation with wall speed v_wall. 
     Same as alpha_plus_max_detonation, because alpha_n = alpha_plus for detonation.
    """
    return alpha_plus_max_detonation(v_wall)


def alpha_plus_min_hybrid(v_wall):
    """
     Minimum allowed value of alpha_plus for a hybrid with wall speed v_wall. 
     Condition from coincidence of wall and shock
    """
    check_wall_speed(v_wall)
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
    """
     Minimum alpha_n for a hybrid. Equal to maximum alpha_n for a detonation.
    """
    check_wall_speed(v_wall)
    return alpha_n_max_detonation(v_wall)


# Functions for calculating approximate solutions
    
def xi_zero(v_wall, v_xi_wall):
    """
     Used in approximate solution near v(xi) = xi: defined as solution to v(xi0) = xi0
    """
    check_wall_speed(v_wall)
    xi0 = (v_xi_wall + 2*v_wall)/3.
    return xi0


def v_approx_high_alpha(xi, v_wall, v_xi_wall):
    """
     Approximate solution for fluid velocity v(xi) near v(xi) = xi. 
    """
    check_wall_speed(v_wall)
    xi0 = xi_zero(v_wall, v_xi_wall)
    A2 = 3*(2*xi0 - 1)/(1 - xi0**2)
    dv = (xi - xi0)
    return xi0 - 2*dv - A2*dv**2


def v_approx_hybrid(xi, v_wall, v_xi_wall):
    """
     Approximate solution for fluid velocity v(xi) near v(xi) = xi (same as v_approx_high_alpha).
    """
    check_wall_speed(v_wall)
    xi0 = xi_zero(v_wall, v_xi_wall)
    A2 = 3*(2*xi0 - 1)/(1 - xi0**2)
    dv = (xi - xi0)
    return xi - 2*dv - A2*dv**2


def w_approx_high_alpha(xi, v_wall, v_xi_wall, w_xi_wall):
    """
     Approximate solution for ehthalpy w(xi) near v(xi) = xi. 
    """
    check_wall_speed(v_wall)
    xi0 = xi_zero(v_wall, v_xi_wall)
    return w_xi_wall*np.exp(-12*(xi - xi0)**2/(1 - xi0**2)**2)


def v_approx_low_alpha(xi, v_wall, alpha):
    """
     Approximate solution for fluid velocity v(xi) at low alpha_plus = alpha_n. 
    """

    def v_approx_fun(x, v_w):
        # Shape of linearised solution for v(xi)
        return (v_w/x)**2 * (cs0**2 - x**2)/(cs0**2 - v_w**2)

    v_app = np.zeros_like(xi)
    v_max = 3 * alpha * v_wall/abs(3*v_wall**2 - 1)
    shell = np.where(np.logical_and(xi > min(v_wall, cs0), xi < max(v_wall, cs0)))
    v_app[shell] = v_max * v_approx_fun(xi[shell], v_wall)

    return v_app


def w_approx_low_alpha(xi, v_wall, alpha):
    """
     Approximate solution for enthalpy w(xi) at low alpha_plus = alpha_n. 
     (Not complete for xi < min(v_wall, cs0)). 
    """
    
    v_max = 3 * alpha * v_wall/abs(3*v_wall**2 - 1)
    gaws2 = 1./(1. - 3*v_wall**2)
    w_app = np.exp(8*v_max * gaws2 * v_wall * ((v_wall/cs0) - 1))

    w_app[np.where(xi > max(v_wall, cs0))] = 1.0
    w_app[np.where(xi < min(v_wall, cs0))] = 0.0

    return w_app

# Functions for calculating quantities derived from solutions

def split_integrate(fun_, v, w, xi, v_wall):
    """
    # Split an integration of a function fun_ of arrays v w xi 
    # according to whether xi is inside or outside the wall (expecting discontinuity there). 
    """
    check_wall_speed(v_wall)
    inside  = np.where(xi < v_wall) 
    outside = np.where(xi > v_wall) 
    int1 = 0.
    int2 = 0.
    if v[inside].size >= 3:
        int1 = part_integrate(fun_, v, w, xi, inside)
    if v[outside].size >= 3:
        int2 = part_integrate(fun_, v, w, xi, outside)
    return int1, int2


def part_integrate(fun_, v, w, xi, where_in):
    """
     Integrate a function fun_ of arrays v w xi over index selection where_in.
    """
    xi_in = xi[where_in]
    v_in = v[where_in]
    w_in = w[where_in]
    integrand = fun_(v_in, w_in, xi_in)
    return np.trapz(integrand, xi_in)


def de_from_w(w, xi, v_wall, alpha_n):
    """
    Calculates energy density difference de = e - e[-1] from enthalpy, assuming
    bag equation of state.
    Can get alpha_n = find_alpha_n_from_w_xi(w,xi,v_wall,alpha_p) 
    """
    check_physical_params([v_wall,alpha_n])
    e_from_w = e(w, phase(xi,v_wall), 0.75*w[-1]*alpha_n)
              
    return e_from_w - e_from_w[-1]


def de_from_w_new(v, w, xi, v_wall, alpha_n):
    """
    For exploring new methods of calculating energy density difference 
    from velocity and enthalpy, assuming bag equation of state.
    """
    check_physical_params([v_wall,alpha_n])
    e_from_w = e(w, phase(xi,v_wall), 0.75*w[-1]*alpha_n)
    
    de = e_from_w - e_from_w[-1]

    # Try adjusting by a factor - currently doesn't do anything    
    de *= 1.0
    
    return de


def mean_energy_change(v, w, xi, v_wall, alpha_n):
    """
     Bubble-averaged change in energy density in bubble relative to outside value. 
    """
#    def ene_diff(v,w,xi):
#        return de_from_w(w, xi, v_wall, alpha_n)
#    int1, int2 = split_integrate(ene_diff, v, w, xi**3, v_wall)
#    integral = int1 + int2
    check_physical_params([v_wall,alpha_n])
    integral = np.trapz(de_from_w(w, xi, v_wall, alpha_n), xi**3)
    return integral / v_wall**3


def mean_enthalpy_change(v, w, xi, v_wall):
    """
     Mean change in enthalphy in bubble relative to outside value. 
    """
#    def en_diff(v, dw, xi):
#        return dw
#    int1, int2 = split_integrate(en_diff, v, w - w[-1], xi**3, v_wall)
#    integral = int1 + int2
    check_wall_speed(v_wall)
    integral = np.trapz((w - w[-1]), xi**3)
    return integral / v_wall**3


def mean_kinetic_energy(v, w, xi, v_wall):
    """
     Kinetic energy of fluid in bubble, averaged over bubble volume, 
     from fluid shell functions.
    """
    check_wall_speed(v_wall)
    integral = np.trapz(w * v**2 * gamma2(v), xi**3)
    
    return integral / (v_wall**3)


def ubarf_squared(v, w, xi, v_wall):
    """
     Enthalpy-weighted mean square space components of 4-velocity of fluid in bubble,  
     from fluid shell functions.
    """
    check_wall_speed(v_wall)
#    def fun(v,w,xi):
#        return w * v**2 * gamma2(v)
#    int1, int2 = split_integrate(fun, v, w, xi**3, v_wall)
#    integral = int1 + int2
#    integral = np.trapz(w * v**2 * gamma2(v), xi**3)
    
    return  mean_kinetic_energy(v, w, xi, v_wall) / w[-1]


def get_ke_frac(v_wall, alpha_n, n_xi=N_XI_DEFAULT):
    """
     Determine kinetic energy fraction (of total energy). 
     Bag equation of state only so far, as it takes 
     e_n = (3./4) w_n (1+alpha_n). This assumes zero trace anomaly in broken phase. 
    """
    ubar2 = get_ubarf2(v_wall, alpha_n, n_xi)

    return ubar2/(0.75*(1 + alpha_n))

    
def get_ke_de_frac(v_wall, alpha_n, n_xi=N_XI_DEFAULT, verbosity=0):
    """
     Kinetic energy fraction and fractional change in energy 
     from wall velocity array. Sum should be 0. Assumes bag model.
    """
    it = np.nditer([v_wall, None, None])
    for vw, ke, de in it:
        wall_type = identify_wall_type(vw, alpha_n)

        if not wall_type=='Error':
            # Now ready to solve for fluid profile
            v, w, xi = fluid_shell(vw, alpha_n, n_xi)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            ke[...] = ubarf_squared(v, w, xi, vw)/(0.75 * (1 + alpha_n))
            de[...] = mean_energy_change(v, w, xi, vw, alpha_n)/(0.75 * w[-1]*(1 + alpha_n))
        else:
            ke[...] = np.nan
            de[...] = np.nan
        if verbosity > 0:
            sys.stderr.write("{:8.6f} {:8.6f} {} {}".format(vw, alpha_n, ke, de),flush=True)

    if isinstance(v_wall,np.ndarray):
        ke_out = it.operands[1]
        de_out = it.operands[2]
    else:
        ke_out = type(v_wall)(it.operands[1])
        de_out = type(v_wall)(it.operands[2])

    return ke_out, de_out


def get_ubarf2(v_wall, alpha_n, n_xi=N_XI_DEFAULT, verbosity=0):
    """
     Get mean square fluid velocity from v_wall and alpha_n.
     v_wall can be scalar or iterable. 
     alpha_n must be scalar.
    """
    it = np.nditer([v_wall, None])
    for vw, Ubarf2 in it:
        wall_type = identify_wall_type(vw, alpha_n)
        if not wall_type=='Error':
            # Now ready to solve for fluid profile
            v, w, xi = fluid_shell(vw, alpha_n, n_xi)
            Ubarf2[...] = ubarf_squared(v, w, xi, vw)
        else:
            Ubarf2[...] = np.nan
        if verbosity > 0:
            sys.stderr.write("{:8.6f} {:8.6f} {} ".format(vw, alpha_n, Ubarf2),flush=True)


    # Ubarf2 is stored in it.operands[1]
    if isinstance(v_wall,np.ndarray):
        ubarf2_out = it.operands[1]
    else:
        ubarf2_out = type(v_wall)(it.operands[1])

    return ubarf2_out


def get_kappa(v_wall, alpha_n, n_xi=N_XI_DEFAULT, verbosity=0):
    """
     Efficiency factor kappa from v_wall and alpha_n. v_wall can be array.
    """
    # NB was called get_kappa_arr
    it = np.nditer([v_wall, None])
    for vw, kappa in it:
        wall_type = identify_wall_type(vw, alpha_n)

        if not wall_type=='Error':
            # Now ready to solve for fluid profile
            v, w, xi = fluid_shell(vw, alpha_n, n_xi)

            kappa[...] = ubarf_squared(v, w, xi, vw)/(0.75*alpha_n)
        else:
            kappa[...] = np.nan
        if verbosity > 0:
            sys.stderr.write("{:8.6f} {:8.6f} {} ".format(vw, alpha_n, kappa),flush=True)

    if isinstance(v_wall,np.ndarray):
        kappa_out = it.operands[1]
    else:
        kappa_out = type(v_wall)(it.operands[1])

    return kappa_out


def get_kappa_de(v_wall, alpha_n, n_xi=N_XI_DEFAULT, verbosity=0):
    """
     Calculates efficiency factor kappa and fractional change in energy 
     from v_wall and alpha_n. v_wall can be an array. Sum should be 0 (bag model).
    """
    it = np.nditer([v_wall, None, None])
    for vw, kappa, de in it:
        wall_type = identify_wall_type(vw, alpha_n)

        if not wall_type=='Error':
            # Now ready to solve for fluid profile
            v, w, xi = fluid_shell(vw, alpha_n, n_xi)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            kappa[...] = (4/3)*ubarf_squared(v, w, xi, vw)
            de[...]    = mean_energy_change(v, w, xi, vw, alpha_n)
        else:
            kappa[...] = np.nan
            de[...] = np.nan
        if verbosity > 0:
            sys.stderr.write("{:8.6f} {:8.6f} {} {}".format(vw, alpha_n, kappa, de),flush=True)

    if isinstance(v_wall,np.ndarray):
        kappa_out = it.operands[1]
        de_out = it.operands[2]
    else:
        kappa_out = type(v_wall)(it.operands[1])
        de_out = type(v_wall)(it.operands[2])

    return kappa_out, de_out


def get_kappa_dq(v_wall, alpha_n, n_xi=N_XI_DEFAULT, verbosity=0):
    """
     Calculates efficiency factor kappa and fractional change in thermal energy 
     from v_wall and alpha_n. v_wall can be an array. Sum should be 1. 
     Thermal energy is defined as q = (3/4)*enthalpy. 
    """
    it = np.nditer([v_wall, None, None])
    for vw, kappa, dq in it:
        wall_type = identify_wall_type(vw, alpha_n)

        if not wall_type=='Error':
            # Now ready to solve for fluid profile
            v, w, xi = fluid_shell(vw, alpha_n, n_xi)
            # Esp+ epsilon is alpha_n * 0.75*w_n
            kappa[...] = ubarf_squared(v, w, xi, vw)/(0.75 * alpha_n)
            dq[...]    = 0.75 * mean_enthalpy_change(v, w, xi, vw)/(0.75 * alpha_n * w[-1])
        else:
            kappa[...] = np.nan
            dq[...] = np.nan
        if verbosity > 0:
            sys.stderr.write("{:8.6f} {:8.6f} {} {}".format(vw, alpha_n, kappa, dq),flush=True)

    if isinstance(v_wall,np.ndarray):
        kappa_out = it.operands[1]
        dq_out = it.operands[2]
    else:
        kappa_out = type(v_wall)(it.operands[1])
        dq_out = type(v_wall)(it.operands[2])

    return kappa_out, dq_out



def check_wall_speed(v_wall):
    """
    Checks that v_wall values are all physical (0 < v_wall <1)
    """
    if isinstance(v_wall,float):
        if v_wall >= 1.0 or v_wall <= 0.0:
            sys.exit('check_wall_speed: error: unphysical parameter(s)\n\
                     v_wall = {}, require 0 < v_wall < 1'.format(v_wall))
    elif isinstance(v_wall,np.ndarray):
        if np.logical_or(np.any(v_wall >= 1.0), np.any(v_wall <= 0.0)):
            sys.exit('check_wall_speed: error: unphysical parameter(s)\n\
                     at least one value outside 0 < v_wall < 1')
    elif isinstance(v_wall,list):
        for vw in v_wall:
            if vw >= 1.0 or vw <= 0.0:
                sys.exit('check_wall_speed: error: unphysical parameter(s)\n\
                         at least one value outside 0 < v_wall < 1')
    else:
        sys.exit('check_wall_speed: error: v_wall must be float, list or array.\n ')             
                
    return None


def check_physical_params(params):
    """
    Checks that v_wall = params[0], alpha_n = params[1] values are physical, i.e.
         0 < v_wall <1
         alpha_n < alpha_n_max(v_wall)
    """
    v_wall = params[0]
    alpha_n = params[1]
    check_wall_speed(v_wall)
    
    if alpha_n > alpha_n_max(v_wall):
            sys.exit('check_alpha_n: error: unphysical parameter(s)\n\
                     v_wall, alpha_n = {}, {}\n\
                     require alpha_n < {}\n'.format(v_wall,alpha_n,alpha_n_max(v_wall)) )
    return None

    
def plot_fluid_shell(v_wall, alpha_n, save_string=None, Np=N_XI_DEFAULT):
    """
     Calls ``fluid_shell`` and plots resulting v, w against xi, returning figure handle.
     Also plots:
     - shock curves (where v and w should form shock)
     - low alpha approximation if alpha_plus < 0.025
     - high alpha approximation if alpha_plus > 0.2
     Annotates titles with:
     - Wall type, v_wall, alpha_n
     - alpha_plus (alpha just in front of wall)
     - r (ratio of enthalpies either side of wall)
     - xi_sh (shock speed)
     - w_0/w_n (ration of internal to external enthalpies)
     - ubar_f (mean square U = gamma(v) v)
     - K kinetic energy fraction
     - kappa (Espinosa et al efficiency factor)
     - omega (thermal energy relative to scalar potential energy, as measured by trace anomaly)
     Last two should sum to 1.
    """
    check_physical_params([v_wall, alpha_n])
    
    high_v_plot = 0.8 # Above which plot v ~ xi approximation
    low_v_plot = 0.025  # Below which plot  low v approximation

    wall_type = identify_wall_type(v_wall, alpha_n)

    if wall_type=='Error':
        sys.stderr.write('shell_plot: error: no solution for v_wall = {}, alpha_n = {}\n'.format(v_wall,alpha_n))
        sys.exit(1)
        
    v, w, xi = fluid_shell(v_wall, alpha_n, Np)
    
    vmax = max(v)
    
    xi_even = np.linspace(1/Np,1-1/Np,Np)
    v_sh = v_shock(xi_even)
    w_sh = w_shock(xi_even)

    n_wall = find_v_index(xi, v_wall)
    n_cs = np.int(np.floor(cs0*Np))
    n_sh = xi.size-2

    r = w[n_wall]/w[n_wall-1]
    alpha_plus = alpha_n*w[-1]/w[n_wall]


    ubarf2 = ubarf_squared(v, w, xi, v_wall)
    # Kinetic energy fraction of total (Bag equation of state)
    ke_frac = ubarf2/(0.75*(1 + alpha_n))
    # Efficiency of turning Higgs potential into kinetic energy
    kappa = ubarf2/(0.75*alpha_n)
    # and efficiency of turning Higgs potential into thermal energy
    dw = 0.75 * mean_enthalpy_change(v, w, xi, v_wall)/(0.75 * alpha_n * w[-1])
    
    if vmax > high_v_plot*v_wall:
        v_approx = v_approx_high_alpha(xi[n_wall:n_sh], v_wall, v[n_wall])
        w_approx = w_approx_high_alpha(xi[n_wall:n_sh], v_wall, v[n_wall], w[n_wall])

    if vmax < low_v_plot and not wall_type == 'Hybrid':
        v_approx = v_approx_low_alpha(xi, v_wall, alpha_plus)

    # Plot
    yscale_v = max(v)*1.2
    xscale_max = min(xi[-2]*1.1,1.0)
    yscale_enth_max = max(w)*1.2
    yscale_enth_min = min(w)/1.2
    xscale_min = xi[n_wall]*0.5

    f = plt.figure(figsize=(7, 8))

# First velocity
    plt.subplot(2,1,1)

    plt.title(r'$\xi_{{\rm w}} =  {}$, $\alpha_{{\rm n}} =  {:.3}$, $\alpha_+ =  {:5.3f}$, $r =  {:.3f}$, $\xi_{{\rm sh}} =  {:5.3f}$'.format(
        v_wall, alpha_n, alpha_plus, r, xi[-2]),size=16)
    plt.plot(xi, v, 'b', label=r'$v(\xi)$')

    if not wall_type == 'Detonation':
        plt.plot(xi_even[n_cs:], v_sh[n_cs:], 'k--', label=r'$v_{\rm sh}(\xi_{\rm sh})$')
        if vmax > high_v_plot*v_wall:
            plt.plot(xi[n_wall:n_sh], v_approx,'b--',label=r'$v$ ($v < \xi$ approx)')
            plt.plot(xi, xi,'k--',label=r'$v = \xi$')

    if not wall_type == 'Deflagration':
        v_minus_max = lorentz(xi_even, cs0)
        plt.plot(xi_even[n_cs:], v_minus_max[n_cs:], 'k-.', label=r'$\mu(\xi,c_{\rm s})$')

    if vmax < low_v_plot and not wall_type == 'Hybrid':
        plt.plot(xi, v_approx, 'b--', label=r'$v$ low $\alpha$ approx')
    
    plt.legend(loc='upper left')

    plt.ylabel(r'$v(\xi)$')
    plt.xlabel(r'$\xi$')
    plt.axis([xscale_min, xscale_max, 0.0, yscale_v])
    plt.grid()
    
# Then enthalpy
    plt.subplot(2,1,2)

    plt.title(r'$w_0/w_n = {:4.2}$, $\bar{{U}}_f = {:.3f}$, $K = {:5.3g}$, $\kappa = {:5.3f}$, $\omega = {:5.3f}$'.format(
              w[0]/w[-1],ubarf2**0.5,ke_frac, kappa, dw),size=16)
    plt.plot(xi, np.ones_like(xi)*w[-1], '--', color='0.5')
    plt.plot(xi, w, 'b', label=r'$w(\xi)$')
    
    if not wall_type == 'Detonation':
        plt.plot(xi_even[n_cs:], w_sh[n_cs:],'k--',label=r'$w_{\rm sh}(\xi_{\rm sh})$')

        if vmax > high_v_plot*v_wall:
            plt.plot(xi[n_wall:n_sh], w_approx[:], 'b--', label=r'$w$ ($v < \xi$ approx)')

    else:
        wmax_det = (xi_even/cs0)*gamma2(xi_even)/gamma2(cs0)
        plt.plot(xi_even[n_cs:], wmax_det[n_cs:],'k-.',label=r'$w_{\rm max}$')
#    if alpha_plus < low_alpha_p_plot and not wall_type == 'Hybrid':
#        plt.plot(xi, w_approx, 'b--', label=r'$w$ low $\alpha$ approx')

    plt.legend(loc='upper left')
    plt.ylabel(r'$w(\xi)$', size=16)
    plt.xlabel(r'$\xi$', size=16)
    plt.axis([xscale_min, xscale_max, yscale_enth_min, yscale_enth_max])
    plt.grid()

    plt.tight_layout()

    if save_string is not None:
        plt.savefig('shell_plot_vw_{}_alphan_{:.3}{}'.format(v_wall,alpha_n,save_string))
    plt.show()
    
    return f


def plot_fluid_shells(v_wall_list, alpha_n_list, multi=False, save_string=None, Np=N_XI_DEFAULT):
    """
     Calls ``fluid_shell`` and plots resulting v, w against xi. Returns figure handle.
     Annotates titles with:
     - Wall type, v_wall, alpha_n
     - alpha_plus (alpha just in front of wall)
     - r (ratio of enthalpies either side of wall)
     - xi_sh (shock speed)
     - w_0/w_n (ration of internal to external enthalpies)
     - ubar_f (mean square U = gamma(v) v)
     - K kinetic energy fraction
     - kappa (Espinosa et al efficiency factor)
     - omega (thermal energy relative to scalar potential energy, as measured by trace anomaly)
     Last two should sum to 1.
    """
    
    xi_even = np.linspace(1/Np,1-1/Np,Np)
    yscale_v = 0.0
    yscale_enth_max = 1.0
    yscale_enth_min = 1.0
    wn_max = 0.0
    
    ncols = 1
    fig_width = 8
    if multi==True:
        ncols = len(v_wall_list)
        fig_width = ncols*5

    f, ax = plt.subplots(2,ncols, figsize=(fig_width, 8), sharex='col', sharey='row', squeeze=False)
    f.subplots_adjust(hspace=0)
    f.subplots_adjust(wspace=0.1)
    
    n=0
    for v_wall, alpha_n in zip(v_wall_list, alpha_n_list):
        check_physical_params([v_wall, alpha_n])

        wall_type = identify_wall_type(v_wall, alpha_n)
    
        if wall_type=='Error':
            sys.stderr.write('plot_fluid_shells: error: no solution for v_wall = {}, alpha_n = {}\n'.format(v_wall,alpha_n))
            sys.exit(1)
            
        v, w, xi = fluid_shell(v_wall, alpha_n, Np)
        n_cs = np.int(np.floor(cs0*Np))
        n_sh = xi.size-2        
        v_sh = v_shock(xi_even)
        w_sh = w_shock(xi_even)
#    
#        n_wall = find_v_index(xi, v_wall)
#        n_cs = np.int(np.floor(cs0*Np))
#        n_sh = xi.size-2
    
#        r = w[n_wall]/w[n_wall-1]
#        alpha_plus = alpha_n*w[-1]/w[n_wall]
#    
#    
        # Plot
        yscale_v = max(max(v), yscale_v)
        yscale_enth_max = max(max(w),yscale_enth_max)
#        yscale_enth_min = min(min(w),yscale_enth_min)
        yscale_enth_min = 2*w[-1] - yscale_enth_max
        wn_max = max(w[-1],wn_max)
        
    # First velocity
        ax[0,n].plot(xi, v, 'b')
        if not wall_type == 'Detonation':
            ax[0,n].plot(xi_even[n_cs:], v_sh[n_cs:], 'k--', label=r'$v_{\rm sh}(\xi_{\rm sh})$')
        if not wall_type == 'Deflagration':
            v_minus_max = lorentz(xi_even, cs0)
            ax[0,n].plot(xi_even[n_cs:], v_minus_max[n_cs:], 'k-.', label=r'$\mu(\xi,c_{\rm s})$')

        if multi:
            n_wall = find_v_index(xi, v_wall)
            r = w[n_wall]/w[n_wall-1]
            alpha_plus = alpha_n*w[-1]/w[n_wall]
            
            ax[0,n].set_title(r'$\alpha_{{\rm n}} =  {:5.3f}$, $\alpha_+ =  {:5.3f}$, $r =  {:5.3f}$, $\xi_{{\rm sh}} =  {:5.3f}$'.format(
                alpha_n, alpha_plus, r, xi[-2]),size=14)

        ax[0,n].grid(True)
        
    # Then enthalpy
#        ax[1,n].plot(xi, np.ones_like(xi)*w[-1], '--', color='0.5')
        ax[1,n].plot(xi, w, 'b')
        if not wall_type == 'Detonation':
            ax[1,n].plot(xi_even[n_cs:n_sh], w_sh[n_cs:n_sh],'k--',label=r'$w_{\rm sh}(\xi_{\rm sh})$')
        else:
            wmax_det = (xi_even/cs0)*gamma2(xi_even)/gamma2(cs0)
            ax[1,n].plot(xi_even[n_cs:], wmax_det[n_cs:],'k-.',label=r'$w_{\rm max}$')

        if multi:
            ubarf2 = ubarf_squared(v, w, xi, v_wall)
            # Kinetic energy fraction of total (Bag equation of state)
            ke_frac = ubarf2/(0.75*(1 + alpha_n))
            # Efficiency of turning Higgs potential into kinetic energy
            kappa = ubarf2/(0.75*alpha_n)
            # and efficiency of turning Higgs potential into thermal energy
            dw = 0.75 * mean_enthalpy_change(v, w, xi, v_wall)/(0.75 * alpha_n * w[-1])
#            ax[1,n].set_title(r'$w_0/w_n = {:4.2}$, $\bar{{U}}_f = {:.3f}$, $K = {:5.3g}$, $\kappa = {:5.3f}$, $\omega = {:5.3f}$'.format(
#                      w[0]/w[-1],ubarf2**0.5,ke_frac, kappa, dw),size=14)
            ax[1,n].set_title(r'$K = {:5.3g}$, $\kappa = {:5.3f}$, $\omega = {:5.3f}$'.format(
                      ke_frac, kappa, dw),size=14)

            
        ax[1,n].set_xlabel(r'$\xi$')
        ax[1,n].grid(True)
        if multi:
            n += 1


    xscale_min = 0.0
    xscale_max = 1.0
    y_scale_enth_min = max(0.0,yscale_enth_min)
    ax[0,0].axis([xscale_min,xscale_max, 0.0, yscale_v*1.2])
    ax[1,0].axis([xscale_min, xscale_max, y_scale_enth_min, yscale_enth_max*1.1 - 0.1*wn_max])

    f.canvas.draw()
    ylabels = [tick.get_text() for tick in ax[1,0].get_yticklabels()]
    ax[1,0].set_yticklabels(ylabels[:-1])
    
    ax[0,0].set_ylabel(r'$v(\xi)$')
    plt.tight_layout()

    ax[1,0].set_ylabel(r'$w(\xi)$')
    plt.tight_layout()

    if save_string is not None:
        plt.savefig('shells_plot_vw_{}-{}_alphan_{:.3}-{:.3}{}'.format(
                    v_wall_list[0],v_wall_list[-1],
                    alpha_n_list[0],alpha_n_list[-1],
                    save_string))


    plt.show()
    
    return f
