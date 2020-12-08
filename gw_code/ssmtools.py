r"""
Functions for calculating velocity and gravitational wave power spectra from 
a first-order phase transition in the Sound Shell Model.

See Hindmarsh 2018, Hindmarsh & Hijazi 2019.

Author: Mark Hindmarsh 2015-20

Changes 06/20
- use analytic formula for high-k sin transforms.  Should eliminate spurious
high-k signal in GWPS from numerical error.
- sin_transform now handles array z, simplifying its calling elsewhere
- resample_uniform_xi function introduced to simply coding for sin_transform of lam
- Allow calls to power spectra and spectral density functions  
with 2-component params list, i.e. params = [v_wall, alpha_n] (parse_params)
exponential nucleation with parameters (1,) assumed.
- reduced NQDEFAULT from 2000 to 320, to reduce high-k numerical error when using numerical sin transform

Changes planned 06/20
- improve docstrings
- introduce function for physical GW power spectrum today
- Check default nucleation type for nu function.
- Allow first three letters to specify nucleation type

"""

# Python 2/3 compatibility
from __future__ import absolute_import, division, print_function

import sys
import numpy as np
from scipy.optimize import fsolve
from pttools.bubble import bubble as b
#import bubble.bubble as b
#import bubble as b

NXIDEFAULT = 2000 # Default number of xi points used in bubble profiles
NTDEFAULT  = 200   # Default number of T-tilde values for bubble lifetime distribution integration
NQDEFAULT  = 320  # Default number of wavevectors used in the velocity convolution integrations.
NPTDEFAULT = [NXIDEFAULT, NTDEFAULT, NQDEFAULT]

#It seems that NPTDEFAULT should be something like NXIDEFAULT/(2.pi), otherwise one 
#gets a GW power spectrum which drifts up at high k.
#
#The maximum trustworthy k is approx NXIDEFAULT/(2.pi)
#
#NTDEFAULT can be left as it is, or even reduced to 100

Z_ST_THRESH = 50    # Default dimensionless wavenumber above which to use approximation for 
                    # sin_transform, sin_transform_approx.
DZ_ST_BLEND = np.pi # Default wavenumber overlap for matching sin_transform_approx

T_TILDE_MAX = 20.0 # Maximum in bubble lifetime distribution integration
T_TILDE_MIN = 0.01 # Minimum in bubble lifetime distribution integration

DEFAULT_NUC_TYPE = 'exponential'
DEFAULT_NUC_PARM = (1,)

cs0 = b.cs0 # Default sound speed


def sin_transform_old(z, xi, v):
    """
     sin transform of v(xi), Fourier transform variable z. 
     xi and v arrays of the same shape, z can be an array of a different shape.
    """
    
    if isinstance(z,np.ndarray):
        array = np.sin(np.outer(z, xi)) * v
        I = np.trapz(array, xi)
    else:
        array = v * np.sin(z * xi)
        I = np.trapz(array, xi)
 
    return I


def envelope(xi,f):
    """
    Returns lists of xi, f pairs "outlining" function f.
    Helper function for sin_transform_approx. 
    Assumes that 
    - max(v) is acheived at a discontinuity (bubble wall)
    - f(xi) finishes at a discontinuity (shock)
    - at least the first element of f is zero
    
    xi1:  last zero value of f,             f1, value just before xi1
    xi_w: positiom of maximum f (wall)      f_m, value just before wall
                                            f_p, value just after wall
    x12:  last non-zero value of f (def)    f2 (at shock, or after wall)
          or 1st zero after wall (det)      
    """
    
    xi_nonzero = xi[np.nonzero(f)]
    xi1 = np.min(xi_nonzero)
    xi2 = np.max(xi_nonzero)
    ind1 = np.where(xi==xi1)[0][0] # where returns tuple, first element array
    ind2 = np.where(xi==xi2)[0][0]
    f1 = f[ind1 - 1] # in practice, f1 is always zero, or very close, so could drop.
    xi1 = xi[ind1 - 1] # line up f1 and xi1
    
    i_max_f = np.argmax(f)
    f_max = f[i_max_f]
    xi_w = xi[i_max_f] # max f always at wall
    
    df_at_max = f[i_max_f+1] - f[i_max_f-1]
    
#    print(ind1, ind2, [xi1,f1], [xi_w, f_max])
    
    if df_at_max > 0:
        # deflagration or hybrid, ending in shock.
        f_m = f[i_max_f - 1]
        f_p = f_max 
        f2 = f[ind2] 

    else:
        # detonation, nothing beyond wall
        f_m = f_max
        f_p = 0
        f2 = 0
    
    xi_list = [xi1, xi_w, xi_w, xi2]
    f_list  = [f1, f_m, f_p, f2]
        
    return xi_list, f_list 


def sin_transform_approx(z,xi,f):
    """
    Approximate sin transform of f(xi), Fourier transform variable z. 
    xi and f arrays of the same shape, z can be an array of a different shape.
    values $f_a$ and $f_b$, we have 
    $$
    \\int_{\\xi_a}^{\\xi_b} d\\xi f(\\xi) \\sin(z \\xi) \\to 
    - \\frac{1}{z} \\left(f_b \\cos(z \\xi_b) - f_a \\cos(z \\xi_a)\\right) + O(1/z^2)
    $$
    as $z \\to \\infty$.
    Function assumed piecewise continuous in intervals $[\\xi_1, \\xi_w]$ and 
    $[\\xi_w,\\xi_2]$.
     
    """
    
    [xi1, xi_w, _, xi2], [f1, f_m, f_p, f2]  = envelope(xi,f)     
    I  = -(f2*np.cos(z * xi2) - f_p*np.cos(z * xi_w))/z
    I += -(f_m*np.cos(z * xi_w) - f1*np.cos(z * xi1))/z
    return I


def sin_transform(z, xi, f, z_st_thresh=Z_ST_THRESH):
    """
     sin transform of f(xi), Fourier transform variable z. 
     xi and f arrays of the same shape, z can be an array of a different shape.
     For z > z_st_thresh, use approximation rather than doing the integral.
     Interpolate between  z_st_thresh - dz_blend < z < z_st_thresh.
    """
    dz_blend = DZ_ST_BLEND
    
    if isinstance(z,np.ndarray):
        lo = np.where(z <= z_st_thresh)
        z_lo = z[lo]
        array_lo = np.sin(np.outer(z_lo, xi)) * f
        I = np.trapz(array_lo, xi)
        
        if len(lo) < len(z):
            z_hi = z[np.where(z > z_st_thresh - dz_blend)]
            I_hi  = sin_transform_approx(z_hi,xi,f)

            if len(z_hi) + len(z_lo) > len(z):
                #If there are elements in the z blend range, then blend
                hi_blend = np.where(z_hi <= z_st_thresh)
                z_hi_blend = z_hi[hi_blend]
                lo_blend = np.where(z_lo > z_st_thresh - dz_blend)
                z_blend_max = max(z_hi_blend)
                z_blend_min = min(z_hi_blend)
                if z_blend_max > z_blend_min:
                    s = (z_hi_blend - z_blend_min)/(z_blend_max-z_blend_min)
                else:
                    s = 0.5
                frac = 3*s**2 - 2*s**3
                I[lo_blend] = I_hi[hi_blend] * frac + I[lo_blend]*(1 - frac)

            I = np.concatenate((I[lo],I_hi[z_hi > z_st_thresh]))
    else:
        if z <= z_st_thresh:
            array = f * np.sin(z * xi)
            I = np.trapz(array, xi)
        else:
            I = sin_transform_approx(z,xi,f)
 
    return I


#############################
# Sound shell model functions.
#############################
def A2_ssm_func(z, vw, alpha, npt=NPTDEFAULT, 
                method='e_conserving', de_method='standard', z_st_thresh=Z_ST_THRESH):
    """
     Returns the value of $|A(z)|^2$, 
     z is an array of scaled wavenumbers $z = kR_*$. 
     |Plane wave amplitude|^2 = T^3 |A(z)|^2
     method: (string) correct method for SSM is ``e_conserving``. 
             Also allows exploring effect of other incorrect 
             methods ``f_only`` and ``with_g``.
     de_method: How energy density fluctuation feeds into GW ps.  See A2_ssm_e_conserving.
     z_st_thresh: wavenumber at which to switch sin_transform to its approximation.
    """
    
    if method=='e_conserving':
        # This is the correct method (as of 12.18)
        A2 = A2_e_conserving(z, vw, alpha, npt, 'A2_only', de_method, z_st_thresh)
    elif method=='f_only':
        print('A2_ssm_func: f_only method, multiplying (f\')^2 by 2')
        f = f_ssm_func(z, vw, alpha, npt)
        df_dz = np.gradient(f) / np.gradient(z)
        A2 = 0.25 * (df_dz ** 2)
        A2 = A2*2
    elif method=='with_g':
        print('A2_ssm_func: with_g method')
        f = f_ssm_func(z, vw, alpha, npt)
        df_dz = np.gradient(f) / np.gradient(z)
        g = (z * df_dz + 2. * f)
        dg_dz = np.gradient(g) / np.gradient(z)
        A2 = 0.25 * (df_dz ** 2)
        A2 = A2 + 0.25 * (dg_dz ** 2 / (cs0 * z) ** 2)
    else:
        sys.stderr.write('A2_ssm_func: warning: method not known, should be\n')
        sys.stderr.write('             [e_conserving | f_only | with_g]\n')
        sys.stderr.write('             defaulting to e_conserving\n')
        A2 = A2_e_conserving(z, vw, alpha, npt)
        
    return A2


def resample_uniform_xi(xi, f, nxi=NPTDEFAULT[0]):
    """
    Provide uniform resample of function defined by (x,y) = (xi,f).
    Returns f interpolated and the uniform grid of nxi points in range [0,1]
    """
    xi_re = np.linspace(0,1-1/nxi,nxi) 
    return xi_re, np.interp(xi_re, xi, f)


def A2_e_conserving(z, vw, alpha_n, npt=NPTDEFAULT, 
                    ret_vals='A2_only', de_method='standard', z_st_thresh=Z_ST_THRESH):
    """
     Returns the value of $|A(z)|^2$, where |Plane wave amplitude|^2 = T^3 |A(z)|^2, 
     calculated from self-similar hydro solution obtained with ``bubble.fluid_shell``.
     z is an array of scaled wavenumbers $z = kR_*$. 
     de_method: 'standard' (e-conserving) method is only accurate to 
     linear order, meaning that there is an apparent $z^0$ piece at very low $z$,
     and may exaggerate the GWs at low vw. ATM no other de_methods, but argument 
     allows trials.
     
    """
    nxi = npt[0]
#    xi_re = np.linspace(0,1-1/nxi,nxi) 
    # need to resample for lam = de/w, as some non-zero points are very far apart
    v_ip, w_ip, xi = b.fluid_shell(vw, alpha_n, nxi)

#    f = np.zeros_like(z)
#    for j in range(f.size):
#        f[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi, v_ip, z_st_thresh)
    f = (4.*np.pi/z) * sin_transform(z, xi, v_ip, z_st_thresh)

    v_ft = np.gradient(f) / np.gradient(z)

    # Now get and resample lam = de/w
    if de_method == 'alternate':
        lam_orig = b.de_from_w_new(v_ip,w_ip,xi,vw,alpha_n)/w_ip[-1]
    else:
        lam_orig = b.de_from_w(w_ip,xi,vw,alpha_n)/w_ip[-1]

    lam_orig += w_ip*v_ip*v_ip/w_ip[-1] #   This doesn't make much difference at small alpha
    xi_re, lam_re = resample_uniform_xi(xi, lam_orig, nxi)

#    lam_re = np.interp(xi_re,xi,lam_orig)
#    lam_ft = np.zeros_like(z)
#    for j in range(lam_ft.size):
#        lam_ft[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi_re, xi_re*lam_re, 
#              z_st_thresh=max(z)) # Need to fix problem with ST of lam for detonations
    lam_ft = (4.*np.pi/z) * sin_transform(z, xi_re, xi_re*lam_re, z_st_thresh) 
    
    A2 = 0.25 * (v_ft**2 + (cs0*lam_ft)**2)

    if ret_vals == 'A2_only':
        return A2
    else:
        return A2, v_ft**2/2, (cs0*lam_ft)**2/2
    
 
def A2_e_conserving_file(z, filename, alpha, skip=1, npt=NPTDEFAULT, z_st_thresh=Z_ST_THRESH):
    """
     Returns the value of $|A(z)|^2$, where |Plane wave amplitude|^2 = T^3 |A(z)|^2, 
     calculated from file, outout by ``spherical-hydro-code``.
     z is an array of scaled wavenumbers $z = kR_*$. 
     Uses method respecting energy conservation, although only accurate to 
     linear order, meaning that there is an apparent $z^0$ piece at very low $z$.
    """
    print('ssmtools.A2_e_conserving_file: loading v(xi), e(xi) from {}'.format(filename))
    try:
        with open(filename) as f:
            t = float(f.readline())
        r, v_all, e_all = np.loadtxt(filename, usecols=(0,1,4), unpack=True, skiprows=skip)
    except IOError:
        sys.exit('ssmtools.A2_e_conserving_file: error loading file:' + filename)
    
    xi_all = r/t
    wh_xi_lt1 = np.where(xi_all < 1.)
    print('ssmtools.A2_e_conserving_file: interpolating v(xi), e(xi) from {} to {} points'.format(len(wh_xi_lt1[0]), npt[0]) )
    xi_lt1 = np.linspace(0.,1.,npt[0])
    v_xi_lt1 = np.interp(xi_lt1,xi_all,v_all)
    e_xi_lt1 = np.interp(xi_lt1,xi_all,e_all)
#    f = np.zeros_like(z)
#    for j in range(f.size):
#        f[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi_lt1, v_xi_lt1)
    f = (4.*np.pi/z) * sin_transform(z, xi_lt1, v_xi_lt1)

    v_ft = np.gradient(f) / np.gradient(z)
    e_n = e_xi_lt1[-1]
    def fun(x):
        return x - b.w(e_n, 0., alpha*(0.75*x))
    w_n0 = b.w(e_n, 0., alpha*(e_n)) # Correct only in Bag, probably good enough
    w_n = fsolve(fun, w_n0)[0] # fsolve returns array, want float 
    lam = (e_xi_lt1 - e_n)/w_n
    print('ssmtools.A2_e_conserving_file: initial guess w_n0: {}, final {}'.format(w_n0,w_n))
    
#    lam_ft = np.zeros_like(z)
#    for j in range(lam_ft.size):
#        lam_ft[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi_lt1, xi_lt1*lam,
#              z_st_thresh=max(z)) # Need to fix problem with ST of lam for detonations
    lam_ft = (4.*np.pi/z) * sin_transform(z, xi_lt1, xi_lt1*lam,
              z_st_thresh) 
    
    return 0.25 * (v_ft**2 + (cs0*lam_ft)**2)    

    
def f_ssm_func(z, vw, alpha_n, npt=NPTDEFAULT, z_st_thresh=Z_ST_THRESH):
    """
     3D FT of radial fluid velocity v(r) from Sound Shell Model fluid profile. 
     z is array of scaled wavenumbers z = kR*
    """
    nxi = npt[0]
    v_ip, _, xi = b.fluid_shell(vw, alpha_n, nxi)

#    f_ssm = np.zeros_like(z)
#    for j in range(f_ssm.size):
#        f_ssm[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi, v_ip)
    f_ssm = (4.*np.pi/z) * sin_transform(z, xi, v_ip, z_st_thresh)

    return f_ssm


def lam_ssm_func(z, vw, alpha_n, npt=NPTDEFAULT, de_method='standard', z_st_thresh=Z_ST_THRESH):
    """
     3D FT of radial energy perturbation from Sound Shell Model fluid profile
     z is array of scaled wavenumbers z = kR*
    """
    nxi = npt[0]
#    xi_re = np.linspace(0,1-1/nxi,nxi) # need to resample for lam = de/w
    v_ip, w_ip, xi = b.fluid_shell(vw, alpha_n, nxi)

    if de_method == 'alternate':
        lam_orig = b.de_from_w_new(v_ip,w_ip,xi,vw,alpha_n)/w_ip[-1]
    else:
        lam_orig = b.de_from_w(w_ip,xi,vw,alpha_n)/w_ip[-1]
    xi_re,lam_re = resample_uniform_xi(xi,lam_orig,nxi)

#    lam_ft = np.zeros_like(z)
#    for j in range(lam_ft.size):
#        lam_ft[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi_re, xi_re*lam_re,
#              z_st_thresh=max(z)) # Need to fix problem with ST of lam for detonations

    lam_ft = (4.*np.pi/z) * sin_transform(z, xi_re, xi_re*lam_re,
              z_st_thresh) 

    return lam_ft


def g_ssm_func(z, vw, alpha, npt=NPTDEFAULT):
    """
     3D FT of radial fluid acceleration \dot{v}(r) from Sound Shell Model fluid profile. 
     z is array of scaled wavenumbers z = kR*.
    """
    f_ssm = f_ssm_func(z, vw, alpha, npt)
    df_ssmdz = np.gradient(f_ssm) / np.gradient(z)
    g_ssm = (z * df_ssmdz + 2. * f_ssm)
    return g_ssm


def f_file(z_arr, t, filename, skip=0, npt=NPTDEFAULT,z_st_thresh=Z_ST_THRESH):
    """
     3D FT of radial fluid velocity v(r) from file. 
     z is array of scaled wavenumbers z = kR*
    """
    print('ssmtools.f_file: loading v(xi) from {} at time {}'.format(filename,t))
    try:
        r, v_all = np.loadtxt(filename, usecols=(0,1), unpack=True, skiprows=skip)
    except IOError:
        sys.exit('ssmtools.f_file: error loading file:' + filename)
    
    xi_all = r/t
    wh_xi_lt1 = np.where(xi_all < 1.)
    print('ssmtools.f_file: interpolating v(xi) from {} to {} points'.format(len(wh_xi_lt1[0]), npt[0]) )
#    xi_lt1 = np.linspace(0.,1.,npt[0])
#    v_xi_lt1 = np.interp(xi_lt1,xi_all,v_all)
    xi_lt1, v_xi_lt1 = resample_uniform_xi(xi_all, v_all, npt[0])
#    f = np.zeros_like(z_arr)
#    for n, z in enumerate(z_arr):
#        f[n] = (4*np.pi/z)*sin_transform(z, xi_lt1, v_xi_lt1, z_st_thresh)
    f = (4*np.pi/z_arr)*sin_transform(z_arr, xi_lt1, v_xi_lt1, z_st_thresh)
    
    return f


def g_file(z, t, filename, skip=0):
    """
     3D FT of radial fluid acceleration \dot{v}(r) from file
     z is array of scaled wavenumbers z = kR*
    """
    f = f_file(z, t, filename, skip)
    df_dz = np.gradient(f) / np.gradient(z)
    g = (z * df_dz + 2. * f)
    return g


def nu(T, nuc_type='simultaneous', args=(1,)):
    """
     Bubble lifetime distribution function as function of (dimensionless) time T. 
     ``nuc_type`` allows ``simultaneous`` or ``exponential`` bubble nucleation. 
    """
    if nuc_type == "simultaneous":
        a = args[0]
        dist = 0.5 * a * (a*T)**2 * np.exp(-(a*T)**3 / 6)
    elif nuc_type == "exponential":
        a = args[0]
        dist = a * np.exp(-a*T)
    else:
        sys.stderr.write('error: nu: nucleation type not recognised')
        sys.exit(1)

    return dist


def pow_spec(z,spec_den):
    """
     Power spectrum from spectral density at dimensionless wavenumber z.
    """
    return z**3  / (2. * np.pi ** 2) * spec_den


def parse_params(params):
    vw = params[0]
    alpha = params[1]
    if len(params) > 2:
        nuc_type = params[2]
    else:
        nuc_type = DEFAULT_NUC_TYPE
    if len(params) > 3:
        nuc_args = params[3]
    else:
        nuc_args = DEFAULT_NUC_PARM

    return vw, alpha, nuc_type, nuc_args


def spec_den_v(z, params, npt=NPTDEFAULT, filename=None, skip=1, 
               method='e_conserving', de_method='standard', z_st_thresh=Z_ST_THRESH):
    """
     Returns dimensionless velocity spectral density $\bar{P}_v$, given array $z = qR_*$ and parameters:
        vw = params[0]       scalar
        alpha = params[1]    scalar
        nuc_type = params[2] string [exponential* | simultaneous]
        nuc_args = params[3] tuple  default (1,)
     
     Gets fluid velocity profile from bubble toolbox or from file if specified. 
     Convolves 1-bubble Fourier transform $|A(q T)|^2$ with bubble wall 
     lifetime distribution $\nu(T \beta)$ specified by ``nuc_type`` and ``nuc_args``.
    """
    b.check_physical_params(params)
    
    
    nz = z.size
#    nxi = npt[0]
    nt = npt[1]
#    nq = npt[2]

    log10zmin = np.log10(min(z))
    log10zmax = np.log10(max(z))
    dlog10z = (log10zmax - log10zmin)/nz

    tmin = T_TILDE_MIN
    tmax = T_TILDE_MAX
    log10tmin = np.log10(tmin)
    log10tmax = np.log10(tmax)

    t_array = np.logspace(log10tmin, log10tmax, nt)

    qT_lookup = 10**(np.arange(log10zmin + log10tmin, log10zmax + log10tmax, dlog10z))

    if filename is None:
        vw, alpha, nuc_type, nuc_args = parse_params(params)
        A2_lookup = A2_ssm_func(qT_lookup, vw, alpha, npt, method, de_method, z_st_thresh)
    else:
        vw, alpha, nuc_type, nuc_args = parse_params(params)
        A2_lookup = A2_e_conserving_file(qT_lookup, filename, alpha, skip, npt, z_st_thresh)

    b_R = (8. * np.pi) ** (1./3.) # $\beta R_* = b_R v_w $

    def qT_array(qRstar,Ttilde):
        return qRstar * Ttilde / (b_R *vw )

    A2_2d_array = np.zeros((nz, nt))
    
    for i in range(nz):
        A2_2d_array[i] = np.interp(qT_array(z[i],t_array), qT_lookup, A2_lookup)

    array2 = np.zeros(nt)
    sd_v = np.zeros(nz) # array for spectral density of v
    factor = 1. / (b_R * vw) ** 6
    factor = 2*factor # because spectral density of v is 2 * P_v

    for s in range(nz):
        array2 = t_array ** 6 * nu(t_array, nuc_type, nuc_args) * A2_2d_array[s]
        D = np.trapz(array2, t_array)
        sd_v[s] = D * factor

    return sd_v


def spec_den_gw_scaled(xlookup, P_vlookup, z=None):
    """
     Spectral density of scaled gravitational wave power at values of kR* given 
     by input z array, or at len(xlookup) values of kR* between the min and max 
     of xlookup where the GW power can be computed. 
     (xlookup, P_vlookup) is used as a lookup table to specify function. 
     P_vlookup is the spectral density of the FT of the velocity field, 
     not the spectral density of plane wave coeffs, which is lower by a 
     factor of 2.
    """

    if z is None:
        nx = len(xlookup)
        zmax = max(xlookup)  / ( 0.5 * (1. + cs0) / cs0)
        zmin = min(xlookup) / (0.5 * (1. - cs0) / cs0)
        z = np.logspace(np.log10(zmin), np.log10(zmax), nx)
    else:
#        nx = len(z)
        nx = len(xlookup)
        xlargest = max(z)  * 0.5 * (1. + cs0) / cs0
        xsmallest = min(z) * 0.5 * (1. - cs0) / cs0
    
        if max(xlookup) < xlargest or min(xlookup) > xsmallest:
            sys.exit("spec_den_gw_scaled: error: range of xlookup not large enough")

    p_gw = np.zeros_like(z)
    
    for i in range(z.size):
        xplus = z[i] / cs0 * (1. + cs0) / 2.
        xminus = z[i] / cs0 * (1. - cs0) / 2.
        x = np.logspace(np.log10(xminus), np.log10(xplus), nx)
        integrand = (x - xplus)**2 * (x - xminus)**2 / x / (xplus + xminus - x) * np.interp(x,
                xlookup, P_vlookup) * np.interp((xplus + xminus - x), xlookup, P_vlookup)
        p_gw_factor = ((1 - cs0**2)/cs0**2)**2 / (4*np.pi*z[i]*cs0)
        p_gw[i] = p_gw_factor * np.trapz(integrand, x)

    # Here we are using G = 2P_v (v spec den is twice plane wave amplitude spec den).
    # Eq 3.48 in SSM paper gives a factor 3.Gamma^2.P_v.P_v = 3 * (4/3)^2.P_v.P_v
    # Hence overall should use (4/3).G.G
    return (4./3.)*p_gw, z


def power_v(z, params, npt=NPTDEFAULT, filename=None, skip=1, 
            method='e_conserving', de_method='standard', z_st_thresh=Z_ST_THRESH):
    """
    Power spectrum of velocity field in Sound Shell Model.
        vw = params[0]       scalar
        alpha = params[1]    scalar
        nuc_type = params[2] string [exponential* | simultaneous]
        nuc_args = params[3] tuple  default (1,)
    """
    b.check_physical_params(params)
    
    p_v = spec_den_v(z,params,npt,filename,skip,method, de_method)
    return pow_spec(z,p_v)    

    
def power_gw_scaled(z, params, npt=NPTDEFAULT, filename=None, skip=1, 
                    method='e_conserving', de_method='standard', z_st_thresh=Z_ST_THRESH):
    """
     Scaled GW power spectrum at array of z = kR* values, where R* is mean bubble centre
     separation and k is comoving wavenumber.  To convert to predicted spectrum, 
     multiply by $(H_n R_*)(H_n \tau_v)$, where $H_n$ is the Hubble rate at the 
     nucleation time, and $\tau_v$ is the lifetime of the shear stress source.

    Input parameters
        vw = params[0]       scalar  (required) [0 < vw < 1]
        alpha = params[1]    scalar  (required) [0 < alpha_n < alpha_n_max(v_w)]
        nuc_type = params[2] string  (optional) [exponential* | simultaneous]
        nuc_args = params[3] tuple   (optional) default (1,)

    Steps:
     1. Getting velocity field spectral density
     2. Geeting gw spectral density
     3. turning SD into power
    """
    if np.any(z <= 0.0):
        sys.exit('power_gw_scaled: error: z values must all be positive\n')
    
    b.check_physical_params(params)
    
    eps = 1e-8 # Seems to be needed for max(z) <= 100. Why?
#    nx = len(z) - this can be too few for velocity PS convolutions
    nx = npt[2]
    xmax = max(z) * (0.5 * (1. + cs0) / cs0) + eps
    xmin = min(z) * (0.5 * (1. - cs0) / cs0) - eps

    x = np.logspace(np.log10(xmin), np.log10(xmax), nx)

    sd_v = spec_den_v(x, params, npt, filename, skip, method, de_method, z_st_thresh)
    sd_gw, y = spec_den_gw_scaled(x, sd_v, z)
    return pow_spec(z, sd_gw)


