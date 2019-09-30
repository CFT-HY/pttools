from __future__ import absolute_import, division, print_function

import sys
sys.path.append('../bubble')

import numpy as np
import bubble.bubble as b

#NZDEFAULT = 2000  # Default Number of points used in the numerical integrations
NXIDEFAULT = 2000 # Default Number of points used ifor FFT
NTDEFAULT = 200 # Number of T-tilde values to integrrate bubble lifetime distribution over


NPTDEFAULT = [NXIDEFAULT, NTDEFAULT]

cs0 = np.sqrt(1./3.)

def sin_transform(z, xi, v):
    # z is a float, xi and v arrays of the same shape
    array = v * np.sin(z * xi)
    I = np.trapz(array, xi)
    return I


#############################
# Sound shell model functions.
#############################
def A2_ssm_func(z, vw, alpha, npt=NPTDEFAULT, method='e_conserving'):
    # Returns the value of |A(z)|^2 
    # z is array of scaled wavenumbers z = kR*
    # Correct method for SSM is e_conserving.
    # Also allows exploring effect of other methods.
    
    if method=='e_conserving':
        # This is the correct method (as of 12.18)
        A2 = A2_e_conserving(z, vw, alpha, npt)
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

    return A2


def A2_e_conserving(z, vw, alpha_n, npt=NPTDEFAULT):
    # Returns the value of |A(z)|2 at z array points (z = kR*)
    # z is array of scaled wavenumbers z = kR*
    # |Plane wave amplitude|^2 = T^3 |A(z)|2
    # uses method respecting energy conservation, although only accurate to 
    # linear order, meaning that there is an apparent k^0 piece at low k
    nxi = npt[0]
    xi_re = np.linspace(0,1-1/nxi,nxi) # need to resample for lam = de/w
    
    v_ip, w_ip, xi = b.fluid_shell(vw, alpha_n, nxi)

    f = np.zeros_like(z)
    for j in range(f.size):
        f[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi, v_ip)

    v_ft = np.gradient(f) / np.gradient(z)

    # Now get and resample lam = de/w
    lam_orig = b.de_from_w(w_ip,xi,vw,alpha_n)/w_ip[-1]
    lam_re = np.interp(xi_re,xi,lam_orig)
    lam_ft = np.zeros_like(z)
    for j in range(lam_ft.size):
        lam_ft[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi_re, xi_re*lam_re)
    
    return 0.25 * (v_ft - cs0*lam_ft)**2
    
 
def A2_e_conserving_file(z, filename, alpha, skip=1, npt=NPTDEFAULT):
    # Calculate |A(z)|^2 from velocity and energy profiles in file (produced by 1D spherical hydro code)
    # z is array of scaled wavenumbers z = kR*
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
    f = np.zeros_like(z)
    for j in range(f.size):
        f[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi_lt1, v_xi_lt1)

    v_ft = np.gradient(f) / np.gradient(z)
    e_n = e_xi_lt1[-1]
    w_n = b.w(e_n, 0., alpha*(e_n)) # Correct only in Bag, probably good enough
    lam = (e_xi_lt1 - e_n)/w_n
    
    lam_ft = np.zeros_like(z)
    for j in range(lam_ft.size):
        lam_ft[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi_lt1, xi_lt1*lam)
    
    return 0.25 * (v_ft - cs0*lam_ft)**2    

    
def f_ssm_func(z, vw, alpha_n, npt=NPTDEFAULT):
    # 3D FT of radial fluid velocity v(r) from Sound Shell Model fluid profile
    # z is array of scaled wavenumbers z = kR*
    nxi = npt[0]
    # nt = npt[1]
    f_ssm = np.zeros_like(z)
    v_ip, _, xi = b.fluid_shell(vw, alpha_n, nxi)

    for j in range(f_ssm.size):
        f_ssm[j] = (4.*np.pi/z[j]) * sin_transform(z[j], xi, v_ip)

    return f_ssm


def g_ssm_func(z, vw, alpha, npt=NPTDEFAULT):
    # 3D FT of radial fluid acceleration \dot{v}(r) from Sound Shell Model fluid profile
    # z is array of scaled wavenumbers z = kR*
    f_ssm = f_ssm_func(z, vw, alpha, npt)
    df_ssmdz = np.gradient(f_ssm) / np.gradient(z)
    g_ssm = (z * df_ssmdz + 2. * f_ssm)
    return g_ssm


def f_file(z_arr, t, filename, skip=0, npt=NPTDEFAULT):
    # 3D FT of radial fluid velocity v(r) from file
    # z is array of scaled wavenumbers z = kR*
    print('ssmtools.f_file: loading v(xi) from {} at time {}'.format(filename,t))
    try:
        r, v_all = np.loadtxt(filename, usecols=(0,1), unpack=True, skiprows=skip)
    except IOError:
        sys.exit('ssmtools.f_file: error loading file:' + filename)
    
    xi_all = r/t
    wh_xi_lt1 = np.where(xi_all < 1.)
    print('ssmtools.f_file: interpolating v(xi) from {} to {} points'.format(len(wh_xi_lt1[0]), npt[0]) )
    xi_lt1 = np.linspace(0.,1.,npt[0])
    v_xi_lt1 = np.interp(xi_lt1,xi_all,v_all)
    f = np.zeros_like(z_arr)
    for n, z in enumerate(z_arr):
        f[n] = (4*np.pi/z)*sin_transform(z, xi_lt1, v_xi_lt1)
    
    return f


def g_file(z, t, filename, skip=0):
    # 3D FT of radial fluid acceleration \dot{v}(r) from file
    # z is array of scaled wavenumbers z = kR*
    f = f_file(z, t, filename, skip)
    df_dz = np.gradient(f) / np.gradient(z)
    g = (z * df_dz + 2. * f)
    return g


def nu(T, nuc_type='simultaneous', args=(1,)):
    # returns the value of the bubble lifetime distribution function 
    # at (dimensionless) time T
    if nuc_type == "simultaneous":
        a = args[0]
        dist = 0.5 * a * (a*T)**2 * np.exp(-(a*T)**3 / 6)
    elif nuc_type == "exponential":
        n = args[0]
        dist = (1. / np.math.factorial(n)) * T ** n * np.exp(-T)
        # delT = args[0]
        # e_fac = np.exp(delT)
        # norm = 1./(1 - np.exp(-e_fac))
        # dist = norm * e_fac*np.exp(-T) * np.exp( - e_fac*np.exp(-T))
    elif nuc_type == "TWW":
        Tm = args[0]
        s = args[1]
        norm = (1 - np.exp(-np.exp(s*Tm)))/s
        dist = np.exp(s*(Tm-T) - np.exp(s*(Tm - T)))/norm
    else:
        sys.stderr.write('error: nu: nucleation type not recognised')
        sys.exit(1)

    return dist


def pow_spec(z,spec_den):
    return z**3  / (2. * np.pi ** 2) * spec_den


def spec_den_v(z, params, npt=NPTDEFAULT, filename=None, skip=1, method='e_conserving'):
    # Gets fluid velocity profile from bubble toolbox or from file
    # Returns dimensionless velocity spectral density $\bar{P}_v$ , given array $z = qR_*$ and parameters
    # Convolves 1-bubble Fourier transform $|A(q T)|^2$ with bubble wall lifetime distribution $\nu(T \beta)$.
    # params are:
    #        vw = params[0]       scalar
    #        alpha = params[1]    scalar
    #        nuc_type = params[2] string [simultaneous | exponential | TWW]
    #        nuc_args = params[3] tuple
    
    nz = z.size
#    nxi = npt[0]
    nt = npt[1]

    log10zmin = np.log10(min(z))
    log10zmax = np.log10(max(z))
    dlog10z = (log10zmax - log10zmin)/nz

    tmin = 0.01
    tmax = 20
    log10tmin = np.log10(tmin)
    log10tmax = np.log10(tmax)

    t_array = np.logspace(log10tmin, log10tmax, nt)

    qT_lookup = 10**(np.arange(log10zmin + log10tmin, log10zmax + log10tmax, dlog10z))

    if filename is None:
        vw = params[0]
        alpha = params[1]
        nuc_type = params[2]
        nuc_args = params[3]
        A2_lookup = A2_ssm_func(qT_lookup, vw, alpha, npt, method)
    else:
        vw = params[0]
        alpha = params[1]
        nuc_type = params[2]
        nuc_args = params[3]
        A2_lookup = A2_e_conserving_file(qT_lookup, filename, alpha, skip, npt)

    b_R = (8. * np.pi) ** (1./3.) # $\beta R_* = b_R v_w $
## The following degrades the fit, so leave out - think why.
#    if nuc_type == 'simultaneous':
#        print('spec_den_v: simultaneous nucleation - adjusting $\beta R_*$')
#        b_R = b_R*3/6**(1./3.)
    def qT_array(qRstar,Ttilde):
        return qRstar * Ttilde / (b_R *vw)

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


def spec_den_gw_scaled(xlookup, P_vlookup, y=None):
    # Returns spectral density of scaled gravitational wave power at values of kR* given 
    # by input y array, or at len(xlookup) values of kR* between the min and max of xlookup
    # where the GW power can be computed.
    # (xlookup, P_vlookup) is used as a lookup table to specify function
    # P_vlookup is the spectral density of the FT of the velocity field, 
    # not the spec den of plane wave coeffs.

    if y is None:
        nz = len(xlookup)
        ymax = max(xlookup)  / ( 0.5 * (1. + cs0) / cs0)
        ymin = min(xlookup) / (0.5 * (1. - cs0) / cs0)
        y = np.logspace(np.log10(ymin), np.log10(ymax), nz)
    else:
        nz = len(y)
        xlargest = max(y)  * 0.5 * (1. + cs0) / cs0
        xsmallest = min(y) * 0.5 * (1. - cs0) / cs0
    
        if max(xlookup) < xlargest or min(xlookup) > xsmallest:
            sys.exit("spec_den_gw_scaled: error: range of xlookup not large enough")

    p_gw = np.zeros_like(y)

    for i in range(nz):
        xplus = y[i] / cs0 * (1. + cs0) / 2.
        xminus = y[i] / cs0 * (1. - cs0) / 2.
        x = np.logspace(np.log10(xminus), np.log10(xplus), nz)
        integrand = (x - xplus)**2 * (x - xminus)**2 / x / (xplus + xminus - x) * np.interp(x,
                xlookup, P_vlookup) * np.interp((xplus + xminus - x), xlookup, P_vlookup)
        p_gw_factor = ((1 - cs0**2)/cs0**2)**2 / (4*np.pi*y[i]*cs0)
        p_gw[i] = p_gw_factor * np.trapz(integrand, x)

    # Here we are using G = 2P_v (v spec den is twice plane wave amplitude spec den)
    return (4./3.)*p_gw, y


def power_v(z, params, npt=NPTDEFAULT, filename=None, skip=1, method='e_conserving'):
    p_v = spec_den_v(z,params,npt,filename,skip,method)
    return pow_spec(z,p_v)    

    
def power_gw_scaled(z, params, npt=NPTDEFAULT, filename=None, skip=1, method='e_conserving'):
    # Computes scaled GW power spectrum at array of z = kR* values by
    # 1. Getting velocity field spectral density
    # 2. Geeting gw spectral density
    # 3. turning SD into power
    eps = 1e-8 # Seems to be needed for max(z) <= 100. Why?
    nz = len(z)
    xmax = max(z) * ( 0.5 * (1. + cs0) / cs0) + eps
    xmin = min(z) * (0.5 * (1. - cs0) / cs0) - eps
    x = np.logspace(np.log10(xmin), np.log10(xmax), nz)

    sd_v = spec_den_v(x, params, npt, filename, skip, method)
    sd_gw, y = spec_den_gw_scaled(x, sd_v, z)
    return pow_spec(z, sd_gw)


