from __future__ import absolute_import, division, print_function

import sys
sys.path.append('../bubble')

import numpy as np
from scipy.integrate import odeint
import bubble as b

# Number of T-tilde values to integrrate bubble lifetime distribution over
NTDEFAULT = 200
NZDEFAULT = 2000   # Default Number of points used in the numerical simulations & integrations
NXIDEFAULT = 2000 # Default Number of points used ifor FFT

NPTDEFAULT = [NZDEFAULT, NXIDEFAULT, NTDEFAULT]

def vpdef(vm, alpha):
    # $\textcolor{red}{v_+}$ in terms of $\textcolor{red}{v_-}$ & Strength of interaction ($\textcolor{red}{\alpha}$) for deflagrations in the wall frame
    return np.fabs((1.0 / 6. / vm + vm / 2.0 - np.sqrt(
        (1.0 / 6. / vm + vm / 2.0) ** 2 + alpha ** 2 + 2.0 / 3.0 * alpha - 1.0 / 3.0)) / (1 + alpha))


def vmdet(vp, alpha):
    # $\textcolor{red}{v_-}$ in terms of $\textcolor{red}{v_+}$ & Strength of interaction ($\textcolor{red}{\alpha}$) for detonations in the wall frame
    return (((1.0 + alpha) ** 2 * vp ** 2 - alpha ** 2 - 2. / 3. * alpha + 1. / 3.) /
            (vp + vp * alpha) + np.sqrt((((1.0 + alpha) ** 2 * vp ** 2 -
                                          alpha ** 2 - 2. / 3. * alpha + 1. / 3.) /
                                         (vp + vp * alpha)) ** 2 - 4. / 3.)) / 2.


def lt(v1, v2):
    # Lorentz Transformation
    return np.fabs(v1 - v2) / (1. - v1 * v2)


def gamma(v):
    # Lorentz Factor $\textcolor{red}{\gamma}$
    return 1. / np.sqrt(1. - v ** 2)


def cs(T):
    # Speed of sound as a function of temperature
    # For our purposes we approximate as sqrt(1/3) in the radiation era
    return np.sqrt(1. / 3.)


def v_ip_nonlin(n, vw, alpha, npt, wall_type):
    # Returns the invariant velocity profile
    nxi = npi[1]
    def diff(y, t):
        fi = 2.0 * y / t
        se = 1.0 / (1 - y ** 2.0)
        th = 1.0 - t * y
        u = (t - y) / (1. - t * y)
        bra = u ** 2.0 / cs(0) ** 2 - 1.0
        dydt = fi / se / th / bra
        return dydt

    if wall_type == 'Detonation':
        v_max = lt(vw, vmdet(vw, alpha))
        xi = np.logspace(np.log10(vw), np.log10(cs(0)), nxi)
        return odeint(diff, v_max, xi), xi
    if wall_type == 'Deflagration':
        v_max = lt(vw, vpdef(vw, alpha))
        t = np.logspace(np.log10(vw), np.log10(1.0), nxi)
        y = odeint(diff, v_max, t)
        i = 0
        found = 0
        while i < Np and found == 0:
            if lt(y[i], t[i]) * t[i] > 1. / 3.:
                xish = t[i - 1]
                found = 1
            i = i + 1
        xi = np.logspace(np.log10(vw), np.log10(xish), nxi)
        return odeint(diff, v_max, xi), xi
    if wall_type == 'Hybrid':
        v_max1 = lt(vw, cs(0))
        xi1 = np.logspace(np.log10(vw), np.log10(cs(0)), nxi)
        g1 = odeint(diff, v_max1, xi1)
        v_max2 = lt(vw, vpdef(cs(0), alpha))
        t = np.logspace(np.log10(vw), np.log10(1.0), nxi)
        y = odeint(diff, v_max2, t)
        i = 0
        found = 0
        while i < Np and found == 0:
            if lt(y[i], t[i]) * t[i] > 1. / 3.:
                xish = t[i - 1]
                found = 1
            i = i + 1
        xi2 = np.logspace(np.log10(vw), np.log10(xish), nxi)
        g2 = odeint(diff, v_max2, xi2)
        ar1 = np.zeros(nxi)
        ar2 = np.zeros(nxi)
        k = 0
        while k < Np:
            ar1[k] = np.asscalar(g1[k])
            ar2[k] = np.asscalar(g2[k])
            k = k + 1
        xi = np.logspace(np.log10(cs(0)), np.log10(xish), nxi)
        vip = np.zeros(nxi)
        j = 0
        while j < Np:
            if xi[j] < vw:
                vip[j] = np.interp(xi[j], sorted(xi1), sorted(ar1))
            if xi[j] > vw:
                vip[j] = np.interp(xi[j], xi2, ar2)
            j = j + 1
        return vip, xi

#############################
# Sound shell model functions.
#############################
def f_nonlin_func(z, vw, alpha, wall_type='Calculate', npt=NPTDEFAULT):
    # Returns the value of f(z) at a specific point z
    # Fourier sine transform
    nz = npt[0]
    nxi = npt[1]
    # nt = npt[2]
    f_nonlin = np.zeros_like(z)
    array = np.zeros(nxi)
    # v_ip, xi = v_ip_nonlin(n, vw, alpha, np, wall_type)
    v_ip, _, xi = b.fluid_shell(vw, alpha, wall_type, nxi)

    for j in range(f_nonlin.size):
        def f(z_):
            array = v_ip * np.sin(z_ * xi)
            I = np.trapz(array, xi)
            return 4. * np.pi / z_ * I

        f_nonlin[j] = f(z[j])
    return f_nonlin


def g_nonlin_func(z, vw, alpha, wall_type='Calculate', npt=NPTDEFAULT):
    # Returns the value of g(z) at a specific point z
    f_nonlin = f_nonlin_func(z, vw, alpha, wall_type, npt)
    df_nonlindz = np.gradient(f_nonlin) / np.gradient(z)
    g_nonlin = (z * df_nonlindz + 2. * f_nonlin)
    return g_nonlin


def A_nonlin_func(z, vw, alpha, wall_type='Calculate', npt=NPTDEFAULT):
    # Returns the value of |A(z)|2 at a specific point z
    f_nonlin = f_nonlin_func(z, vw, alpha, wall_type, npt)
    df_nonlindz = np.gradient(f_nonlin) / np.gradient(z)
    g_nonlin = (z * df_nonlindz + 2. * f_nonlin)
    dg_nonlindz = np.gradient(g_nonlin) / np.gradient(z)
    A_nonlin = 0.25 * (df_nonlindz ** 2 + dg_nonlindz ** 2 / (cs(0) * z) ** 2)
    return A_nonlin


def nu(T, nuc_type='simultaneous', args=()):
    # returns the value of the distribution function at time $\textcolor{red}{\tilde{T}}$
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

    return dist


def Pbar_vfunc(z, vw, alpha, wall_type, nuc_type, nuc_args, npt=NPTDEFAULT):
    # Returns dimensionless velocity spectral density $\bar{P}_v$ , given array $z = qR_*$ and parameters
    # Convolves 1-bubble Fourier transform $|A(q T)|^2$ with bubble wall lifetime distribution $\nu(T \beta)$.
    nz = npt[0]
    nxi = npt[1]
    nt = npt[2]

    log10zmin = np.log10(min(z))
    log10zmax = np.log10(max(z))
    dlog10z = (log10zmax - log10zmin)/z.size

    tmin = 0.01
    tmax = 20
    log10tmin = np.log10(tmin)
    log10tmax = np.log10(tmax)

    t = np.logspace(log10tmin, log10tmax, nt)

    def qT_array(qRstar,Ttilde):
        return qRstar * Ttilde / (8. * np.pi) ** (1. / 3.) / vw

    # qT_lookup = np.logspace(np.log10(qT_array(z[0],tmin)), np.log10(qT_array(z[-1],tmax)), nz)
    qT_lookup = 10**(np.arange(log10zmin + log10tmin, log10zmax + log10tmax, dlog10z))
    A_nonlin_lookup = A_nonlin_func(qT_lookup, vw, alpha, wall_type, npt)

    A_nonlin_2d_array = np.zeros((nz, nt))
    for i in range(nz):
        A_nonlin_2d_array[i] = np.interp(qT_array(z[i],t), qT_lookup, A_nonlin_lookup)

    array2 = np.zeros(nt)
    Pbar_v = np.zeros(nz)
    Factor = 1. / (8. * np.pi) ** 2 / vw ** 6

    for s in range(nz):
        array2 = t ** 6 * nu(t, nuc_type, nuc_args) * A_nonlin_2d_array[s]
        D = np.trapz(array2, t)
        Pbar_v[s] = D * Factor
    # Straight-through for testing purposes
    # Pbar_v = Factor*np.interp(qT_array(z,1.), qT_lookup, A_nonlin_lookup)
    return Pbar_v


def Mathcal_P_v(z, vw, alpha, wall_type, nuc_type, nuc_args,  npt=NPTDEFAULT):
    # Converts spectral density P_v into velocity power spectrum
    return z**3 / (2. * np.pi**2) * Pbar_vfunc(z, vw, alpha, wall_type, nuc_type, nuc_args,  npt)


def Tilde_P_GW(y, vw, alpha, wall_type, nuc_type, nuc_args,  npt=NPTDEFAULT):
    # Returns an array of $\textcolor{red}{\tilde{P}_{GW}}$ at values given by input y array
    nz = npt[0]
    nxi = npt[1]
    nt = npt[2]
    xmax = max(y) / cs(0) * (1. + cs(0)) / 2.
    xmin = min(y) / cs(0) * (1. - cs(0)) / 2.
    xlookup = np.logspace(np.log10(xmin), np.log10(xmax), nz)
    P_vlookup = Pbar_vfunc(xlookup, vw, alpha, wall_type, nuc_type, nuc_args,  npt)

    array3 = np.zeros(nz)
    p_gw = np.zeros(nz)
    for i in range(nz):
        xplus = y[i] / cs(0) * (1. + cs(0)) / 2.
        xminus = y[i] / cs(0) * (1. - cs(0)) / 2.
        x = np.logspace(np.log10(xminus), np.log10(xplus), nz)
        # for j in range(npt):
        #     array3[j] = (x[j] - xplus) ** 2 * (x[j] - xminus) ** 2 / x[j] / (xplus + xminus - x[j]) * np.interp(x[j],
        #         xlookup, P_vlookup) * np.interp((xplus + xminus - x[j]), xlookup, P_vlookup)
        array3 = (x - xplus)**2 * (x - xminus)**2 / x / (xplus + xminus - x) * np.interp(x,
                xlookup, P_vlookup) * np.interp((xplus + xminus - x), xlookup, P_vlookup)
        p_gw_factor = ((1 - cs(0)**2)/cs(0)**2)**2 / (4*np.pi*y[i]*cs(0))
        # Corrected MBH 14.12.17 - was (1 - cs(0)) ** 2 / 4 / np.pi / y[i] / cs(0)**3
        p_gw[i] = p_gw_factor * np.trapz(array3, x)
    return p_gw


def Mathcal_P_GW(z, vw, alpha, wall_type, nuc_type, nuc_args,  npt):
    # Returns an array of $\textcolor{red}{\mathcal{P}_{GW}}$ corresponding to an array with limits kmin & kmax
    return 16. / 3. * z**3  / (2. * np.pi ** 2) * Tilde_P_GW(z, vw, alpha, wall_type, nuc_type, nuc_args,  npt)



