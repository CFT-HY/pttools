"""Sound shell model functions"""

import logging

import numpy as np
from scipy.optimize import fsolve

import pttools.type_hints as th
from pttools import bubble
from . import calculators
from . import const

logger = logging.getLogger(__name__)


def A2_ssm_func(
        z: np.ndarray,
        vw,
        alpha: float,
        npt: const.NPT_TYPE = const.NPTDEFAULT,
        method: str = 'e_conserving',
        de_method: str = 'standard',
        z_st_thresh: float = const.Z_ST_THRESH):
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
    if method == 'e_conserving':
        # This is the correct method (as of 12.18)
        A2 = A2_e_conserving(z, vw, alpha, npt, 'A2_only', de_method, z_st_thresh)
    elif method == 'f_only':
        logger.debug("f_only method, multiplying (f\')^2 by 2")
        f = f_ssm_func(z, vw, alpha, npt)
        df_dz = np.gradient(f) / np.gradient(z)
        A2 = 0.25 * (df_dz ** 2)
        A2 = A2 * 2
    elif method == 'with_g':
        logger.debug("With_g method")
        f = f_ssm_func(z, vw, alpha, npt)
        df_dz = np.gradient(f) / np.gradient(z)
        g = (z * df_dz + 2. * f)
        dg_dz = np.gradient(g) / np.gradient(z)
        A2 = 0.25 * (df_dz ** 2)
        A2 = A2 + 0.25 * (dg_dz ** 2 / (const.CS0 * z) ** 2)
    else:
        logger.warning("Method not known, should be [e_conserving | f_only | with_g]. Defaulting to e_conserving.")
        A2 = A2_e_conserving(z, vw, alpha, npt)

    return A2


def A2_e_conserving(
        z: np.ndarray,
        vw,
        alpha_n: float,
        npt: const.NPT_TYPE = const.NPTDEFAULT,
        ret_vals: str = "A2_only",
        de_method: str = "standard",
        z_st_thresh: float = const.Z_ST_THRESH):
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
    v_ip, w_ip, xi = bubble.fluid_shell(vw, alpha_n, nxi)

    #    f = np.zeros_like(z)
    #    for j in range(f.size):
    #        f[j] = (4.*np.pi/z[j]) * calculators.sin_transform(z[j], xi, v_ip, z_st_thresh)
    f = (4. * np.pi / z) * calculators.sin_transform(z, xi, v_ip, z_st_thresh)

    v_ft = np.gradient(f) / np.gradient(z)

    # Now get and resample lam = de/w
    if de_method == 'alternate':
        lam_orig = bubble.de_from_w_new(v_ip, w_ip, xi, vw, alpha_n) / w_ip[-1]
    else:
        lam_orig = bubble.de_from_w(w_ip, xi, vw, alpha_n) / w_ip[-1]

    lam_orig += w_ip * v_ip * v_ip / w_ip[-1]  # This doesn't make much difference at small alpha
    xi_re, lam_re = calculators.resample_uniform_xi(xi, lam_orig, nxi)

    #    lam_re = np.interp(xi_re,xi,lam_orig)
    #    lam_ft = np.zeros_like(z)
    #    for j in range(lam_ft.size):
    #        lam_ft[j] = (4.*np.pi/z[j]) * calculators.sin_transform(z[j], xi_re, xi_re*lam_re,
    #              z_st_thresh=max(z)) # Need to fix problem with ST of lam for detonations
    lam_ft = (4. * np.pi / z) * calculators.sin_transform(z, xi_re, xi_re * lam_re, z_st_thresh)

    A2 = 0.25 * (v_ft ** 2 + (const.CS0 * lam_ft) ** 2)

    if ret_vals == 'A2_only':
        return A2
    return A2, v_ft ** 2 / 2, (const.CS0 * lam_ft) ** 2 / 2


def A2_e_conserving_file(
        z: np.ndarray,
        filename: str,
        alpha: float,
        skip: int = 1,
        npt: const.NPT_TYPE = const.NPTDEFAULT,
        z_st_thresh: float = const.Z_ST_THRESH):
    """
     Returns the value of $|A(z)|^2$, where |Plane wave amplitude|^2 = T^3 |A(z)|^2,
     calculated from file, outout by ``spherical-hydro-code``.
     z is an array of scaled wavenumbers $z = kR_*$.
     Uses method respecting energy conservation, although only accurate to
     linear order, meaning that there is an apparent $z^0$ piece at very low $z$.
    """
    logger.debug(f"loading v(xi), e(xi) from {filename}")
    try:
        with open(filename) as f:
            t = float(f.readline())
        r, v_all, e_all = np.loadtxt(filename, usecols=(0, 1, 4), unpack=True, skiprows=skip)
    except IOError as error:
        raise IOError(f"Error loading file: \"{filename}\"") from error

    xi_all = r / t
    wh_xi_lt1 = np.where(xi_all < 1.)
    logger.debug(f"Interpolating v(xi), e(xi) from {wh_xi_lt1[0]} to {npt[0]} points")

    xi_lt1 = np.linspace(0., 1., npt[0])
    v_xi_lt1 = np.interp(xi_lt1, xi_all, v_all)
    e_xi_lt1 = np.interp(xi_lt1, xi_all, e_all)
    #    f = np.zeros_like(z)
    #    for j in range(f.size):
    #        f[j] = (4.*np.pi/z[j]) * calculators.sin_transform(z[j], xi_lt1, v_xi_lt1)
    f = (4. * np.pi / z) * calculators.sin_transform(z, xi_lt1, v_xi_lt1)

    v_ft = np.gradient(f) / np.gradient(z)
    e_n = e_xi_lt1[-1]

    def fun(x):
        return x - bubble.get_w(e_n, 0, alpha * (0.75 * x))

    w_n0 = bubble.get_w(e_n, 0, alpha * e_n)  # Correct only in Bag, probably good enough
    w_n = fsolve(fun, w_n0)[0]  # fsolve returns array, want float
    lam = (e_xi_lt1 - e_n) / w_n
    logger.debug(f"Initial guess w_n0: {w_n0}, final {w_n}")

    #    lam_ft = np.zeros_like(z)
    #    for j in range(lam_ft.size):
    #        lam_ft[j] = (4.*np.pi/z[j]) * calculators.sin_transform(z[j], xi_lt1, xi_lt1*lam,
    #              z_st_thresh=max(z)) # Need to fix problem with ST of lam for detonations
    lam_ft = (4. * np.pi / z) * \
        calculators.sin_transform(z, xi_lt1, xi_lt1 * lam, z_st_thresh)

    return 0.25 * (v_ft ** 2 + (const.CS0 * lam_ft) ** 2)


def f_ssm_func(
        z: th.FLOAT_OR_ARR,
        vw,
        alpha_n,
        npt: const.NPT_TYPE = const.NPTDEFAULT,
        z_st_thresh: float = const.Z_ST_THRESH):
    """
     3D FT of radial fluid velocity v(r) from Sound Shell Model fluid profile.
     z is array of scaled wavenumbers z = kR*
    """
    nxi = npt[0]
    v_ip, _, xi = bubble.fluid_shell(vw, alpha_n, nxi)

    # f_ssm = np.zeros_like(z)
    # for j in range(f_ssm.size):
    #    f_ssm[j] = (4.*np.pi/z[j]) * calculators.sin_transform(z[j], xi, v_ip)
    f_ssm = (4.*np.pi/z) * calculators.sin_transform(z, xi, v_ip, z_st_thresh)

    return f_ssm


def lam_ssm_func(
        z,
        vw,
        alpha_n,
        npt: const.NPT_TYPE = const.NPTDEFAULT,
        de_method: str = "standard",
        z_st_thresh: float = const.Z_ST_THRESH):
    """
     3D FT of radial energy perturbation from Sound Shell Model fluid profile
     z is array of scaled wavenumbers z = kR*
    """
    nxi = npt[0]
#    xi_re = np.linspace(0,1-1/nxi,nxi) # need to resample for lam = de/w
    v_ip, w_ip, xi = bubble.fluid_shell(vw, alpha_n, nxi)

    if de_method == 'alternate':
        lam_orig = bubble.de_from_w_new(v_ip, w_ip, xi, vw, alpha_n) / w_ip[-1]
    else:
        lam_orig = bubble.de_from_w(w_ip, xi, vw, alpha_n) / w_ip[-1]
    xi_re, lam_re = calculators.resample_uniform_xi(xi, lam_orig, nxi)

    # lam_ft = np.zeros_like(z)
    # for j in range(lam_ft.size):
    #    lam_ft[j] = (4.*np.pi/z[j]) * calculators.sin_transform(z[j], xi_re, xi_re*lam_re,
    #          z_st_thresh=max(z)) # Need to fix problem with ST of lam for detonations

    lam_ft = (4.*np.pi/z) * calculators.sin_transform(z, xi_re, xi_re*lam_re, z_st_thresh)

    return lam_ft


def g_ssm_func(z: np.ndarray, vw, alpha, npt: const.NPT_TYPE = const.NPTDEFAULT) -> np.ndarray:
    r"""
     3D FT of radial fluid acceleration \dot{v}(r) from Sound Shell Model fluid profile.
     z is array of scaled wavenumbers z = kR*.
    """
    f_ssm = f_ssm_func(z, vw, alpha, npt)
    df_ssmdz = np.gradient(f_ssm) / np.gradient(z)
    g_ssm = (z * df_ssmdz + 2. * f_ssm)
    return g_ssm


def f_file(
        z_arr: np.ndarray,
        t,
        filename: str,
        skip: int = 0,
        npt: const.NPT_TYPE = const.NPTDEFAULT,
        z_st_thresh: float = const.Z_ST_THRESH):
    """
     3D FT of radial fluid velocity v(r) from file.
     z is array of scaled wavenumbers z = kR*
    """
    logger.debug(f"Loading v(xi) from {filename} at time {t}")
    try:
        r, v_all = np.loadtxt(filename, usecols=(0, 1), unpack=True, skiprows=skip)
    except IOError as error:
        raise IOError(f"Error loading file: \"{filename}\"") from error

    xi_all = r / t
    wh_xi_lt1 = np.where(xi_all < 1.)
    logger.debug(f"Interpolating v(xi) from {len(wh_xi_lt1[0])} to {npt[0]} points")
    #    xi_lt1 = np.linspace(0.,1.,npt[0])
    #    v_xi_lt1 = np.interp(xi_lt1,xi_all,v_all)
    xi_lt1, v_xi_lt1 = calculators.resample_uniform_xi(xi_all, v_all, npt[0])
    #    f = np.zeros_like(z_arr)
    #    for n, z in enumerate(z_arr):
    #        f[n] = (4*np.pi/z)*calculators.sin_transform(z, xi_lt1, v_xi_lt1, z_st_thresh)
    f = (4 * np.pi / z_arr) * calculators.sin_transform(z_arr, xi_lt1, v_xi_lt1, z_st_thresh)

    return f


def g_file(z: np.ndarray, t, filename: str, skip: int = 0) -> np.ndarray:
    r"""
     3D FT of radial fluid acceleration \dot{v}(r) from file
     z is array of scaled wavenumbers z = kR*
    """
    f = f_file(z, t, filename, skip)
    df_dz = np.gradient(f) / np.gradient(z)
    g = (z * df_dz + 2. * f)
    return g
