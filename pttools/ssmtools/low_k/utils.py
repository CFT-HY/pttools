import numpy as np

#: Lorenzo eq. 3.7
# In the Sound Shell Model this is an integral instad of a constant
Iv: float = 1/(32*np.pi**2)

#: Another constant
Jv: float = 1/(128*np.pi**4)


def parse_params_gw(params):
    """
    Input parameters for gravitational wave power spectrum:
        cs = params_gw[0]       scalar  (required) [0 < cs < 1/sqrt(3)]
        tau_star = params_gw[1] scalar  (required) [tau_star = eta_star/Lf]
        tau_end = params_gw[2]  scalar  (required) [tau_end = eta_end/Lf]
    """
    cs = params[0]
    tau_star = params[1]
    tau_end = params[2]
    if not (0 < cs < 1/np.sqrt(3)):
        raise ValueError("Sound speed cs must be in the range (0, 1/sqrt(3)).")
    return cs, tau_star, tau_end


def rho(z, x, cs):
    r"""Lorenzo eq. 2.37"""
    xp = 0.5*z * (1+cs)/cs
    xm = 0.5*z * (1-cs)/cs
    return ((xp+xm-x)**2 - (x-z)**2)**2 * ((xp+xm-x)**2 - (x+z)**2)**2 /16/x/(xp+xm-x)/z**2


def U(x, a):
    r"""Lorenzo eq. 3.13 stuff in the brackets"""
    return (1-x**a)/a
