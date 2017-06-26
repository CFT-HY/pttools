import numpy as np


def w_plus(tplus, aplus):
    return 4./3.*aplus*tplus**4


def gamma(v):
    return np.sqrt(1./(1-v**2))


def t_minus(tplus, vplus, vminus, aplus, aminus):
    return (3./4.*(w_plus(tplus, aplus)*vplus*(gamma(vplus))**2)/(aminus*vminus*(gamma(vminus))**2))**0.25


