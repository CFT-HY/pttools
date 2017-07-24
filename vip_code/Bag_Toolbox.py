import numpy as np

aplus = (np.pi ** 2) * 106.75 / 30.
aminus=(np.pi**2)*106.75/30.
epsilon = 0.399


def w_plus(tplus):
    return 4./3.*aplus*tplus**4


def w_minus(t):
    return 4./3.*aminus*t**4


def tps_from_wps(tms, vms, vps):
    g_m_shock2 = gamma(vms)**2
    g_p_shock2 = gamma(vps)**2
    wms = w_minus(tms)
    wps = (w_minus(tms)*g_m_shock2*vms)/(g_p_shock2*vps)
    return ((3./4.)*(wps/aplus))**0.25


def gamma(v):
    return np.sqrt(1./(1-v**2))


def t_minus(tplus, vplus, vminus):
    return (3./4.*(w_plus(tplus)*vplus*(gamma(vplus))**2)/(aminus*vminus*(gamma(vminus))**2))**0.25


def t_plus(al_p):
    return (epsilon/(aplus*al_p))**0.25


def cs(t):
    return 1./np.sqrt(3.) * np.ones_like(t)  # 0.577

# reverse this
def cs2(t):
    return cs(t)**2


def mu(vw, v):
    return (vw-v)/(1.-vw*v)



