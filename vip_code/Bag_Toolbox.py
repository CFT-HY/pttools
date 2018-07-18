import numpy as np
import Mechanics_Toolbox as Mech
import sys


def set_params(name, new_value=None):
    global aplus, aminus, epsilonplus, epsilonminus
    if name == 'default':
        aplus = (np.pi ** 2) * 106.75 / 30. # As if all ultrarelativistic
        aminus = (np.pi**2)*86.25/30. # As if h, W, Z, t are non-rel
        epsilonplus = 0.1
        epsilonminus = 0
    elif name == 'aplus':
        aplus = new_value
    elif name == 'aminus':
        aminus = new_value
    elif name == 'epsilonplus':
        epsilonplus = new_value
    elif name == 'epsilonminus':
        epsilonminus = new_value
    else:
        sys.exit('set_params_bag: params name not recognised')
    print_params()
    return


def print_params():
    print('aplus = ',aplus)
    print('aminus = ', aminus)
    print('epsilonplus = ', epsilonplus)
    print('epsilonminus = ', epsilonminus)
    return


def call_params():
    return np.array([aplus, aminus, epsilonplus, epsilonminus])


def w_plus(tplus):
    return 4./3.*aplus*tplus**4


def w_minus(t):
    return 4./3.*aminus*t**4


def tps_from_wps(tms, vms, vps):
    g_m_shock2 = Mech.gamma(vms)**2
    g_p_shock2 = Mech.gamma(vps)**2
    wms = w_plus(tms)
    wps = (wms*g_m_shock2*vms)/(g_p_shock2*vps)
    return ((3./4.)*(wps/aplus))**0.25


def t_minus(tplus, vplus, vminus):
    return (3./4.*(w_plus(tplus)*vplus*(Mech.gamma(vplus))**2)/(aminus*vminus*(Mech.gamma(vminus))**2))**0.25


def t_plus(al_p):
    return (epsilonplus/(aplus*al_p))**0.25


def cs(t):
    return np.sqrt(cs2(t))  # 0.577


def cs2(t):
    return np.ones_like(t) * (1./3.)


def cs2_w(w):
    return 1./3.


def cs_w(w, dummy):
    return np.sqrt(cs2_w(w))


def p_w(w, phi=None):
    if phi is None:
        e = epsilonminus
    else:
        e = epsilonplus
    return 0.25 * w - e


def epsilon_w(w, phi=None):
    if phi is None:
        return call_params()[3]
    else:
        return call_params()[2]

print('Bag_Toolbox: setting default params')
set_params('default')

