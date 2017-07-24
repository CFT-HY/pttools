# Toolbox for relativistic equations and equations that are duplicated across main program and EoS modules
import numpy as np


def gamma(v):
    return np.sqrt(1./(1-v**2))


def mu(vw, v):
    return (vw-v)/(1.-vw*v)


def v_minus(vp, al_p):
    if vp < 1./np.sqrt(3.):
        vm = (((1+al_p)/2.)*vp+((1.-3.*al_p)/(6*vp)))-np.sqrt(((((1.+al_p)*vp)/2.)+((1.-3.*al_p)/(6.*vp)))**2-(1./3.))
    elif vp > 1./np.sqrt(3.):
        vm = (((1+al_p)/2.)*vp+((1.-3.*al_p)/(6*vp)))+np.sqrt(((((1.+al_p)*vp)/2.)+((1.-3.*al_p)/(6.*vp)))**2-(1./3.))
    # Edge case?
    return vm


def v_plus(vm, al_p):
    if vm <= 1./np.sqrt(3.):
        vp = (1./(1.+al_p))*(((vm/2.)+(1./(6.*vm)))-np.sqrt(((vm/2.)+(1./(6.*vm)))**2+(al_p**2)+((2./3.)*al_p)-(1./3.)))
    elif vm > 1./np.sqrt(3.):
        vp = (1./(1.+al_p))*(((vm/2.)+(1./(6.*vm)))+np.sqrt(((vm/2.)+(1./(6.*vm)))**2+(al_p**2)+((2./3.)*al_p)-(1./3.)))
    return vp
