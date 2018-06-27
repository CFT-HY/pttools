import numpy as np


def cs2_w(w):
    a_0 = 0.1
    return (1./3.)*(1-(a_0/(1+w)))


def cs_w(w):
    return np.sqrt(cs2_w(w))


def p_w(w, e):
    return 0.25*w - e


