#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""

Calculate physical gravitational wave power spectrum $\Omega_{\rm gw}(f)$
as a function of physical frequency $f$ in Sound shell model.

Created on 10/11/21

@author: cg411, markh

"""
# import os
# os.path.dirname(os.path.realpath(__file__))

import numpy as np

import pttools.ssmtools.spectrum as ssm

import pttools.omgw0.ke_frac_approx as K
import pttools.omgw0.suppression as sup
import pttools.ssmtools.const
from . import const

#%%
# T_default = 100 # GeV
# Fgw0 = 3.57e-5 #arXiv:1910.13125v1 eqn 20

# As used in PTtools
# NXIDEFAULT = 2000 # Default number of xi points used in bubble profiles
# NTDEFAULT  = 200   # Default number of T-tilde values for bubble lifetime distribution integration
# NZDEFAULT  = 320  # Default number of z points used in the velocity convolution integrations.
# NPTDEFAULT = (NXIDEFAULT, NTDEFAULT, NZDEFAULT)
# previously used to get higher accuracy (7000,200,7000)
# SUP_METHOD_DEFAULT = 'none'


################################
# SGWB as calculated by PTtools (SSM)
################################

def J(r_star, K_frac):
    """
    pre-factor to convert power_gw_scaled to predicted spectrum
    approximation of $(H_n R_*)(H_n \tau_v)$
    updating to properly convert from flow time to source time
    See Eqn 2.8 of 	arXiv:2106.05984
    """
    sqrt_K = np.sqrt(K_frac)

    J_factor= r_star * (1 - 1/(np.sqrt(1 + 2*r_star/sqrt_K)))

    return J_factor

def get_f0(rs, T=const.T_default):
    """
    Factor required to take into account redshift of frequency scale
    See Eqn 2.13 of arXiv:2106.05984
    
    
    """
    return const.fs0_ref/rs * (T/100) * (100/100)**(1/6)

def calc_omgw0(freqs, vw, alpha, rs, T=const.T_default, npt=pttools.ssmtools.const.NPTDEFAULT, suppression=const.SUP_METHOD_DEFAULT):
    """
    For given set of thermodynamic parameters vw, alpha, rs and Tn calculates the power spectrum using
    the SSM as encoded in the PTtools module (omgwi) Eqn 2.14 in arXiv:2106.05984.

    """

    params = (vw, alpha, ssm.NucType.EXPONENTIAL, (1,))
    fp0 = get_f0(rs, T)
    z = freqs/fp0

    K_frac = K.calc_ke_frac(vw, alpha)
    omgwi = ssm.power_gw_scaled(z, params, npt=npt) # z_st_thresh=np.inf, ,npt=npt

    # entry options for power_gw_scaled
    #          z: np.ndarray,
    #        params: bubble.PHYSICAL_PARAMS_TYPE,
    #        npt=const.NPTDEFAULT,
    #        filename: str = None,
    #        skip: int = 1,
    #        method: ssm.Method = ssm.Method.E_CONSERVING,
    #        de_method: ssm.DE_Method = ssm.DE_Method.STANDARD,
    #        z_st_thresh: float = const.Z_ST_THRESH)

    if suppression == 'none':
        omgw0 = const.Fgw0 * J(rs, K_frac) * omgwi

    elif suppression == 'no_ext':

        Sup_fac = sup.get_suppression_factor(vw,alpha, method =suppression)
        omgw0 = const.Fgw0 * J(rs, K_frac) * omgwi* Sup_fac

    elif suppression == 'ext_constant':

        Sup_fac = sup.get_suppression_factor(vw, alpha, method=suppression)
        omgw0 = const.Fgw0 * J(rs, K_frac) * omgwi * Sup_fac


    return omgw0

