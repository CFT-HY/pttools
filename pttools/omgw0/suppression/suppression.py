#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolate or extrapolate kinetic suppression data in the sound shell model.

Created on wed 4 Aug  2021

@author: chloeg
"""
import os
import numpy as np
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline

os.path.dirname(os.path.realpath(__file__))
suppression_path = os.path.join(os.path.dirname(__file__), 'suppression_ssm_data/suppression_no_hybrids_ssm.npz')

# :TODO why is there a difference in the low alpha low vw region between hybrids and no hybrids data set?

ssm_sup_data = np.load(suppression_path)
vws_sim = ssm_sup_data['vw_sim']
alphas_sim = ssm_sup_data['alpha_sim']
ssm_sup = ssm_sup_data['sup_ssm']

METHOD_DEFAULT = 'no_ext'

"""
To improve the extrapolation of the suppression factor when later using gridata, first extend the 
low vw and low alpha region as follows
"""
# alpha values in suppression dataset for vw = 0.24
ssm_sup_vw_0_24_alphas = [0.05000, 0.07300, 0.11000, 0.16000, 0.23000, 0.34000]
# Suppression values for vw = 0.24
ssm_sup_vw_0_24 = [0.01675, 0.01218, 0.00696, 0.00251, 0.00054, 0.00007]

spl = InterpolatedUnivariateSpline(ssm_sup_vw_0_24_alphas, ssm_sup_vw_0_24, k=1, ext=0)

ssm_sup_vw_0_24_alphas_ext = [0.00500, 0.05000, 0.07300, 0.11000, 0.16000, 0.23000, 0.34000]
ssm_sup_vw_0_24_ext = spl(ssm_sup_vw_0_24_alphas_ext)

# create the extrapolated dataset
vws_sim_ext = np.concatenate(([0.24], vws_sim))
alphas_sim_ext = np.concatenate(([ssm_sup_vw_0_24_alphas_ext[0]], alphas_sim))
ssm_sup_ext = np.concatenate(([ssm_sup_vw_0_24_ext[0]], ssm_sup))
# %%
def alpha_n_max_approx(vw):
    """
    Approximate form of alpha_n_max function
    """
    alpha_n_max = (1/3)*(1 + 3*vw**2)/(1 - vw**2)
    return alpha_n_max

def alpha_n_max(vw):
    #  vw ,al
    # [0.24000,0.34000]
    # [0.44000, 0.50000]
    # [0.56000,0.67000]
    if vw < 0.44:
        m1 = (0.5 - 0.34) /(0.44 - 0.24)#dal/dvw
        c1 = 0.34 - m1 * 0.24
        al_max = m1 * vw + c1
    elif vw >= 0.44:
        m2 = (0.67 - 0.5)/(0.56 - 0.44)
        c2 = 0.67000 - m2 * 0.56000
        al_max = m2 * vw + c2
    return al_max


def get_suppression_factor(vw,alpha, method =METHOD_DEFAULT):
    """
    current simulation data bounds are
    0.24<vw<0.96
    0.05<alpha<0.67
    methods options :
    - 'no_ext' = returns NaN outside of data region
    - 'ext_constant' = extends the boundaries with a constant value
    - 'ext_linear_Ubarf' = :TODO extend with linear Ubarf
    """
    if alpha > alpha_n_max_approx(vw):
        print('vw alpha combo unphysical')
        supp_factor = np.nan
    else :
        vv_n, aa_n = np.meshgrid(vw, alpha)
        if method == 'no_ext':
            supp_factor = interpolate.griddata((vws_sim, alphas_sim), ssm_sup, (vv_n, aa_n), method='linear')[0]

        elif method == "ext_constant":

            supp_factor = interpolate.griddata((vws_sim_ext, alphas_sim_ext), ssm_sup_ext, (vv_n, aa_n), method='linear')[0]

        else:
            supp_factor = np.nan
            print("Invalid option given , options are method = 'no_ext' or 'ext_constant'")


    return supp_factor


def get_suppression_factor_with_hybrids(vws, alphas):
    """
    0.24<vw<0.96
    0.05<alpha<0.67
    """
    vv_n, aa_n = np.meshgrid(vws, alphas)
    suppression_path_hybrids = os.path.join(os.path.dirname(__file__), 'suppression_2_ssm.npz')
    ssm_sup_data_hybrids = np.load(suppression_path_hybrids)
    vws_sim_hybrids = ssm_sup_data_hybrids['vw_sim']
    alphas_sim_hybrids = ssm_sup_data_hybrids['alpha_sim']
    ssm_sup_hybrids = ssm_sup_data_hybrids['sup_ssm']


    supp_factor = interpolate.griddata((vws_sim_hybrids, alphas_sim_hybrids), ssm_sup_hybrids, (vv_n, aa_n), method='linear')

    return supp_factor
