#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolate or extrapolate kinetic suppression data in the sound shell model.

Created on wed 4 Aug  2021

@author: cg411
"""
import os
import numpy as np
import math
from scipy import interpolate

os.path.dirname(os.path.realpath(__file__))
suppression_path = os.path.join(os.path.dirname(__file__), 'suppression_ssm_data/suppression_no_hybrids_ssm.npz')

# :TODO why is there a difference in the low alpha low vw region between hybrids and no hybrids data set?

ssm_sup_data = np.load(suppression_path)
vws_sim = ssm_sup_data['vw_sim']
alphas_sim = ssm_sup_data['alpha_sim']
ssm_sup = ssm_sup_data['sup_ssm']

METHOD_DEFAULT = 'no_ext'

# %%
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
    vv_n, aa_n = np.meshgrid(vw, alpha)

    if method == 'no_ext':
        supp_factor = interpolate.griddata((vws_sim, alphas_sim), ssm_sup, (vv_n, aa_n), method='linear')

    elif method == "ext_constant":

        supp_factor = interpolate.griddata((vws_sim, alphas_sim), ssm_sup, (vv_n, aa_n ), method='linear')

        if math.isnan(supp_factor) == True:
            if alpha < 0.1:
                fill_val = 0.2
                print(fill_val)
            elif alpha >= 0.1:
                if vw < 0.5:
                    fill_val = 0.004
                elif vw > 0.9:
                    fill_val = 1
                print(fill_val, vw, alpha)

            supp_factor = np.nan_to_num(supp_factor, nan=fill_val)
            print('WARNING: using extrapolated value, edge values extended')
    else:
        print("Invalid option given , options are...")


    return supp_factor[0]

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