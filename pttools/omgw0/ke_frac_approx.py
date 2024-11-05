#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:37:36 2019

@author: chloeg
"""
import numpy as np


def  k_a( xi_w, alpha_n ):
    """
    Small wall speeds xi_w <<cs
    """
    return xi_w**(6/5) * 6.9 * alpha_n/(1.36 - 0.037 * np.sqrt(alpha_n) + alpha_n)


def k_b(alpha_n ):
    """
    transition from subsonic to  supersonic deflagrations
    xi_w = cs
    """

    return alpha_n ** (2/5)/ (0.017 + ( 0.997 + alpha_n)**(2/5))


def k_c( alpha_n):
    """
    For Jouget detonations xi_w = xi_j
    """
    return np.sqrt(alpha_n) / (0.135 + ( np.sqrt( 0.98 + alpha_n)))

def xi_j(alpha_n):
    """
    Jouget detonation speed
    """
    return (np.sqrt(2/3 * alpha_n + alpha_n**2 ) + np.sqrt(1/3))  /  (1 + alpha_n)


def k_d (alpha_n):
    """
    xi_w => 1 v. large wall speed
    """
    return alpha_n/ (0.73 + 0.083 * np.sqrt(alpha_n) + alpha_n)

def delta_k (alpha_n):

    return  -0.9 * np.log( np.sqrt(alpha_n)/ (1 + np.sqrt(alpha_n)))



def calc_ke_frac(xi_w, alpha_n):

    cs = 1/np.sqrt(3)
    def  k_a( xi_w, alpha_n ):
        """
        Small wall speeds xi_w <<cs
        """
        return xi_w**(6/5) * 6.9 * alpha_n/(1.36 - 0.037 * np.sqrt(alpha_n) + alpha_n)


    def k_b(alpha_n ):
        """
        transition from subsonic to  supersonic deflagrations
        xi_w = cs
        """

        return alpha_n ** (2/5)/ (0.017 + ( 0.997 + alpha_n)**(2/5))


    def k_c( alpha_n):
        """
        For Jouget detonations xi_w = xi_j
        """
        return np.sqrt(alpha_n) / (0.135 +  np.sqrt( 0.98 + alpha_n))

    def xi_j(alpha_n):
        """
        Jouget detonation speed
        """
        return (np.sqrt(2/3 * alpha_n + alpha_n**2 ) + np.sqrt(1/3))  /  (1 + alpha_n)


    def k_d (alpha_n):
        """
        xi_w => 1 v. large wall speed
        """
        return alpha_n/ (0.73 + 0.083 * np.sqrt(alpha_n) + alpha_n)

    def delta_k (alpha_n):

        return  -0.9 * np.log( np.sqrt(alpha_n)/ (1 + np.sqrt(alpha_n))) # natural log


    xi_j = xi_j(alpha_n)

    if xi_w < cs :

        k_a =k_a(xi_w, alpha_n)
        k_b = k_b(alpha_n)


        k =cs**(11/5) * k_a * k_b /  ((cs**(11/5) - xi_w**(11/5)) * k_b + xi_w * cs**(6/5) * k_a)

    elif xi_w == cs:
        k = k_b(alpha_n)

    elif cs< xi_w<xi_j :
        delta_k =  delta_k(alpha_n)
        k_b = k_b(alpha_n)
        k_c = k_c( alpha_n)


        k = k_b + (xi_w - cs)* delta_k + ((xi_w - cs)/(xi_j- cs))**3 * ( k_c - k_b -( xi_j - cs) * delta_k)

    elif xi_w == xi_j:
        k = k_c(alpha_n)

    elif xi_w > xi_j:
        k_c = k_c(alpha_n)
        k_d = k_d(alpha_n)

        k = ((xi_j - 1)**3  * xi_j**(5/2) * xi_w **(-5/2) * k_c * k_d ) / ( ((xi_j -1)**3 - (xi_w-1)**3) * xi_j**(5/2) * k_c + ( xi_w -1 )**3 * k_d)

    elif xi_w> 0.85 :
        k = k_d(alpha_n)
    else:
        pass







    return k * alpha_n /( 1+ alpha_n)
