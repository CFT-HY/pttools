#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed 4/8/2021

@author: cg411

Removes the unpublished hybrid data from the suppression data set
"""
import numpy as np
import pttools.bubble as bbl

sim_data = np.loadtxt("./../suppresion_2.txt", skiprows=1)

#Removing hybrids from simulation data
#order of entries in txt file
#vw alph suppress sim_omgw exp_omgw exp_ubarf
vw_no_hybrid = []
al_no_hybrid = []
sup_sim_no_hybrids = []
sim_omgw_no_hybrids= []
exp_omgw_no_hybrids = []
exp_Ubarf_no_hybrids = []


# speed of sound
cs = 1/np.sqrt(3)


for i, vw in enumerate(sim_data[:,0]):
    print(vw)
    alpha =sim_data[i,1]

    if vw >cs  and vw < bbl.min_speed_deton(alpha):
        print("ignoring hybrid")
        pass
    else:
        vw_no_hybrid.append(sim_data[i,0])
        al_no_hybrid.append(sim_data[i,1])
        sup_sim_no_hybrids .append(sim_data[i,2])
        sim_omgw_no_hybrids.append(sim_data[i,3])
        exp_omgw_no_hybrids.append(sim_data[i,4])
        exp_Ubarf_no_hybrids.append(sim_data[i,5])



with open('suppression_no_hybrids.txt', 'w') as f:
    f.write("vw" + " " + "alph" + " " + "suppress" + " " + "sim_omgw" + " " + "exp_omgw" + "exp_ubarf" )
    f.write('\n')


    for i in range(0,len(vw_no_hybrid)):
        line = str(vw_no_hybrid[i] )+ " "+ str(al_no_hybrid[i])+ " " + str(sup_sim_no_hybrids[i]) + " "+ str(sim_omgw_no_hybrids[i]) + " "+ str(exp_omgw_no_hybrids[i]) + " "+ str(exp_Ubarf_no_hybrids[i])
        f.write(line)
        f.write('\n')
print('simulation suppression data without hybrids file created')

