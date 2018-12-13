#!/usr/bin/env python
#
# Programme for calculating and plotting scaling velocity profile around expanding Higgs-phase bubble.
# See Espinosa et al 2010
#
# Mark Hindmarsh 2015-18
# with Mudhahir Al-Ajmi


import sys
#import numpy as np
#import matplotlib as mpl
#import matplotlib.pyplot as plt
import bubble as b




def main():
    if not len(sys.argv) in [3, 4]:
        sys.stderr.write('usage: %s <v_wall> <alpha_n> [save_string]\n' % sys.argv[0])
        sys.exit(1)
            
    v_wall = float(sys.argv[1])
    alpha_n = float(sys.argv[2])

    if len(sys.argv) < 4:
        save_string = None
    else:
        save_string = sys.argv[3]

    b.plot_fluid_shell(v_wall, alpha_n, save_string)



if __name__ == '__main__':
    main()
