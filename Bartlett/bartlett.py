# -*- coding:utf-8 -*-
import math
import os
import sys

import numpy as np
import scipy.constants as sconst
from scipy.stats import norm

sys.path.append(os.path.dirname(__file__) + os.sep + '../')

class BartlettAoA():
    """Bartlett Algorithm Searching AoA space
    """
    def __init__(self):
        filepath =  "Bartlett/atn_loc.csv"
        antenna_loc = np.loadtxt(filepath, delimiter=",")    # 3x16
        x,y = antenna_loc[0,:],antenna_loc[1,:]
        antenna_num = 16                                     # the number of the antenna array element
        atn_polar   = np.zeros((antenna_num,2))              # Polar Coordinate
        for i in range(antenna_num):
            atn_polar[i,0] = math.sqrt(x[i] * x[i] + y[i] * y[i])
            atn_polar[i,1] = math.atan2(y[i], x[i])
        self.theory_phase = self.get_theory_phase(atn_polar)


    def get_theory_phase(self, atn_polar):
        """get theory phase, return (360x90)x16 array
        """
        a_step = 1 * 360
        e_step = 1 * 90
        spacealpha = np.linspace(0, np.pi * 2, a_step)  # 0-2pi
        spacebeta = np.linspace(0, np.pi / 2, e_step)   # 0-pi/2

        alpha,beta = np.meshgrid(spacealpha, spacebeta)
        alpha, beta = alpha.flatten(), beta.flatten()   #alpha[0,1,..0,1..], beta [0,0,..1,1..]
        theta_k = atn_polar[:,1].reshape(16,1)          #alpha 1x(360x90)
        r = atn_polar[:,0].reshape(16,1)
        lamda = sconst.c / 920e6
        theta_t = -2 * math.pi / lamda * r * np.cos(alpha - theta_k) * np.cos(beta)
        return theta_t.T


    def get_aoa_heatmap(self, phase_m):
        """got aoa heatmap
        """
        delta_phase = self.theory_phase - phase_m.reshape(1,16)     #(720x180)x16 - 1x16
        cosd = (np.cos(delta_phase)).sum(1)
        sind = (np.sin(delta_phase)).sum(1)
        p = np.sqrt(cosd * cosd + sind * sind) / 16
        p = p.reshape(90,360)
        return p


if __name__ == '__main__':
    pass
