"""
Intraparticle heat conduction within woody biomass particle at fast pyrolysis
conditions.
"""

import numpy as np
import matplotlib.pyplot as py
from funcHeatCond import hc2
from funcOther import vol, Tvol

# Parameters
# ------------------------------------------------------------------------------

Gb = 0.54       # basic specific gravity, Wood Handbook Table 4-7, (-)
k = 0.12        # thermal conductivity, W/mK
x = 0           # moisture content, %
h = 350         # heat transfer coefficient, W/m^2*K
Ti = 293        # initial particle temp, K
Tinf = 773      # ambient temp, K

sa = 1.716e-5   # surface area of particle, m^2
v = 2.877e-9    # volume of particle, m^3

# number of nodes from center of particle (m=0) to surface (m)
m = 1000

# time vector from 0 to max time
tmax = 8.0                      # max time, s
nt = 1000                       # number of time steps
dt = tmax/nt                    # time step, s
t = np.arange(0, tmax+dt, dt)   # time vector, s

# 1D Transient Heat Conduction
# ------------------------------------------------------------------------------

# surface area, volume, and dsv for each particle size
ds = (sa/np.pi)**(1/2)     # surface area equivalent sphere diameter, m
dv = (6/np.pi*v)**(1/3)    # volume equivalent sphere diameter, m
dsv = (dv**3)/(ds**2)      # surface area to volume sphere diameter, m

# calculate temp profiles as Tsv in each particle as based on Dsv where
# row = time step, column = center to surface temperature
# determine center volume and shell volumes as v
# calculate volume averaged temperature of particle at each time step as Tsv_v
T = hc2(dsv, x, k, Gb, h, Ti, Tinf, 2, m, t)    # temperature array, K

rad = np.linspace(0, dsv/2, m)      # radius vector from center to surface, m
v = vol(rad)                        # volumes in particle vector, m^3
Tv = Tvol(T, v)                     # volume average temperatures vector, K

# Plot Results
# ------------------------------------------------------------------------------

py.ion()
py.close('all')

def despine():
    ax = py.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.tick_params(axis='both', bottom='off', top='off', left='off', right='off')

py.figure(1)
py.plot(t, T[:,0], lw=2, label='center')
py.plot(t, T[:,-1], lw=2, label='surface')
py.plot(t, Tv, lw=2, label='volume')
py.xlabel('Time (s)')
py.ylabel('Temperature (K)')
py.legend(loc='best', numpoints=1)
py.grid()
despine()

