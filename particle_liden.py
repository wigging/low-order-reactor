"""
Plot the 1D intraparticle transient heat conduction and pyrolysis kinetics for a
solid sphere, cylinder, or slab shape.

Assumptions:
Convection boundary condition at surface.
Moisture content in the biomass particle as free water.
Heat transfer via radiation assumed to be negligable.
Particle does not shrink or expand in size during pyrolysis.
"""

import numpy as np
import matplotlib.pyplot as py
from funcHeatCond import hc
from funcKinetics import liden

# Parameters
#-------------------------------------------------------------------------------

rhow = 540      # density of wood, kg/m^3
kw = 0.12       # thermal conductivity of wood, W/m*K
kc = 0.08       # thermal conductivity of char, W/m*K

h = 350         # heat transfer coefficient, W/m^2*K
H = -220000     # heat of reaction, J/kg where (-) exothermic, (+) endothermic
Ti = 293        # initial particle temp, K
Tinf = 773      # ambient temp, K

sa = 1.716e-5   # surface area of particle for DF = 5.4 mm pine, m^2
v = 2.877e-9    # volume of particle for DF = 5.4 mm pine, m^3

ds = (sa/np.pi)**(1/2)     # surface area equivalent sphere diameter, m
dv = (6/np.pi*v)**(1/3)    # volume equivalent sphere diameter, m
dsv = (dv**3)/(ds**2)      # surface area to volume sphere diameter, m

# Shape factor, time, and node (radius point) vectors
#-------------------------------------------------------------------------------

b = 2           # run model as a cylinder (b = 1) or as a sphere (b = 2)

dt = 0.01                               # time step as delta t, s
tmax = 8.0                              # max time, s
t = np.linspace(0, tmax, num=tmax/dt)   # time vector, s
nt = len(t)                             # total number of time steps

nr = 999    # number or radius steps
r = dsv/2   # radius of particle, m
dr = r/nr   # radius step, delta r
m = nr+1    # nodes from center m=0 to surface m=steps+1

# Temperature and Density arrays, Mass Fraction vector
#-------------------------------------------------------------------------------

# temperture array
# rows = time step, columns = node points from center to surface node
T = np.zeros((nt, m))       # create array to store temperatures
T[0] = Ti                   # initial temperature at all nodes

# density array
# rows = time step, columns = node points from center to surface
wood = np.zeros((nt, m))        # create array for wood concentrations
gas1 = np.zeros((nt, m))        # create array for gas1 concentrations
tar = np.zeros((nt, m))         # create array for tar concentrations
gaschar = np.zeros((nt, m))     # create array for gaschar concentrations
gas = np.zeros((nt, m))         # create array for gas concentrations
char = np.zeros((nt, m))        # create array for char concentrations

wood[0] = rhow      # initial wood density at all nodes

# Initial thermal properties 
#-------------------------------------------------------------------------------

cpw = 1112.0 + 4.85 * (T[0] - 273.15)    # wood heat capacity, J/(kg*K) 
cpc = 1003.2 + 2.09 * (T[0] - 273.15)    # char heat capacity, J/(kg*K)

Yw = wood[0]/rhow       # wood fraction, Yw=1 all wood, Yw=0 all char

cp_bar = Yw*cpw + (1-Yw)*cpc    # effective heat capacity
k_bar = Yw*kw + (1-Yw)*kc       # effective thermal conductivity
rho_bar = wood[0] + char[0]     # effective density

g = np.ones(m)*(1e-10)  # assume initial heat generation is negligible

# Solve system of equations [A]{T}={C} where T = A\C for each time step
#-------------------------------------------------------------------------------

for i in range(1, nt):
    
    # heat conduction
    T[i] = hc(m, dr, b, dt, h, Tinf, g, T[i-1], r, rho_bar, cp_bar, k_bar)
    
    # kinetic reactions
    wood[i], gas1[i], tar[i], gaschar[i], gas[i], char[i], g = liden(wood[i-1], gas1[i-1], tar[i-1], gaschar[i-1], gas[i-1],  char[i-1], T[i], H, dt)

    # update thermal properties
    cpw = 1112.0 + 4.85 * (T[i] - 273.15)
    cpc = 1003.2 + 2.09 * (T[i] - 273.15)
    
    # update effective thermal properties
    Yw = wood[i] / rhow
    cp_bar = Yw*cpw + (1-Yw)*cpc
    k_bar = Yw*kw + (1-Yw)*kc
    rho_bar = wood[i] + char[i]
    
# Print Concentration Balance
# ------------------------------------------------------------------------------

tot_rhow = wood + gas + tar + char    # should be equal to rhow
print('total concentration balance\n', tot_rhow)

# Plot Results
#-------------------------------------------------------------------------------

py.ion()
py.close('all')

def despine():
    ax = py.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    py.tick_params(axis='both', bottom='off', top='off', left='off', right='off')

py.figure(1)
py.plot(t, T[:, 0], '-b', lw=2, label='center')
py.plot(t, T[:, m-1], '-r', lw=2, label='surface')
py.axhline(Tinf, c='k', ls='-.', label='ambient')
py.xlabel('Time (s)')
py.ylabel('Temperature (K)')
py.legend(loc='best', numpoints=1)
py.grid()
despine()

py.figure(2)
py.plot(t, wood[:, 0], lw=2, label='wood')
py.plot(t, gas[:, 0], lw=2, label='gas')
py.plot(t, tar[:, 0], lw=2, label='tar')
py.plot(t, char[:, 0], lw=2, label='char')
py.xlabel('Time (s)')
py.ylabel('Concentration (kg/m^3)')
py.legend(loc='best', numpoints=1)
py.grid()
despine()

