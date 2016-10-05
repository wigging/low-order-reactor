"""
Plot conversion for different residence times
"""

import numpy as np
import matplotlib.pyplot as py

# Function
# ------------------------------------------------------------------------------

def cstr(T, nstages, taus, taug):
    tsn = taus/nstages  # solids residence time in each stage (s)
    tgn = taug/nstages  # gas residence time in each stage (s)
    Rgas = 8.314        # ideal gas constant (J/mole K)

    # kinetics parameters
    phi = 0.80      # Max tar yield fraction
    FC = 0.14       # Wt. fraction fixed C
    t1 = 1          # Tar mass formed/mass wood converted in rxn. 1
    g2 = 1          # Gas mass formed/mass tar converted in rxn. 2
    c3 = FC/(1-phi) # Char mass formed/mass wood converted in rxn. 3
    g3 = 1-c3       # Gas mass formed/mass wood converted in rxn. 3

    k = 1e13*np.exp(-183.3e3/Rgas/T)        # Sum of rxn. 1 & 3 rate coefficients (1/s)
    k2 = 4.28e6*np.exp(-107.5e3/Rgas/T)     # Rxn. 2 rate coeff. (1/s)
    k1 = phi*k                              # Rxn 1 rate constant (1/s)
    k3 = (1-phi)*k                          # Rxn. 3 rate constant (1/s)

    # Set up species solution vectors
    yfw = 1.0                   # Normalized wood feed
    nstages = nstages + 1       # Numbers of stages
    yW = yfw*np.ones(nstages)   # Unconverted wood (normalized to feed)
    yT = np.zeros(nstages)      # Tar (noramlized to feed)
    yG = np.zeros(nstages)      # Light gases (normalized to feed)
    yC = np.zeros(nstages)      # Char (normalized to feed)
    yCW = np.zeros(nstages)     # Char + wood (normalized to feed)

    # Mass balance for stage 1
    yW[1] = yfw/(1+k*tsn)                       # Wood in exit
    yT[1] = t1*k1*yW[1]*tsn/(1+k2*tgn)          # Tar in exit
    yG[1] = g2*k2*yT[1]*tgn+g3*k3*yW[1]*tsn     # Gas in exit
    yC[1] = c3*k3*yW[1]*tsn                     # Carbonized char in exit

    # Mass balances for remaining stages
    for i in range(2, nstages):
        yW[i] = yW[i-1]/(1+k*tsn)                       # Wood in exit of stage i
        yT[i] = (yT[i-1]+t1*k1*yW[i]*tsn)/(1+k2*tgn)    # Tar in exit of stage i
        yG[i] = yG[i-1]+g2*k2*yT[i]*tgn+g3*k3*yW[i]*tsn # Gas in exit
        yC[i] = yC[i-1]+c3*k3*yW[i]*tsn                 # Carbonized char in exit

    return yW, yT, yG, yC


# Calculations
# ------------------------------------------------------------------------------

ts = 3.0    # solids residence time (s)
tg = 3.0    # gas residence time (s)

n3 = 3      # number of stages (-)
dt3 = ts/n3 # individual stage residence time (s)
tm3 = np.arange(0, ts+dt3, dt3)     # time vector for x-axis of plot (s)

n5 = 5      # number of stages (-)
dt5 = ts/n5 # individual stage residence time (s)
tm5 = np.arange(0, ts+dt5, dt5)     # time vector for x-axis of plot (s)

n10 = 10        # number of stages (-)
dt10 = ts/n10   # individual stage residence time (s)
tm10 = np.arange(0, ts+dt10, dt10)  # time vector for x-axis of plot (s)

n20 = 20        # number of stages (-)
dt20 = ts/n20   # individual stage residence time (s)
tm20 = np.arange(0, ts+dt20, dt20) # time vector for x-axis of plot (s)

w3, t3, g3, c3 = cstr(773, n3, ts, tg)
w5, t5, g5, c5 = cstr(773, n5, ts, tg)
w10, t10, g10, c10 = cstr(773, n10, ts, tg)
w20, t20, g20, c20 = cstr(773, n20, ts, tg)

# Plots
# ------------------------------------------------------------------------------

py.ion()
py.close('all')

py.figure(1)
py.plot(t5, lw=2)
py.plot(t10, lw=2)

py.figure(2)
py.plot(tm3, t3, lw=2, label='n=3')
py.plot(tm5, t5, lw=2, label='n=5')
py.plot(tm10, t10, lw=2, label='n=10')
py.plot(tm20, t20, lw=2, label='n=20')
py.xlabel('Solids Residence Time (s)')
py.ylabel('Tar Yield (mass fraction)')
py.legend(loc='best', numpoints=1)

