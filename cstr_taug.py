"""
Compare tar yield for different gas residence times for reactor model based on
a series of CSTR reactors at steady-state conditions. Chemistry in each reactor
based on Liden 1988 kinetic scheme for biomass fast pyrolysis in a bubbling
fluidized bed reactor.

Test rxns- Based on Liden's (1988) kinetics
R1: W => t1*T (wood to tar), k1 = rate coeff. (1/s)
R2: T => g2*G (tar to gas), k2 = rate coeff. (1/s)
R3: W => c3*C + g3*G (wood to char + gas), k3 = rate coeff. (1/s)

Stagewise mass balances for each species:
dyW(i)/dt = -(k1+k3)*yW(i)+yW(i-1)/tau-yW(i)/tau => (Wood)
dyT(i)/dt = t1*k1*yW(i)-k2*yT(i)+yT(i-1)/tau-yT(i)/tau (Tar)
dyG(i)/dt = g2*k2*yT(i)+g3*k3*yW(i)+yG(i-1)/tau-yG(i)/tau (Gas)
dyC(i)/dt = c3*k3*yW(i)+yC(i-1)/tau-yC(i)/tau (Carbonized char)

Explicit s.s. solution to mass balances if done in proper sequence
General pattern yi = (i inflow + i gen rate*tau)/(1+i sink ks*tau)
yW = (1 + 0*tau)/(1+(k1+k3)*tau)
yT = (0 + t1*k1*yW*tau)/(1+k2*tau)
yG = (0 + g2*k2*yT+g3*k3*yW*tau)/(1+0)
yC = (0 + c3*k3*yW*tau)/(1+0)
"""

import numpy as np
import matplotlib.pyplot as py

# Test case operating conditions and constant parameters
Rgas = 8.314    # Ideal gas constant (J/mole K)
TK = 773        # Reaction temp (K)
yfw = 1.0       # Normalized wood feed

def prod(taus, taug, nstages):
    # Residence time in each stage
    tsn = taus/nstages  # Solids residence time in each stage (s)
    tgn = taug/nstages  # Gas residence time in each stage (s)
    # Kinetics parameters
    phi = 0.703     # Max tar yield fraction
    FC = 0.14       # Wt. fraction fixed C
    t1 = 1          # Tar mass formed/mass wood converted in rxn. 1
    g2 = 1          # Gas mass formed/mass tar converted in rxn. 2
    c3 = FC/(1-phi) # Char mass formed/mass wood converted in rxn. 3
    g3 = 1-c3       # Gas mass formed/mass wood converted in rxn. 3
    k2 = 4.28e6*np.exp(-107.5e3/Rgas/TK)    # Rxn. 2 rate coeff. (1/s)
    k = 1e13*np.exp(-183.3e3/Rgas/TK)       # Sum of rxn. 1 & 3 rate coefficients (1/s)
    k1 = phi*k                              # Rxn 1 rate constant (1/s)
    k3 = (1-phi)*k                          # Rxn. 3 rate constant (1/s)

    # Set up species solution vectors
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
    yCW[1] = yW[1]+yC[1]      # Total carbonized char + uncoverted wood

    # Mass balances for remaining stages
    for i in range(2, nstages):
        yW[i] = yW[i-1]/(1+k*tsn)                       # Wood in exit of stage i
        yT[i] = (yT[i-1]+t1*k1*yW[i]*tsn)/(1+k2*tgn)    # Tar in exit of stage i
        yG[i] = yG[i-1]+g2*k2*yT[i]*tgn+g3*k3*yW[i]*tsn # Gas in exit
        yC[i] = yC[i-1]+c3*k3*yW[i]*tsn                 # Carbonized char in exit
        yCW[i] = yC[i]+yW[i]                            # Total wood + carbonized char

    return yW, yT, yG, yC, yCW

nstages = 10        # No. of CSTR stages
taus = 4            # Total solids residence time (s)
taug = 0.5          # Total gas residence time (s)

yW, yT, yG, yC, yCW = prod(taus=4, taug=0.5, nstages=10)
yW2, yT2, yG2, yC2, yCW2 = prod(taus=4, taug=1, nstages=10)
yW3, yT3, yG3, yC3, yCW3 = prod(taus=4, taug=2, nstages=10)

# Plot Results
# -----------------------------------------------------------------------------

ns = range(nstages)   # list for numbers of stages

def despine():
    ax = py.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.tick_params(bottom='off', top='off', left='off', right='off')

py.ion()
py.close('all')

py.figure(1)
py.plot(yT, lw=2, label='0.5 s')
py.plot(yT2, lw=2, label='1.0 s')
py.plot(yT3, lw=2, label='2.0 s')
py.xlabel('Reactor Height')
py.ylabel('Tar Yield (normalized)')
py.legend(loc='best', numpoints=1)
py.grid()

py.figure(2, figsize=(5, 8))
py.plot(yT, ns, lw=2, label='0.5 s')
py.plot(yT2, ns, lw=2, label='1.0 s')
py.plot(yT3, ns, lw=2, label='2.0 s')
py.ylabel('Reactor Height')
py.xlabel('Tar Yield (normalized)')
py.legend(loc='best', numpoints=1)
py.grid()
despine()

# py.figure(3)
# py.plot(yW, lw=2, label='wood')
# py.plot(yT, lw=2, label='tar')
# py.plot(yG, lw=2, label='gas')
# py.plot(yC, lw=2, label='char')
# py.xlabel('Stage Number')
# py.ylabel('Product Yield (normalized)')
# py.legend(loc='best', numpoints=1)
# py.grid()
