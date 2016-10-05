"""
Reactor model using multiple CSTR in series at steady-state conditions.
Chemistry provided from Liden 1988 kinetics scheme for wood pyrolysis in a
bubbling fluidized bed reactor.

Script to test solution of general CSTR steady-state conversions for multiple
parallel and series 1st-order reactions.

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

# Parameters
# ------------------------------------------------------------------------------

T = 773             # reaction temperature, K
nstages = 10        # number of CSTR stages
taus = 4            # total solids residence time, s
taug = 0.5          # total gas residence time, s
yfw = 1             # normalized mass fraction of initial wood, (-)

# Function
# ------------------------------------------------------------------------------

def cstr(T, nstages, taus, taug, tv, wood):
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
    k2 = 4.28e6*np.exp(-107.5e3/Rgas/T)    # Rxn. 2 rate coeff. (1/s)

    tvr = 0.54                                  # reference tau for Liden 1988
    k = 1e13*np.exp(-183.3e3/Rgas/T)*(tvr/tv)   # Sum of rxn. 1 & 3 rate coefficients (1/s)

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

    return yW, yT, yG, yC

# Calculate Yields
# ------------------------------------------------------------------------------

tv = 1.6            # devolatilization time for Dp = 0.5 mm, s
yW, yT, yG, yC = cstr(T, nstages, taus, taug, tv, yfw)

tv = 8.6            # devolatilization time for Dp = 2 mm, s
yW2, yT2, yG2, yC2 = cstr(T, nstages, taus, taug, tv, yfw)

# Print Mass Balances
# ------------------------------------------------------------------------------

mout = yW+yT+yG+yC  # total mass out
mratio = mout/yfw   # ratio total mass out/total mass in
print('total mass out\n', mout)
print('total mass ratio (mass out / mass in)\n', mratio)

# Print Product Yields
# ------------------------------------------------------------------------------

print('- 0.5 mm sieve yields (top of reactor) -')
print('wood = ', yW[-1]*100)
print('tar = ', yT[-1]*100)
print('gas = ', yG[-1]*100)
print('char = ', yC[-1]*100)
print('mass balance = ', yW+yT+yG+yC)

print('- 2.0 mm sieve yields (top of reactor) -')
print('wood = ', yW2[-1]*100)
print('tar = ', yT2[-1]*100)
print('gas = ', yG2[-1]*100)
print('char = ', yC2[-1]*100)
print('mass balance = ', yW2+yT2+yG2+yC2)

# Plot Results
# -----------------------------------------------------------------------------

ns = range(nstages)   # list for numbers of stages

def despine():
    ax = py.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.tick_params(bottom='off', top='off', left='off', right='off')
    py.grid()

py.ion()
py.close('all')
#py.style.use('presentation')

py.figure(1)
py.plot(ns, yW, label='wood')
py.plot(ns, yT, label='tar')
py.plot(ns, yG, label='gas')
py.plot(ns, yC, label='char')
py.xlabel('Axial Reactor Height')
py.ylabel('Product Yields (wt. fraction)')
py.legend(loc='best', numpoints=1)
despine()

py.figure(2, figsize=(5, 8))
py.plot(yT, ns, lw=2, label='0.5 mm model')
py.plot(yT2, ns, lw=2, label='2.0 mm model')
py.axvline(0.71, c='b', lw=2, ls='--', label='0.5 mm sieve')
py.axvline(0.64, c='g', lw=2, ls='--', label='2.0 mm sieve')
py.xlim([0, 0.80])
py.xlabel('Tar Yield (wt. fraction)')
py.ylabel('Axial Reactor Height')
py.title('Static Particle Size')
py.legend(loc='best', numpoints=1, fontsize='medium', frameon=False)
despine()

