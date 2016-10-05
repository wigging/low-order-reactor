"""
Plot exit product yield when same reactor is divided into a number of stages.
"""

import numpy as np
import matplotlib.pyplot as py

# Function
# ------------------------------------------------------------------------------

def cstr(T, nstages, taus, taug, yfw):
    tsn = taus/nstages  # solids residence time in each stage (s)
    tgn = taug/nstages  # gas residence time in each stage (s)
    Rgas = 8.314        # ideal gas constant (J/mole K)

    print('tsn = ', tsn, 'and tgn = ', tgn)

    # kinetics parameters
    phi = 0.703     # Max tar yield fraction
    FC = 0.14       # Wt. fraction fixed C
    t1 = 1          # Tar mass formed/mass wood converted in rxn. 1
    g2 = 1          # Gas mass formed/mass tar converted in rxn. 2
    c3 = FC/(1-phi) # Char mass formed/mass wood converted in rxn. 3
    g3 = 1-c3       # Gas mass formed/mass wood converted in rxn. 3
    k2 = 4.28e6*np.exp(-107.5e3/Rgas/T)     # Rxn. 2 rate coeff. (1/s)
    k = 1e13*np.exp(-183.3e3/Rgas/T)        # Sum of rxn. 1 & 3 rate coefficients (1/s)
    k1 = phi*k                              # Rxn 1 rate constant (1/s)
    k3 = (1-phi)*k                          # Rxn. 3 rate constant (1/s)

    if nstages == 1:
        nstages = nstages+1
        # Set up species solution vectors
        yW = yfw*np.ones(nstages)   # Unconverted wood (normalized to feed)
        yT = np.zeros(nstages)      # Tar (noramlized to feed)
        yG = np.zeros(nstages)      # Light gases (normalized to feed)
        yC = np.zeros(nstages)      # Char (normalized to feed)

        # Mass balance for stage 1
        yW[1] = yfw/(1+k*tsn)                  # Wood in exit
        yT[1] = t1*k1*yW[1]*tsn/(1+k2*tgn)        # Tar in exit
        yG[1] = g2*k2*yT[1]*tgn+g3*k3*yW[1]*tsn      # Gas in exit
        yC[1] = c3*k3*yW[1]*tsn                   # Carbonized char in exit

        return yW, yT, yG, yC
    
    if nstages > 1:
        nstages = nstages+1
        # Set up species solution vectors
        yW = yfw*np.ones(nstages)   # Unconverted wood (normalized to feed)
        yT = np.zeros(nstages)      # Tar (noramlized to feed)
        yG = np.zeros(nstages)      # Light gases (normalized to feed)
        yC = np.zeros(nstages)      # Char (normalized to feed)

        # Mass balance for stage 1
        yW[1] = yfw/(1+k*tsn)                       # Wood in exit of stage 1
        yT[1] = t1*k1*yW[1]*tsn/(1+k2*tgn)          # Tar in exit of stage 1
        yG[1] = g2*k2*yT[1]*tgn+g3*k3*yW[1]*tsn     # Gas in exit of stage 1
        yC[1] = c3*k3*yW[1]*tsn                     # Char in exit of stage 1

        # Mass balances for remaining stages
        for i in range(2, nstages):
            yW[i] = yW[i-1]/(1+k*tsn)                       # Wood in exit of stage i
            yT[i] = (yT[i-1]+t1*k1*yW[i]*tsn)/(1+k2*tgn)    # Tar in exit of stage i
            yG[i] = yG[i-1]+g2*k2*yT[i]*tgn+g3*k3*yW[i]*tsn # Gas in exit of stage i
            yC[i] = yC[i-1]+c3*k3*yW[i]*tsn                 # Char in exit of stage i

        return yW, yT, yG, yC


def scstr(n, tau, t):
    """
    Exit age distribution (RTD) for solids in multistaged fluidized beds from 
    Eq 5 in Kunii 1991 book from pg 339.
    
    Parameters
    ----------
    n = number of stages, number of equal-sized beds in series
    tau = total solids residence time, s
    t = time vector, s
    
    Returns
    -------
    et = exit age distribution or RTD of solids as a whole
    """
    # solids residence time for each stage from Eq. 4
    ti = tau/n
    # RTD of the solids for the beds as a whole from Eq. 5
    et = 1/(np.math.factorial(n-1)*ti)*(t/ti)**(n-1)*np.exp(-t/ti)
    return et


# Parameters
# ------------------------------------------------------------------------------

T = 773         # reaction temperature, K
taus = 2.0      # total solids residence time, s
taug = 0.50     # total gas residence time, s
yfw = 1         # normalized mass fraction of initial wood, (-)

# Calculate Yields
# ------------------------------------------------------------------------------

w1, t1, g1, c1 = cstr(T, 1, taus, taug, yfw)
w2, t2, g2, c2 = cstr(T, 2, taus, taug, yfw)
w3, t3, g3, c3 = cstr(T, 3, taus, taug, yfw)
w4, t4, g4, c4 = cstr(T, 4, taus, taug, yfw)
w5, t5, g5, c5 = cstr(T, 5, taus, taug, yfw)
w6, t6, g6, c6 = cstr(T, 6, taus, taug, yfw)
w7, t7, g7, c7 = cstr(T, 7, taus, taug, yfw)
w8, t8, g8, c8 = cstr(T, 8, taus, taug, yfw)
w9, t9, g9, c9 = cstr(T, 9, taus, taug, yfw)
w10, t10, g10, c10 = cstr(T, 10, taus, taug, yfw)

print('1 stage, tar = ', t1)
print('2 stages, tar = ', t2)
print('3 stages, tar = ', t3)
print('4 stages, tar = ', t4)

stages = range(1, 11)
tar = [t1[-1], t2[-1], t3[-1], t4[-1], t5[-1], t6[-1], t7[-1], t8[-1], t9[-1], t10[-1]]

# Calculate RTD
# ------------------------------------------------------------------------------

tau = 2.0                       # residence time, s
t = np.linspace(0, 10, 200)     # time vector, s

n1 = 1                          # number of stages
rtd1 = scstr(n1, tau, t)        # RTD based on series CSTR

n3 = 3                          # number of stages
rtd3 = scstr(n3, tau, t)        # RTD based on series CSTR

n20 = 20                        # number of stages
rtd20 = scstr(n20, tau, t)      # RTD based on series CSTR

# Plot
# ------------------------------------------------------------------------------

def despine():
    ax = py.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    py.tick_params(bottom='off', top='off', left='off', right='off')

py.ion()
py.close('all')
py.style.use('presentation')

py.figure(1)
py.plot(stages, tar)
py.xlabel('Number of CSTRs')
py.ylabel('Exit Tar Yield (mass fraction)')
py.annotate('Solids residence time = {:.0f} sec'.format(taus), xy=(6, 0.625), size=14)
despine()

py.figure(2)
py.gca().invert_xaxis()
py.plot(stages, tar, lw=4)
py.ylabel('Exit Tar Yield (mass fraction)', fontsize=18)
py.annotate('Solids residence time = {:.0f} sec'.format(taus), xy=(9, 0.62), size=18)
py.annotate('PFR', xy=(65, 20), xycoords='figure points', size=18)
py.annotate('Mixing Level', xy=(250, 20), xycoords='figure points', size=18)
py.annotate('CSTR', xy=(500, 20), xycoords='figure points', size=18)
py.tick_params(axis='both', which='major', labelsize=18)
py.gca().axes.xaxis.set_ticklabels([])
despine()

py.figure(3)
py.plot(t, rtd1, lw=6)
py.ylim((0, 1.0))
py.xlabel('Time (s)', fontsize=18)
py.ylabel('RTD', fontsize=18)
py.tick_params(axis='both', which='major', labelsize=18)
py.grid()
despine()

py.figure(4)
py.plot(t, rtd3, lw=6)
py.ylim((0, 1.0))
py.xlabel('Time (s)', fontsize=18)
py.ylabel('RTD', fontsize=18)
py.tick_params(axis='both', which='major', labelsize=18)
py.grid()
despine()

py.figure(5)
py.plot(t, rtd20, lw=6)
py.ylim((0, 1.0))
py.xlabel('Time (s)')
py.ylabel('RTD')
py.tick_params(axis='both', which='major', labelsize=18)
py.grid()
despine()

