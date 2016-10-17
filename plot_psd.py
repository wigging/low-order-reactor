import numpy as np
import matplotlib.pyplot as py

# Particle size distribution from Figure 2 in Carpenter2014 paper
# -----------------------------------------------------------------------------

x05, y05 = np.loadtxt('data/05mm.csv', delimiter=",", unpack=True)   # 0.5 mm sieve
x2, y2 = np.loadtxt('data/2mm.csv', delimiter=",", unpack=True)      # 2.0 mm sieve

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
py.plot(x05, y05, color='w')
py.plot(x2, y2, color='w')
py.axvline(0.5, c='b', ls='--', lw=4, label='0.5 mm static size')
py.axvline(2.0, c='g', ls='--', lw=4, label='2.0 mm static size')
py.xlabel('Particle Sizes From Sieve (mm)', fontsize=18)
py.ylabel('Volume (%)', fontsize=18)
py.tick_params(axis='both', which='major', labelsize=18)
py.legend(loc='best', numpoints=1, fontsize=18)
despine()

py.figure(2)
py.plot(x05, y05, label='0.5 mm distribution')
py.plot(x2, y2, label='2.0 mm distribution')
#py.axvline(0.5, c='b', ls='--', label='0.5 mm static size')
#py.axvline(2.0, c='g', ls='--', label='2.0 mm static size')
py.xlabel('Particle Sizes From Sieve (mm)', fontsize=18)
py.ylabel('Volume (%)', fontsize=18)
py.tick_params(axis='both', which='major', labelsize=18)
py.legend(loc='best', numpoints=1, fontsize=18)
despine()

