"""
Papadikis 2010 kinetic scheme for biomass fast pyrolysis. Primary and secondary
reaction pathways are included along with heat of generation.
"""

import numpy as np


def papadikis(wood, gas, tar, char, T, H, dt):

    # A = pre-factor (1/s) and E = activation energy (kJ/mol)
    A1 = 1.3e8;     E1 = 140    # wood -> gas
    A2 = 2e8;       E2 = 133    # wood -> tar
    A3 = 1.08e7;    E3 = 121    # wood -> char
    A4 = 4.28e6;    E4 = 108    # tar -> gas
    A5 = 1e6;       E5 = 108    # tar -> char
    R = 0.008314    # universal gas constant, kJ/mol*K
    
    # evaluate reaction rate constant for each reaction, 1/s
    K1 = A1 * np.exp(-E1 / (R * T))  # wood -> gas
    K2 = A2 * np.exp(-E2 / (R * T))  # wood -> tar
    K3 = A3 * np.exp(-E3 / (R * T))  # wood -> char
    K4 = A4 * np.exp(-E4 / (R * T))  # tar -> gas
    K5 = A5 * np.exp(-E5 / (R * T))  # tar -> char
    
    # reaction rate for each pathway as a concentration per second, C/s
    rw = -(K1+K2+K3)*wood           # wood rate
    rg = K1*wood + K4*tar           # gas reaction rate
    rt = K2*wood - (K4+K5)*tar      # tar reaction rate
    rc = K3*wood + K5*tar           # char reaction rate

    # new concentration for products
    nwood = wood + rw*dt        # wood concentration
    ngas = gas + rg*dt          # gas concentration
    ntar = tar + rt*dt          # tar concentration
    nchar = char + rc*dt        # char concentration

    # revise heat of generation term
    g = H*rw    # heat generation, W/m^3  

    return nwood, ngas, ntar, nchar, g
    

def liden(wood, gas1, tar, gaschar, gas, char, T, H, dt):
    """
    Primary and secondary kinetic reactions from Liden 1988 paper. Parameters
    for total wood converstion (K = K1+K3) and secondary tar conversion (K2) are
    the only ones provided in paper. Can calculate K1 and K3 from phistar.

    Parameters
    ----------
    wood = wood concentration, kg/m^3
    gas1 = gas concentration from tar -> gas, kg/m^3
    tar = tar concentation, kg/m^3
    gaschar = (gas+char) concentration, kg/m^3
    gas = gas concentration from tar -> gas and gaschar, kg/m^3
    char = char concentration from gaschar, kg/m^3
    T = temperature, K
    dt = time step, s

    Returns
    -------
    nwood = new wood concentration, kg/m^3
    ngas1 = new gas1 concentration, kg/m^3
    ntar = new tar concentration, kg/m^3
    ngaschar = new (gas+char) concentration, kg/m^3
    ngas = new gas concentration, kg/m^3
    nchar = new char concentration, kg/m^3
    g = new heat generation, W/m^3
    """
    # A = pre-factor (1/s) and E = activation energy (kJ/mol)
    A = 1.0e13;         E = 183.3     # wood -> tar and (gas + char)
    A2 = 4.28e6;        E2 = 107.5    # tar -> gas
    R = 0.008314        # universal gas constant, kJ/mol*K
    phistar = 0.703     # maximum theoretical tar yield, (-)

    # reaction rate constant for each reaction, 1/s
    K = A * np.exp(-E / (R * T))        # wood -> tar and (gas + char)
    K1 = K * phistar                    # from phistar = K1/K
    K2 = A2 * np.exp(-E2 / (R * T))     # tar -> gas
    K3 = K - K1                         # from K = K1 + K3
    
    # assume a fixed carbon to estimate individual char and gas yields
    FC = 0.14               # weight fraction of fixed carbon in wood, (-)
    c3 = FC/(1-phistar)     # char fraction of wood, (-)
    g3 = 1-c3               # gas fraction of wood, (-)
    
    # primary and secondary reactions
    rw = -K*wood            # wood rate
    rg = K2*tar             # gas rate
    rt = K1*wood - K2*tar   # tar rate
    rgc = K3*wood           # (gas+char) rate
    nwood = wood + rw*dt            # update wood concentration
    ngas1 = gas1 + rg*dt            # update gas concentration
    ntar = tar + rt*dt              # update tar concentration
    ngaschar = gaschar + rgc*dt     # update (gas+char) concentration
    ngas = ngas1 + g3*ngaschar      # update total gas concentration
    nchar = c3*ngaschar             # update char conentration

    # revise heat of generation term
    g = H*rw    # heat generation, W/m^3

    return nwood, ngas1, ntar, ngaschar, ngas, nchar, g

