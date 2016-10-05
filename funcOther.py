"""
Utility functions for project.
"""

import numpy as np


def vol(rad):
    vr = (4/3)*np.pi*(rad**3)   # volume of each sphere at each radius
    v = vr[1:] - vr[0:-1]       # center sphere volume and outer shell volumes
    return v                    # return vector of volumes


def Tvol(T, vol):
    """
    Use center sphere volume and shell volumes as weights to calculate the
    volume average temperature of entire sphere.

    Parameters
    ----------
    T = vector or array of temperatures at each radius point, K
    vol = vector for center and subsequent shell volumes in sphere, m^3

    Returns
    -------
    Tv = volume average temperature of sphere as a weighted mean, K
    """
    # calculation depends on dimensions of T as 1D vector or 2D array
    if T.ndim == 1:
        Tavg = (T[:-1] + T[1:]) / 2         # average temp between each node
        Tv = np.average(Tavg, weights=vol)  # volume average weighted temp
    if T.ndim == 2:
        Tavg = (T[:, :-1] + T[:, 1:]) / 2
        Tv = np.average(Tavg, axis=1, weights=vol)
    return Tv

