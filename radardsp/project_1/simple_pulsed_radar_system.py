# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 20:42:26 2025

This project sees the simulation of a basic pulsed radar system. The goals are:
    - Implement a simple monostatic radar model with a single target at a fixed
        range.
    - Generate a pulsed radar waveform (rectangular pulse first.)
    - Propagate it to the target and back with a delay corresponding to the
        target's range.
    - Add noise to simulate an Additive White Gaussian Noise channel.
    - Plot the received signal and demonstrate how range can be extracted by
        measuring the delay.

Concepts used: signal models, time delay, range equation.

@author: ericdvet
"""

import numpy as np
import scipy.constants as const


def main():
    # Radar Range equation

    P_t = 30
    G = 44.14  # ~26000
    f = 10e9
    wavelength = const.c / f
    RCS = 100
    R = 10000

    P_r = radar_range_simple_point_target(P_t, G, wavelength, RCS, R)

    print(f"P_r = {P_r:.4}")


def db2linear(dB_unit: float):
    """

    Parameters
    ----------
    dB_unit : float
        Unit to be converted to linear.

    Returns
    -------
    float
        Unit in linear.

    """

    return 10**(dB_unit / 10)


def radar_range_simple_point_target(P_t: float, G: float, wavelength: float,
                                    RCS: float, R: float,
                                    L_system: float = 1,
                                    L_atmospheric: float = 1):
    """

    Parameters
    ----------
    P_t : float
        Transmitted power in dB.
    G : float
        Gains in the system in dB.
    wavelength : float
        Wavelength in meters.
    RCS : float
        Radar cross section in meters^2.
    R : float
        Range.
    L_system : float, optional
        System losses. The default is 1 (negligibile).
    L_atmospheric : float, optional
        Atmospheric losses. The default is 1 (negligibile).

    Returns
    -------
    P_r : float
        Received power.

    """

    P_t = db2linear(P_t)
    G = db2linear(G)

    P_r = (P_t * G**2 * wavelength**2 * RCS)/((4 * np.pi)**3 * R**4 * L_system
                                              * L_atmospheric)
    return P_r


if __name__ == "__main__":
    main()
