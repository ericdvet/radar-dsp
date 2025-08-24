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

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt

from radardsp.waveforms.waveforms import SquarePulse
from radardsp.waveforms.tools import matched_filter
from radardsp.tools import db2linear, AWGN


def main():

    f = 10e6        # Carrier frequency
    R = 12000       # Distance to target
    n = 1000        # Number of samples

    tau = 1e-4                              # Pulse length 
    times = np.linspace(-tau*5, tau*5, n)   # Time vector
    dT = times[1] - times[0]                # Sampling period
    
    square_pulse = SquarePulse(n = n, tau = tau, f = f)
    x = square_pulse.waveform(times)                    # Transmitted waveform
    y = square_pulse.waveform(times - 2*R/const.c)      # Received waveform
                                                        # of 2*R/c delay.
    
    y = AWGN(y = y, noise_percent = 0.1)    # Adding gaussian noise to received
                                            # waveform
    
    plt.figure()
    plt.plot(y / np.max(y), label="Received Signal")
    plt.plot(x / np.max(x), label="Transmitted Signal")
    
    # Matched filter
    alpha = 1
    h = alpha * np.conj(x[::-1])    # Because the matched filter is the time
                                    # reversed conjugate of the complex 
                                    # waveform.
    
    # Matched filterd
    y_mf, _ = matched_filter(waveform = x, y = y)
    
    # Computing the time of flight, and converting it to meters.
    peak_idx = np.argmax(np.abs(y_mf))
    tof = (peak_idx * dT)
    R_o = 0.5 * const.c * tof
    
    plt.plot(np.abs(y_mf) / np.max(np.abs(y_mf)), label="Received Signal (after matched filter)")
    plt.legend()
    plt.show()
    
    print(f"Estimated distance = {R_o:.0f}.")    

# TODO: Find a spot for this.
def radar_range_simple_point_target(P_t: float, G: float, wavelength: float,
                                    RCS: float, R: float,
                                    L_system: float = 1,
                                    L_atmospheric: float = 1):
    """
    Compute the received power for a simple point target using the radar range
    equation.

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
