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
import matplotlib.pyplot as plt


def main():

    plt.close('all')

    # Radar Range equation

    P_t = 30
    G = 44.14  # ~26000
    f = 10e6
    wavelength = const.c / f
    RCS = 1000
    R = 10000

    # P_r = radar_range_simple_point_target(P_t, G, wavelength, RCS, R)
    # print(f"P_r = {P_r:.4}")

    n = 1000

    tau = 1e-4
    times = np.linspace(-tau*5, tau*5, n)
    dT = times[1] - times[0]
    
    square_pulse = SquarePulse(n = n, tau = tau, f = f)
    x = square_pulse.waveform(times)
    y = square_pulse.waveform(times - 2*R/const.c)
    
    plt.figure()
    plt.plot(x)
    plt.plot(y)
    
    y = AWGN(y = y, noise_percent = 0.1)
    
    # plt.figure()
    # plt.plot(y)
    # plt.plot(y_og)
    
    # Matched filter
    alpha = 1
    h = alpha * np.conj(x[::-1])    # Because the matched filter is the time
                                    # reversed conjugate of the complex 
                                    # waveform.
    
    # Matched filterd
    y_mf = np.convolve(y, h, mode = "full")
    
    # plt.figure()
    # plt.plot(y_mf)
    # plt.show()
    
    # plt.figure()
    # plt.plot(np.abs(y_mf) / np.max(np.abs(y_mf)))
    # plt.plot(np.abs(y) / np.max(np.abs(y)))
    # plt.show()
    
    grd = len(x) - 1 # Oppenheim's group delay to fix the convolution offset
    tof = ((np.argmax(np.abs(y_mf)) - grd) * dT)
    R_o = 0.5 * const.c * tof
    
    print(R_o)

def AWGN(y : np.ndarray, noise_percent : float):
    """
    Add white gaussian noise to input vector.

    Parameters
    ----------
    y : np.ndarray
        Signal to add noise to.
    noise_percent : float
        Percent of signal power noise's variance should be at.

    Returns
    -------
    y : float
        Noisy signal.

    """
    signal_power = np.mean(np.abs(y)**2)
    noise = (np.random.normal(0, signal_power*noise_percent, len(y)) + 
             1j * np.random.normal(0, signal_power*noise_percent, len(y)))
    y = y + noise
    return y
    
class SquarePulse:
    """
    Square Pulse waveform class.
    """
    
    def __init__(self, n : int, tau : float, f : float):
        """

        Parameters
        ----------
        n : int
            Number of samples.
        tau : float
            Pulse duration.
        f : float
            Carrier frequency.

        Returns
        -------
        None.

        """
        
        self.n = n
        self.tau = tau
        self.f = f
        
    def waveform(self, t: np.ndarray):
        """

        Parameters
        ----------
        t : np.ndarray
            Time array.

        Returns
        -------
        x : np.ndarray
            Square pulse waveform.

        """
        omega = 2 * np.pi * self.f   # Convert frequency in Hz to radians
        x = self.a(t) * np.exp(1j*(omega*t + self.theta(t)))
        
        return x
        
    def a(self, t : np.ndarray):
        """

        Parameters
        ----------
        t : np.ndarray
            Time array.

        Returns
        -------
        a : np.ndarray
            Amplitude modification for square pulse waveform.

        """
        
        a = np.zeros(self.n)
        for i, t_i in enumerate(t):
            if (-self.tau/2 <= t_i and t_i <= self.tau/2):
                a[i] = 1 / np.sqrt(self.tau)
                
        return a
                
    def theta(self, t : np.ndarray):
        """

        Parameters
        ----------
        t : np.ndarray
            DESCRIPTION.

        Returns
        -------
        float
            Phase of frequency modification for square pulse waveform.

        """
        return np.zeros(self.n)


def db2linear(dB_unit: float):
    """
    Convert input from dB to linear units.

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
