# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 14:05:00 2025

This project sees the exploration of the ambiguity function for different
waveforms:
    - Write a function to compute and plot the ambiguity function for different
        radar waveforms:
        - Rectangular pulse
        - Linear frequency modulated (LFM) chirp
        - Phase-coded waveform (e.g, Barker code)
    - Compare range-Doppler resolution trade-offs.
    - Discuss ambiguity sidelobes and how waveform design impacts them.

Concepts used: waveform analysis, range-Doppler coupling, ambiguity functions.

@author: ericdvet
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import scipy.fft as fft
from mpl_toolkits.mplot3d import Axes3D  

from radardsp.waveforms.waveforms import SimplePulse
from radardsp.waveforms.tools import matched_filter
from radardsp.tools import db2linear, AWGN


def main():
    
    SIMPLE_PULSE = True
    LFM_PULSE = True

    f = 1e5        # Carrier frequency
    sampling_frequency = f*4
    
    sampling_frequency = sampling_frequency
    period = 1 / sampling_frequency

    tau = 1e-4                              # Pulse length 
    times = np.arange(-tau, tau, period)    # |tau| includes entire pulse
    n = len(times)
    
    # Zero-Doppler narrow = good range resolution
    # Zero-delay narrow = good Doppler resolution
    # Flat overall other than peak = good side lobe supression
    
    # Simple pulse:
    # Since the zero doppler cut is a triangle, which is rather wide,
    # it shows that the simple pulse has poor resolution. The zero-Doppler
    # is a sinc function, which isn't ideal for great resolution either.
    # There's also no sidelobe supression happening.
    if SIMPLE_PULSE:
        simple_pulse = SimplePulse(n=n, tau=tau, f=f)
        x = simple_pulse.waveform(times)                    # Transmitted waveform
        
        A, f_d = ambiguity_function(x = x, tau = tau, times = times)
        
        plot_ambiguity_function(A = A, times = times, f_d = f_d, 
                                waveform_name = "Simple Pulse")
    
    # LFM Pulse
    # The zero-doppler is very narrow which means the LFM pulse has good range
    # reslution. THe zero-delay cut is okay, which means it has okay 
    # Doppler resolution. There are still sidelobes so there isn't much
    # sidelobe suppression.
    if LFM_PULSE:
        lfm_pulse = LFMPulse(n = n, tau = tau, bandwidth = f)
        x = lfm_pulse.waveform(t = times)
        
        A, f_d = ambiguity_function(x = x, tau = tau, times = times)
        
        plot_ambiguity_function(A = A, times = times, f_d = f_d, 
                                waveform_name = "LFM")
        
    # TODO: Barker codes
    
class LFMPulse:
    """
    LFM Pulse waveform class.
    """
    
    def __init__(self, n : int, tau : float, bandwidth : float):
        """

        Parameters
        ----------
        n : int
            Number of samples.
        tau : float
            Pulse duration.
        bandwidth : float
            Bandwidth.

        Returns
        -------
        None.

        """
        
        self.n = n
        self.tau = tau
        self.bandwidth = bandwidth
        
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

        x = self.a(t) * np.exp(1j*(self.theta(t)))
        
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
            Amplitude modification for LFM pulse waveform.

        """
        
        a = np.zeros(self.n)
        for i, t_i in enumerate(t):
            if (0 <= t_i and t_i <= self.tau):
                a[i] = 1
                
        return a
                
    def theta(self, t : np.ndarray):
        """

        Parameters
        ----------
        t : np.ndarray
            Time array.

        Returns
        -------
        theta: np.ndarray
            Phase of frequency modification for LFM pulse waveform.

        """
        
        theta = np.pi * self.bandwidth * t**2 / self.tau
        
        return theta
    
def ambiguity_function(x : np.ndarray, tau : float, times : np.ndarray):
    """
    Generate ambiguity function for waveform.

    Parameters
    ----------
    x : np.ndarray
        Waveform x(t).
    tau : float
        Sampling frequency.
    times : np.ndarray
        Time samples of x(t).

    Returns
    -------
    A : np.ndarray
        Ambiguity function.
    f_d : np.ndarray
        Frequency axis. Returned for plotting purposes.

    """
    
    f_d = np.linspace(-10/tau, 10/tau, np.size(times))  # To make the figures
                                                        # identical to Richards
                                                    
    N = np.size(x)
    A = np.zeros((N, N), dtype=complex)     # Ambiguity function initialized
    
    s = times
    ds = s[1] - s[0]            
    
    for t_i, tau in enumerate(times):
        shift = int(round(tau / ds))        # tau / ds = index to shift by
        x_shifted = np.conj(shift_signal(x, shift))
        for f_i, fd in enumerate(f_d):
            A[t_i, f_i] = np.sum(   # Richards 2005, Equation 4.30
                x * x_shifted * np.exp(-1j * 2*np.pi * fd * s)) * ds
            
    return A, f_d
        

    
def plot_ambiguity_function(A : np.ndarray, times : np.ndarray, f_d,
                            waveform_name : str):
    """
    Plot ambiguity function in 3D, and the zero-Doppler and zero-Cut of the 
    pulse.

    Parameters
    ----------
    A : np.ndarray
        x(t).
    times : np.ndarray
        t.
    f_d : TYPE
        F_d.
    waveform_name : str
        Waveform name.

    Returns
    -------
    None.

    """
    
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    
    Tau, FD = np.meshgrid(times, f_d, indexing='ij')  # Create meshgrid
    ax.plot_surface(Tau, FD, np.abs(A), cmap='inferno')
    
    ax.set_xlabel('Delay [s]')
    ax.set_ylabel('Doppler mismatch [Hz]')
    ax.set_zlabel('|A(t, F_D)|')
    ax.set_title(f'{waveform_name} Ambiguity Function')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    t0 = np.argmin(np.abs(times)) # Index at which the time was zero
    f0 = np.argmin(np.abs(f_d))
    
    plt.figure()
    plt.plot(f_d, np.abs(A[t0,:]))
    plt.xlabel("Doppler Mismatch")
    plt.ylabel("Amplitude")
    plt.title(f"{waveform_name} Zero-Delay Cut")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    plt.figure()
    plt.plot(times, np.abs(A[:,f0]))
    plt.xlabel("Delay")
    plt.ylabel("Amplitude")
    plt.title(f"{waveform_name} Zero-Doppler Cut")
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
def shift_signal(x : np.ndarray, shift : int):
    """
    Shift a signal. Unlike a normal shift that could be accomplished with
    something like numpy.roll(), this ensures that the shift is 
    pre and post logged with zeros.

    Parameters
    ----------
    x : np.ndarray
        Signal to be shifted.
    shift : int
        Amount to be shifted.

    Returns
    -------
    x_shifted : np.ndarray
        Shifted signal.

    """
    N = len(x)
    x_shifted = np.zeros(N, dtype=complex)
    if shift > 0:
        x_shifted[shift:] = (x[:N-shift])
    elif shift < 0:
        x_shifted[:N+shift] = (x[-shift:])
    else:
        x_shifted = (x)
    return x_shifted

if __name__ == "__main__":
    main()
