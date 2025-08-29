# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 16:56:15 2025

@author: ericdvet
"""

import numpy as np
import matplotlib.pyplot as plt
from ..tools import shift_signal

def matched_filter(waveform : np.ndarray, y : np.ndarray, alpha : float = 1):
    """
    Creates a matched filter for waveform and applies it to signal.

    Parameters
    ----------
    waveform : np.ndarray
        Complex waveform (carrier).
    y : np.ndarray
        Signal ti apply matched filter to.
    alpha : float, optional
        Gain of matched filter. Defaulted to 0 since it has no impact on the
        SNR.

    Returns
    -------
    y_mf : TYPE
        DESCRIPTION.
    h : TYPE
        DESCRIPTION.

    """
    
    h = alpha * np.conj(waveform[::-1])

    
    y_mf = np.convolve(y, h, mode = "full")
    grd = len(waveform) - 1 # Group delay to fix the convolution offset
    y_mf = y_mf[grd:]
    
    return y_mf, h

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