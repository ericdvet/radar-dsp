# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 17:00:57 2025

This project sees the simulation of how sampling and quantization affects the 
system. The goals are:
    - Simulate a radar pulse and sample it at different rates (Nyquist rate, 
        just above it, below it).
    - Quantize with different bit-depths (e.g., 2-bit, 4-bit, 8-bit).
    - Compare distortion, SNR, and information loss.
    - Demonstrate aliasing when sampling too slowly.

Concepts used: sampling theorem, quantization noise, impact on radar signals.

@author: ericdvet
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
import scipy.fft as fft

from radardsp.waveforms.waveforms import SquarePulse
from radardsp.waveforms.tools import matched_filter
from radardsp.tools import db2linear, AWGN


def main():
    
    # Simulation of the effect of different sampling rates on the spectrum
    # when using a simple square pulse waveform. There is minimal aliasing
    # when at the Nyquist frequency (2f), and no aliasing above it, and 
    # overwheling aliasing below it.

    f = 2e6        # Carrier frequency
    R = 12000       # Distance to target
    
    plt.figure()
    sampling_frequencies = [1e6, 2e6, 4e6, 8e6, 16e6]
    for i, sampling_frequency in enumerate(sampling_frequencies):
        sampling_frequency = sampling_frequency
        period = 1 / sampling_frequency
    
        tau = 1e-4                              # Pulse length 
        times = np.arange(-tau, tau, period)
        n = len(times)
        dT = times[1] - times[0]                # Sampling period
        
        square_pulse = SquarePulse(n = n, tau = tau, f = f)
        x = square_pulse.waveform(times)                    # Transmitted waveform
        y = square_pulse.waveform(times - 2*R/const.c)      # Received waveform
                                                            # of 2*R/c delay
        
        spectrum = fft.fftshift(fft.fft(x))
        xf = fft.fftfreq(n, period)
        xf = fft.fftshift(xf)
        # plt.subplot(1, 5, i+1)
        plt.plot(xf, np.abs(spectrum), 
                 label=f"Sampling at {sampling_frequency:.2} Hz")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    main()
