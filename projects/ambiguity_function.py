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

from radardsp.waveforms.waveforms import SimplePulse, LFMPulse
from radardsp.waveforms.tools import plot_ambiguity_function, ambiguity_function
from radardsp.tools import db2linear, AWGN, shift_signal


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
    
if __name__ == "__main__":
    main()
