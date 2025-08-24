# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 16:56:15 2025

@author: ericdvet
"""

import numpy as np

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