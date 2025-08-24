# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 16:56:51 2025

@author: ericdvet
"""

import numpy as np

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