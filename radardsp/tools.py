# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 16:56:51 2025

@author: ericdvet
"""

import os
import sys


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