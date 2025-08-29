# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 16:52:30 2025

@author: ericdvet
"""

import numpy as np
import scipy.constants as const

class SimplePulse:
    """
    Simple Pulse waveform class.
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