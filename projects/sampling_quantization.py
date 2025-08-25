# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 17:00:57 2025

This project sees the simulation of how sampling and quantization affects the 
system. The goals are:
    - Simulate a radar pulse and sample it at different rates (Nyquist rate, just above it, below it).
    - Quantize with different bit-depths (e.g., 2-bit, 4-bit, 8-bit).
    - Compare distortion, SNR, and information loss.
    - Demonstrate aliasing when sampling too slowly.

Concepts used: sampling theorem, quantization noise, impact on radar signals.

@author: ericdvet
"""

