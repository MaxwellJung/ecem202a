"""
filters.py
Collection of filters.
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def bilateral_filter_1d(signal, spatial_sigma=2.0, range_sigma=0.05, window_size=11):
    """
    Bilateral filter for 1D temporal signals to reduce motion artifacts
    
    Args:
        signal: 1D array of shape (T,) - the temporal signal
        spatial_sigma: Controls smoothing across time
        range_sigma: Controls smoothing based on intensity differences  
        window_size: Temporal neighborhood size
    """
    filtered_signal = np.zeros_like(signal)
    half_window = window_size // 2
    
    for t in range(len(signal)):
        # Get temporal neighborhood
        t_start = max(0, t - half_window)
        t_end = min(len(signal), t + half_window + 1)
        
        # Get neighborhood values and their time indices
        neighborhood = signal[t_start:t_end]
        time_indices = np.arange(t_start, t_end)
        
        # Calculate weights
        time_diffs = time_indices - t  # Temporal distance
        intensity_diffs = neighborhood - signal[t]  # Intensity difference
        
        # Gaussian kernels
        spatial_weights = np.exp(-0.5 * (time_diffs / spatial_sigma) ** 2)
        range_weights = np.exp(-0.5 * (intensity_diffs / range_sigma) ** 2)
        
        # Combined weights
        weights = spatial_weights * range_weights
        weights_sum = np.sum(weights)
        
        if weights_sum > 0:
            filtered_signal[t] = np.sum(weights * neighborhood) / weights_sum
        else:
            filtered_signal[t] = signal[t]
            
    return filtered_signal
