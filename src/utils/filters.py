"""
filters.py
Collection of filters.
"""

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

def bilateral_filter_1d(signal, spatial_sigma=2.0, range_sigma=0.05, window_size=11):
    """
    Bilateral filter for 1D temporal signals to reduce motion artifacts
    
    Args:
        signal: 1D array of shape (T,) - the temporal signal
        spatial_sigma: Controls smoothing across time
        range_sigma: Controls smoothing based on intensity differences  
        window_size: Temporal neighborhood size
    """

    # reflect padding is the default padding in cv::bilateralFilter
    # https://docs.opencv.org/4.12.0/d4/d86/group__imgproc__filter.html#ga9d7064d478c95d60003cf839430737ed
    padded_signal = np.pad(signal, (window_size//2, window_size//2), 'reflect')
    signal_windows = sliding_window_view(padded_signal, window_size, writeable=True)

    time_diffs = np.arange(window_size) - window_size//2 # Temporal distance
    intensity_diffs = (signal_windows.T-signal_windows[:, window_size//2]).T  # Intensity difference

    spatial_weights = np.exp(-0.5 * (time_diffs / spatial_sigma) ** 2)
    range_weights = np.exp(-0.5 * (intensity_diffs / range_sigma) ** 2)

    filtered_signal = np.sum(spatial_weights * range_weights * signal_windows, axis=1) / np.sum(spatial_weights * range_weights, axis=1)

    return filtered_signal
