"""Given vector c and y, generate alignment matrix using PyTorch (GPU-accelerated)

Usage:
    python3 ./src/align-mat.py
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import torch
import os
import time

def main():
    # Check if GPU is available (CUDA for NVIDIA, MPS for Apple Silicon)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    start_time = time.time()

    y = np.load('../out/y.npy')
    VIDEO_WIDTH = y.shape[1]
    VIDEO_HEIGHT = y.shape[2]
    c = np.load('../out/c.npy')
    
    # Simulate malicious video cut
    y = np.concatenate((y[:500], y[700:]))

    window_size = 256
    print("Creating sliding windows...")
    c_prime_windows = sliding_window_view(c, window_size)
    # c_prime_windows axes = (window_index, sample_index)
    y_windows = sliding_window_view(y, (window_size, 1, 1)).squeeze().reshape(-1, VIDEO_WIDTH * VIDEO_HEIGHT, window_size)
    # rearrange y_windows axes to (pixel_index, window_index, sample_index)
    y_windows = y_windows.transpose((1, 0, 2))

    print(f"c_prime_windows shape: {c_prime_windows.shape}")
    print(f"y_windows shape: {y_windows.shape}")

    # Convert to PyTorch tensors and move to GPU
    c_prime_windows_torch = torch.from_numpy(c_prime_windows).float().to(device)
    y_windows_torch = torch.from_numpy(y_windows).float().to(device)

    # Perform matrix multiplication on GPU/MPS (or by default your CPU)
    x = torch.matmul(y_windows_torch, c_prime_windows_torch.T).transpose(0, 1).transpose(1, 2)
    x = torch.mean(x, dim=2)
    x = x.T

    # Move result back to CPU and convert to NumPy
    x = x.cpu().numpy()
    print(f"x shape: {x.shape}")
    elapsed_time = time.time() - start_time
    print(f"Computation time: {elapsed_time:.4f} seconds")

    fig = plt.figure(figsize=(16, 9))
    plt.matshow(x)
    plt.title("Alignment Matrix (GPU-Accelerated)")
    plt.xlabel("Y index")
    plt.ylabel("C index")
    plt.savefig("../out/align-mat.png")
    print("Saved alignment matrix to ../out/align-mat.png")

if __name__ == '__main__':
    main()
