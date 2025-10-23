"""Given vector c and y, generate alignment matrix using PyTorch (GPU-accelerated)

Usage:
    python3 ./src/align-mat.py
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import torch
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

    y = np.load('out/y.npy')
    VIDEO_WIDTH = y.shape[1]
    VIDEO_HEIGHT = y.shape[2]
    VIDEO_CHANNEL = y.shape[3]
    c = np.load('out/c.npy')
    
    # Simulate malicious video cut
    y = np.concatenate((y[:500], y[700:]))

    # Create global vector y by reducing each frame to a single value
    y = np.mean(y, axis=(1,2,3))

    window_size = 512
    print("Creating sliding windows...")
    c_prime_windows = sliding_window_view(np.concat([c, c[:window_size-1]]), window_size, writeable=True)
    # c_prime_windows axes = (window_index, sample_index)
    y_windows = sliding_window_view(np.concat([y, y[:window_size-1]]), window_size, writeable=True)

    print(f"{c_prime_windows.shape=}")
    print(f"{y_windows.shape=}")

    # Convert to PyTorch tensors and move to GPU
    c_prime_windows_torch = torch.from_numpy(c_prime_windows).float().to(device)
    y_windows_torch = torch.from_numpy(y_windows).float().to(device)

    # Perform matrix multiplication on GPU/MPS (or by default your CPU)
    align_mat = torch.matmul(y_windows_torch, c_prime_windows_torch.T)
    align_mat = align_mat.T
    print(align_mat)

    # Move result back to CPU and convert to NumPy
    align_mat = align_mat.cpu().numpy()
    print(f"{align_mat.shape=}")
    elapsed_time = time.time() - start_time
    print(f"Computation time: {elapsed_time:.4f} seconds")

    fig = plt.figure(figsize=(16, 9))
    plt.matshow(align_mat)
    plt.title("Alignment Matrix (GPU-Accelerated)")
    plt.xlabel("Y index")
    plt.ylabel("C index")
    plt.savefig("out/align-mat.png")
    print("Saved alignment matrix to out/align-mat.png")

if __name__ == '__main__':
    main()
