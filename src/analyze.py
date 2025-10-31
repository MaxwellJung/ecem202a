"""Analyze noise coded video (section 4)

Usage:
    python3 ./src/analyze.py
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import torch
import time
import cv2

# Check if GPU is available (CUDA for NVIDIA, MPS for Apple Silicon)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def main():
    c = np.load('out/c.npy')
    y = np.load('out/y.npy')
    
    # Simulate malicious video cut
    y = np.concatenate((y[:500], y[700:]))

    print(f"{y.shape=}")
    print(f"{c.shape=}")

    start_time = time.time()
    align_mat = generate_alignment_matrix(y, c)
    elapsed_time = time.time() - start_time

    print(align_mat)
    print(f"{align_mat.shape=}")
    print(f"Computation time: {elapsed_time:.4f} seconds")

    fig = plt.figure(figsize=(16, 9))
    plt.matshow(align_mat)
    plt.title("Alignment Matrix (GPU-Accelerated)")
    plt.xlabel("Y index")
    plt.ylabel("C index")
    plt.savefig("out/align-mat.png")
    print("Saved alignment matrix to out/align-mat.png")
    plt.close()

    y_order = align_mat.argmax(axis=0)

    r = calculate_r(y, c, y_order)
    print(f'{r}')
    print(f'{r.shape=}')
    fig = plt.figure(figsize=(16, 9))
    plt.imshow(r[0])
    plt.title("Code Image")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.savefig("out/r_estimate.png")
    plt.close()

    VIDEO_FPS = 30
    VIDEO_WIDTH = y.shape[1]
    VIDEO_HEIGHT = y.shape[2]
    out = cv2.VideoWriter('out/r_estimate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (VIDEO_HEIGHT, VIDEO_WIDTH), True)
    for frame in 255*r:
        out.write(frame.astype(np.uint8))
    out.release()


def generate_alignment_matrix(y, c, window_size=511):
    # make sure window_size is odd number
    # ideally some power of 2 minus 1

    # Create global vector y by reducing each frame to a single value
    y = np.mean(y, axis=(1,2,3))

    print("Creating sliding windows...")
    padded_c = np.pad(c, (window_size//2, window_size//2))
    padded_y = np.pad(y, (window_size//2, window_size//2))
    c_prime_windows = sliding_window_view(padded_c, window_size, writeable=True)
    y_windows = sliding_window_view(padded_y, window_size, writeable=True)

    print(f"{c_prime_windows.shape=}")
    print(f"{y_windows.shape=}")

    # Convert to PyTorch tensors and move to GPU
    c_prime_windows_torch = torch.from_numpy(c_prime_windows).float().to(device)
    y_windows_torch = torch.from_numpy(y_windows).float().to(device)

    # Perform matrix multiplication on GPU/MPS (or by default your CPU)
    align_mat = torch.matmul(y_windows_torch, c_prime_windows_torch.T)
    align_mat = align_mat.T

    # Move result back to CPU and convert to NumPy
    align_mat = align_mat.cpu().numpy()

    return align_mat


def calculate_r(y, c, y_order=None, window_size=511):
    VIDEO_LENGTH = y.shape[0]
    VIDEO_WIDTH = y.shape[1]
    VIDEO_HEIGHT = y.shape[2]
    VIDEO_CHANNEL = y.shape[3]
    CODE_LENGTH = c.shape[0]

    if y_order is None:
        align_mat = generate_alignment_matrix(y, c)
        y_order = align_mat.argmax(axis=0)

    # [TODO] vectorize below nested loops
    r = []
    for y_index, c_index in enumerate(y_order):
        for width_i in range(VIDEO_WIDTH):
            for height_i in range(VIDEO_HEIGHT):
                for channel_i in range(VIDEO_CHANNEL):
                    y_window_start = 0 if y_index - window_size//2 < 0 else y_index - window_size//2
                    y_window_end = y_index + window_size//2 if y_index + window_size//2 <= VIDEO_LENGTH else VIDEO_LENGTH
                    c_window_start = 0 if c_index - window_size//2 < 0 else c_index - window_size//2
                    c_window_end = c_index + window_size//2 if c_index + window_size//2 <= CODE_LENGTH else CODE_LENGTH

                    y_window = y[y_window_start:y_window_end, width_i, height_i, channel_i]
                    c_window = c[c_window_start:c_window_end]
                    r_x = np.dot(c_window, y_window)/np.dot(c_window.T, c_window)
                    r.append(r_x)
    r = np.array(r).reshape((VIDEO_LENGTH, VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_CHANNEL))

    return r


if __name__ == '__main__':
    main()
