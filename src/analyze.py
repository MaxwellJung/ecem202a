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
    VIDEO_LENGTH = y.shape[0]
    VIDEO_WIDTH = y.shape[1]
    VIDEO_HEIGHT = y.shape[2]

    # Simulate malicious video cut
    y = np.concatenate((y[:VIDEO_LENGTH//4], y[2*VIDEO_LENGTH//4:]))
    # Simulate malicious photoshop (edit in a grey square in the middle)
    y[0:VIDEO_LENGTH//8, VIDEO_WIDTH//4:VIDEO_WIDTH//2, VIDEO_HEIGHT//4:VIDEO_HEIGHT//2] = 0.5

    # export edited video
    VIDEO_FPS = 30
    out = cv2.VideoWriter('out/y_edited.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (VIDEO_HEIGHT, VIDEO_WIDTH), True)
    for frame in (255*y).clip(min=0, max=255):
        out.write(frame.astype(np.uint8))
    out.release()

    print(f"{y.shape=}")
    print(f"{c.shape=}")

    start_time = time.time()
    align_mat = get_alignment_matrix(y, c)
    elapsed_time = time.time() - start_time

    fig = plt.figure(figsize=(16, 9))
    plt.matshow(align_mat)
    plt.title("Alignment Matrix")
    plt.xlabel("Y index")
    plt.ylabel("C index")
    plt.savefig("out/align-mat.png")
    plt.close()
    print("Saved alignment matrix to out/align-mat.png")

    r = calculate_r(y, c)
    fig = plt.figure(figsize=(16, 9))
    plt.imshow(r[0])
    plt.title("Code Image")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.savefig("out/r_estimate.png")
    plt.close()
    print("Saved code image to out/r_estimate.png")

    out = cv2.VideoWriter('out/r_estimate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (VIDEO_HEIGHT, VIDEO_WIDTH), True)
    for frame in (255*r).clip(min=0, max=255):
        out.write(frame.astype(np.uint8))
    out.release()
    print("Saved code video to out/r_estimate.mp4")


def get_alignment_matrix(y, c, window_size=511):
    # make sure window_size is odd number
    # ideally some power of 2 minus 1

    # Create global vector y by reducing each frame to a single value
    y = np.mean(y, axis=(1,2,3))

    padded_c = np.pad(c, (window_size//2, window_size//2))
    padded_y = np.pad(y, (window_size//2, window_size//2))
    c_prime_windows = sliding_window_view(padded_c, window_size, writeable=True)
    y_windows = sliding_window_view(padded_y, window_size, writeable=True)

    # Convert to PyTorch tensors and move to GPU
    c_prime_windows_torch = torch.from_numpy(c_prime_windows).float().to(device)
    y_windows_torch = torch.from_numpy(y_windows).float().to(device)

    # Perform matrix multiplication on GPU/MPS (or by default your CPU)
    align_mat = torch.matmul(y_windows_torch, c_prime_windows_torch.T)
    align_mat = align_mat.T

    # Move result back to CPU and convert to NumPy
    align_mat = align_mat.cpu().numpy()

    return align_mat


def calculate_r(y, c, window_size=511):
    align_mat = get_alignment_matrix(y, c)
    y_to_c = align_mat.argmax(axis=0)

    padded_c = np.pad(c, (window_size//2, window_size//2))
    padded_y = np.pad(y, ((window_size//2, window_size//2), (0, 0), (0, 0), (0, 0)))
    c_prime_windows = sliding_window_view(padded_c, window_size, writeable=True)[y_to_c]
    y_windows = sliding_window_view(padded_y, (window_size, 1, 1, 1), writeable=True).squeeze()
    # rearrange axes to allow broadcasting
    y_windows = y_windows.transpose([1,2,3,0,4])

    # Convert to PyTorch tensors and move to GPU
    c_prime_windows_torch = torch.from_numpy(c_prime_windows).float().to(device)
    y_windows_torch = torch.from_numpy(y_windows).float().to(device)

    # Perform matrix multiplication on GPU/MPS (or by default your CPU)
    r = torch.divide(torch.sum(torch.multiply(c_prime_windows_torch, y_windows_torch), dim=4), 
                     torch.sum(torch.multiply(c_prime_windows_torch, c_prime_windows_torch), dim=1))
    r = r.cpu().numpy()
    r = r.transpose([3,0,1,2])

    return r


if __name__ == '__main__':
    main()
