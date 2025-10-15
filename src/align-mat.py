"""Given vector c and y, generate alignment matrix

Usage:
    python3 ./src/align-mat.py
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

def main():
    y = np.load('out/y.npy')
    VIDEO_WIDTH = y.shape[1]
    VIDEO_HEIGHT = y.shape[2]
    c = np.load('out/c.npy')
    # simulate malicious video cut
    y = np.concat((y[:500], y[1000:], y[600:800]))

    window_size = 1024
    c_prime_windows = sliding_window_view(c, window_size)
    # c_prime_windows axes = (window_index, sample_index)
    y_windows = sliding_window_view(y, (window_size,1,1)).squeeze().reshape(-1, VIDEO_WIDTH*VIDEO_HEIGHT, window_size)
    # rearrange y_windows axes to (pixel_index, window_index, sample_index)
    y_windows = y_windows.transpose((1, 0, 2))

    print(np.info(c_prime_windows))
    print(np.info(y_windows))

    x = np.dot(y_windows, c_prime_windows.T).T
    x = np.mean(x, axis=2)
    print(np.info(x))

    fig = plt.figure(figsize=(16, 9))
    plt.matshow(x)
    plt.title("Alignment Matrix")
    plt.xlabel("Y index")
    plt.ylabel("C index")
    plt.savefig("out/align-mat.png")

if __name__ == '__main__':
    main()
