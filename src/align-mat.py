"""Given vector c and y, generate alignment matrix

Usage:
    python3 ./src/align-mat.py
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

def main():
    y = np.load('out/y.npy')
    VIDEO_WIDTH = y.shape[0]
    VIDEO_HEIGHT = y.shape[1]
    c = np.load('out/c.npy')
    # simulate malicious video cut
    # y = np.concat((y.T[:500], y.T[1000:], y.T[500:1000])).T

    window_size = 1024
    c_prime_windows = sliding_window_view(c, window_size)
    y_windows = sliding_window_view(y, (1,1,window_size)).squeeze().reshape(VIDEO_WIDTH*VIDEO_HEIGHT, -1, c_prime_windows.shape[1])

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
