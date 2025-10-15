"""Given vector c and y, generate alignment matrix

Usage:
    python3 ./src/align-mat.py
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt

def main():
    y = np.load('out/y.npy')
    c = np.load('out/c.npy')
    # y = np.concat((y[:500], y[1000:], y[500:1000]))

    window_size = 1024
    c_prime_windows = sliding_window_view(c, window_size)
    y_windows = sliding_window_view(y, window_size)

    print(np.info(c_prime_windows))
    print(np.info(y_windows))

    x = np.dot(c_prime_windows, y_windows.T)
    print(np.info(x))

    fig = plt.figure(figsize=(16, 9))
    plt.matshow(x)
    plt.title("Alignment Matrix")
    plt.xlabel("Y index")
    plt.ylabel("C index")
    plt.savefig("out/align-mat.png")

if __name__ == '__main__':
    main()
