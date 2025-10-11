"""Simulation of single-pixel video in a static scene illuminated by just one light source.

Usage:
    python3 ./src/nci.py
"""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

def main():
    y = generate_video(l=1, r=1, noise_variance=0.01, fps=30, duration=10)
    plt.plot(y, '.')
    plt.show()


def generate_video(l: float, r: float, noise_variance: float, fps: int = 30, duration: int = 10) -> NDArray:
    """Model of single-pixel video in a static scene without NCI

    Args:
        l (float): Power of light source
        r (float): Light transport coefficient
        noise_variance (float): Variance of random noise
        fps (int, optional): Video frames per second. Defaults to 30.
        duration (int, optional): Duration of video in seconds. Defaults to 10.

    Returns:
        NDArray: array representing pixel intensity over time
    """
    frame_count = fps*duration

    # noise from camera sensor (photon shot noise)
    n = np.random.normal(loc=0, scale=np.sqrt(noise_variance), size=frame_count)
    # noise coded illumination
    c = generate_nci(size=frame_count)

    # equation 2 from paper
    y = (l+c)*r + n

    return y


def generate_nci(size: int):
    # [TODO] Generate pseudo random code

    return 0


if __name__ == '__main__':
    main()
