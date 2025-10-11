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
    c = generate_nci(sampling_freq=fps, size=frame_count)

    # equation 2 from paper
    y = (l+c)*r + n

    return y


def generate_nci(sampling_freq: int, size: int):
    """Generate code signal; see section 5 from paper.
    Code signal should be random, noise-like, zero-mean, and uncorrelated with each other.
    Currently outputs 1 Hz cosine wave.

    Args:
        sampling_freq (int): _description_
        size (int): _description_

    Returns:
        _type_: _description_
    """

    # convert sample index (0, 1, 2, ..., size-1) to corresponding time value
    sampling_period = 1/sampling_freq
    n = np.arange(size)
    t = sampling_period * n

    code_freq = 1
    w = 2*np.pi*code_freq
    c = np.cos(w*t)

    return c


if __name__ == '__main__':
    main()
