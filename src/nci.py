"""Simulation of single-pixel video in a static scene illuminated by just one light source.

Usage:
    python3 ./src/nci.py
"""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

rng = np.random.default_rng()

def main():

    # generate NCI video @ 30fps for 60 seconds
    c, y = generate_video(l=1, r=1, noise_variance=0.01, fps=30, duration=60)

    plt.plot(c)
    plt.plot(y, '.')
    plt.show()


def generate_video(l: float, r: float, noise_variance: float, fps: int, duration: int) -> NDArray:
    """Model of single-pixel video in a static scene without NCI

    Args:
        l (float): Power of light source
        r (float): Light transport coefficient
        noise_variance (float): Variance of random noise
        fps (int, optional): Video frames per second.
        duration (int, optional): Duration of video in seconds.

    Returns:
        NDArray: array representing pixel intensity over time
    """
    frame_count = fps*duration

    # noise from camera sensor (photon shot noise)
    n = rng.normal(loc=0, scale=np.sqrt(noise_variance), size=frame_count)
    # noise coded illumination
    c = generate_nci(size=frame_count)

    # equation 2 from paper
    y = (l+c)*r + n

    return c, y


def generate_nci(size: int) -> NDArray:
    """Generate code signal; see section 5 from paper.
    Code signal should be random, noise-like, zero-mean, and uncorrelated with each other.
    Create random discrete spectrum, then convert to time-domain signal with inverse FFT.
    Ensure time-domain signal is real by constructing spectrum such that lower and upper
    frequency bins are complex conjugate and mirrored versions of each other.

    Args:
        size (int): Length of output array

    Returns:
        NDArray: Array representing noise coded light intensity over time
    """

    bin_count = 127
    N = 2*bin_count + 2

    # randomly generate first half of freq bins
    phases = rng.uniform(0, 2*np.pi, bin_count)
    magnitudes = rng.uniform(0, 5, bin_count)
    freq_bins_lower = magnitudes*np.exp(1j*phases)

    # concatenate with mirrored conjugate version of lower bins
    freq_bins_upper = np.conjugate(freq_bins_lower)[::-1]
    # If N is odd, first bin is real (DC component)
    # If N is even, first and middle bins are real (DC and Nyquist components)
    # set DC and Nyquist components to 0 for now
    freq_bins = np.concat(([0], freq_bins_lower, freq_bins_upper)) if N%2 == 1 else np.concat(([0], freq_bins_lower, [0], freq_bins_upper))

    # generate real-valued signal in time-domain
    x = np.fft.ifft(freq_bins)
    np.testing.assert_almost_equal(x.imag, 0, err_msg='time-domain signal is not real-valued')
    # imaginary components of x are all less than 1e-17, so just discard them
    np.isclose
    x = x.real

    # concatenate x's back to back to desired length
    c = np.tile(x.real, int(np.ceil(size/N)))
    c = c[:size]

    return c


if __name__ == '__main__':
    main()
