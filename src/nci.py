"""Simulation of single-pixel video in a static scene illuminated by one NCI light source.

Usage:
    python3 ./src/nci.py
"""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

rng = np.random.default_rng()

def main():

    # generate a 10 second NCI video @ 30fps
    VIDEO_FPS = 30
    VIDEO_DURATION = 100
    FRAME_COUNT = VIDEO_FPS*VIDEO_DURATION
    c, y = generate_video(l=1, r=1, noise_variance=0.01, fps=VIDEO_FPS, duration=VIDEO_DURATION)
    t = np.arange(FRAME_COUNT)/VIDEO_FPS

    np.save('out/c', c)
    np.save('out/y', y)

    # Plots for analysis
    fig = plt.figure(figsize=(16, 9))
    plt.plot(t, y, '.')
    plt.title("Y")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixel Intensity")
    plt.savefig("out/y.png")

    fig = plt.figure(figsize=(16, 9))
    plt.hist(y, bins=100)
    plt.title("Y Distribution")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Count")
    plt.savefig("out/y_histogram.png")

    fig = plt.figure(figsize=(16, 9))
    plt.step(t, c, where='post')
    plt.title("Coded Light Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig("out/c.png")

    fig = plt.figure(figsize=(16, 9))
    plt.hist(c, bins=100)
    plt.title("C Distribution")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.savefig("out/c_histogram.png")

    fig = plt.figure(figsize=(16, 9))
    plt.step(VIDEO_FPS*np.arange(len(c))/len(c), np.abs(np.fft.fft(c)), where='mid')
    plt.title("Magnitude Spectrum of C")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.savefig("out/c_spectrum_magnitude.png")

    fig = plt.figure(figsize=(16, 9))
    plt.step(VIDEO_FPS*np.arange(len(c))/len(c), np.angle(np.fft.fft(c)), where='mid')
    plt.title("Phase Spectrum of C")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase")
    plt.savefig("out/c_spectrum_phase.png")


def generate_video(l: float, r: float, noise_variance: float, fps: int, duration: int) -> NDArray:
    """Model of single-pixel video in a static scene with NCI

    Args:
        l (float): Power of light source
        r (float): Light transport coefficient
        noise_variance (float): Variance of random noise
        fps (int): Video frames per second.
        duration (int): Duration of video in seconds.

    Returns:
        NDArray: array representing pixel intensity over time
    """
    frame_count = fps*duration

    # noise from camera sensor (photon shot noise)
    n = rng.normal(loc=0, scale=np.sqrt(noise_variance), size=frame_count)
    # noise coded illumination
    c = generate_nci(w_m=9, w_s=fps, size=frame_count)

    # equation 2 from paper
    y = (l+c)*r + n

    return c, y


def generate_nci(w_m: float, w_s: float, size: int) -> NDArray:
    """Generate code signal; see section 5 from paper.
    Code signal should be random, noise-like, zero-mean, and uncorrelated with each other.
    Create random discrete spectrum, then convert to time-domain signal with inverse FFT.
    Ensure time-domain signal is real by constructing spectrum such that lower and upper
    frequency bins are complex conjugate and mirrored versions of each other.

    Args:
        w_m (float): Maximum bandwidth of signal in Hz
        w_s (float): Sampling frequency in Hz
        size (int): Length of output array

    Returns:
        NDArray: Array representing noise coded light intensity over time
    """

    # If N is odd, first bin (DC component) is real
    # If N is even, first and middle bins (DC and Nyquist components) are real
    # For now, choose even N
    N = 1024
    freq_bins = np.empty(N, dtype=complex)

    nyquist_freq = w_s/2
    valid_bins = int(N//2*(w_m/nyquist_freq))

    # randomly generate lower half of freq bins
    phases = rng.uniform(0, 2*np.pi, valid_bins)
    magnitudes = rng.uniform(0, 5, valid_bins)
    freq_bins[1:valid_bins+1] = magnitudes*np.exp(1j*phases)
    # set DC component to 0 (or other real value)
    freq_bins[0] = 0
    # set freq components outside bandwidth to 0
    freq_bins[valid_bins+1:N//2] = 0

    # set upper half of freq bins as mirrored and conjugate version of lower bins
    freq_bins[N//2+1:] = np.conjugate(freq_bins[1:N//2])[::-1]
    # set Nyquist component to 0 (or other real value)
    freq_bins[N//2] = 0

    # generate real-valued signal in time-domain
    x = np.fft.ifft(freq_bins)
    np.testing.assert_almost_equal(x.imag, 0, err_msg='Time-domain signal is not real-valued')
    # imaginary components of x are all less than 1e-17, so just discard them
    x = x.real

    # concatenate x's back to back to desired length
    c = np.tile(x, int(np.ceil(size/N)))
    c = c[:size]

    return c


if __name__ == '__main__':
    main()
