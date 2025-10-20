"""Simulation of greyscale video in a static scene illuminated by one NCI light source.

Usage:
    python3 ./src/nci.py
"""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import cv2

rng = np.random.default_rng()

def main():
    # generate 60 second NCI video @ 30fps
    VIDEO_WIDTH = 16
    VIDEO_HEIGHT = 16
    VIDEO_FPS = 30
    VIDEO_DURATION = 60
    FRAME_COUNT = VIDEO_FPS*VIDEO_DURATION
    c, y = generate_video(l=128, r=1, noise_variance=1,
                          width=VIDEO_WIDTH, height=VIDEO_HEIGHT,
                          fps=VIDEO_FPS, duration=VIDEO_DURATION)

    np.save('out/c', c)
    np.save('out/y', y)

    out = cv2.VideoWriter('out/y.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT), False)
    for frame in y:
        out.write(frame.astype(np.uint8).T)
    out.release()

    # Plots for debugging
    t = np.arange(FRAME_COUNT)/VIDEO_FPS

    fig = plt.figure(figsize=(16, 9))
    plt.imshow(y[0], cmap='gray', vmin=0, vmax=255)
    plt.title("Y Frame 0")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.savefig("out/y_frame0.png")

    fig = plt.figure(figsize=(16, 9))
    plt.imshow(y[1], cmap='gray', vmin=0, vmax=255)
    plt.title("Y Frame 1")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.savefig("out/y_frame1.png")

    fig = plt.figure(figsize=(16, 9))
    plt.plot(t, y[:, 0, 0], '.')
    plt.title("Pixel(0,0) Intensity")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixel Intensity")
    plt.savefig("out/y.png")

    fig = plt.figure(figsize=(16, 9))
    plt.hist(y.flatten(), bins=100)
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


def generate_video(l: float, r: float, noise_variance: float, 
                   width: int, height: int, 
                   fps: int, duration: int) -> tuple[NDArray, NDArray]:
    """Model of multi-pixel video in a static scene with NCI

    Args:
        l (float): Power of light source
        r (float): Light transport coefficient
        noise_variance (float): Variance of random noise
        width (int): Video frame width in pixels
        height (int): Video frame height in pixels
        fps (int): Video frames per second.
        duration (int): Duration of video in seconds.

    Returns:
        tuple[NDArray, NDArray]: array representing noise coded light (c)  
        and pixel intensity over time (y)

        shape of c = (fps * duration)  
        shape of y = (fps * duration, width, height)
    """
    frame_count = fps*duration

    # noise coded illumination
    c = generate_nci(f_m=9, f_s=fps, size=frame_count)

    # noise from camera sensor (photon shot noise)
    n = rng.normal(loc=0, scale=np.sqrt(noise_variance), size=(frame_count, width, height))

    # equation 2 from paper
    # need to transpose n for numpy broadcasting to work
    y = (l+c)*r + n.T

    # transpose y so axes becomes (frame_index, width_index, height_index)
    return c, y.T


def generate_nci(f_m: float, f_s: float, size: int) -> NDArray:
    """Generate code signal; see section 5 from paper.
    Code signal should be random, noise-like, zero-mean, and uncorrelated with each other.

    Args:
        f_m (float): Maximum bandwidth of signal in Hz
        f_s (float): Sampling frequency in Hz
        size (int): Length of output array

    Returns:
        NDArray: Array representing noise coded light intensity over time
    """

    N = 2**8
    C_AMPLITUDE = 256
    # create c by concatenating copies of x's
    c = np.concat([generate_random_signal(f_m, f_s, N) for i in range(int(np.ceil(size/N)))])
    c = c[:size]
    c = C_AMPLITUDE*c

    return c


def generate_random_signal(f_m: float, f_s: float, N: int) -> NDArray:
    """ Generate random signal with maximum bandwidth of f_m.
    Create random discrete spectrum, then convert to time-domain signal using inverse FFT.
    Ensure time-domain signal is real by constructing spectrum such that lower and upper
    frequency bins are complex conjugate and mirrored versions of each other.

    Args:
        f_m (float): Maximum bandwidth of signal in Hz
        f_s (float): Sampling frequency in Hz
        N (int): Number of FFT bins

    Returns:
        NDArray: _description_
    """
    # If N is odd, first bin (DC component) is real
    # If N is even, first and middle bins (DC and Nyquist components) are real
    # For now, choose even N

    nyquist_freq = f_s/2
    valid_bins = int(N//2*(f_m/nyquist_freq))

    freq_bins = np.empty(N, dtype=complex)

    # randomly generate lower half of freq bins
    phases = rng.uniform(0, 2*np.pi, valid_bins)
    magnitudes = rng.uniform(0, 1, valid_bins)
    freq_bins[1:valid_bins+1] = magnitudes*np.exp(1j*phases)
    # set DC component to 0 (or other real value)
    freq_bins[0] = 0
    # set freq components outside maximum bandwidth to 0
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

    return x


if __name__ == '__main__':
    main()
