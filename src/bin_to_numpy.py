import struct
import numpy as np
import matplotlib.pyplot as plt

def main():
    # for little-endian uint16 use '<H' dtype
    # for little-endian float32 use '<f' dtype
    c = np.fromfile('./out/c_normalized.bin', dtype=np.dtype('<H'))
    print(c)

    # remove DC component
    c = c.astype(int)
    c -= int(np.mean(c))

    # check results
    fig = plt.figure(figsize=(16, 9))
    plt.hist(c, bins=100)
    plt.title("C Distribution")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.savefig("out/c_histogram.png")

    VIDEO_FPS = 30
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

if __name__ == '__main__':
    main()
