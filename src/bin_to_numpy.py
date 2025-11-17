import struct
import numpy as np
import matplotlib.pyplot as plt

def main():
    # for little-endian uint16 use '<H' dtype
    # for little-endian float32 use '<f' dtype
    c = np.fromfile('./in/irl/esp32/c.bin', dtype=np.dtype('<H'))
    print(np.info(c))
    print(c)

    # remove DC component
    c = c.astype(int)
    c -= int(np.mean(c))
    C_SAMPLE_RATE = 120

    # check results
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.step(np.arange(len(c))/C_SAMPLE_RATE, c, where='post')
    ax.set_title("Coded Light Signal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.savefig("out/c.png")
    print("Saved plot out/c.png")
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.hist(c, bins=100)
    ax.set_title("C Distribution")
    ax.set_xlabel("Amplitude")
    ax.set_ylabel("Count")
    fig.savefig("out/c_histogram.png")
    print("Saved plot out/c_histogram.png")
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.step(C_SAMPLE_RATE*np.arange(len(c))/len(c), np.abs(np.fft.fft(c)), where='mid')
    ax.set_title("Magnitude Spectrum of C")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    fig.savefig("out/c_spectrum_magnitude.png")
    print("Saved plot out/c_spectrum_magnitude.png")
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.step(C_SAMPLE_RATE*np.arange(len(c))/len(c), np.angle(np.fft.fft(c)), where='mid')
    ax.set_title("Phase Spectrum of C")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase")
    fig.savefig("out/c_spectrum_phase.png")
    print("Saved plot out/c_spectrum_phase.png")
    plt.close(fig)

if __name__ == '__main__':
    main()
