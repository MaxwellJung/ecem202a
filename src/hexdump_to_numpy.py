import struct
import numpy as np
import matplotlib.pyplot as plt

def main():
    c = hexdump_to_uint16_array('./out/c_normalized_hexdump.txt')
    # remove DC component
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

def hexdump_to_uint16_array(hexdump_filepath):
    # size of each array element
    bytes_per_element = 2
    with open(hexdump_filepath, 'r') as f:
        s = f.read().split()
        hex_str_list = ["".join(s[i:i+bytes_per_element]) for i in range(0, len(s), bytes_per_element)]
        uint16_list = [hex_to_uint16(hex_str) for hex_str in hex_str_list]
    
    return np.array(uint16_list)

def hex_to_uint16(hex_str):
    byte_array = bytes.fromhex(hex_str)
    float_num = struct.unpack('<H', byte_array)[0]
    return float_num

def hex_to_float(hex_str):
    byte_array = bytes.fromhex(hex_str)
    float_num = struct.unpack('<f', byte_array)[0]
    return float_num

if __name__ == '__main__':
    main()
