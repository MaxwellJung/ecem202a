#!/usr/bin/env python3

import numpy as np
from scipy.signal import resample, correlate
import matplotlib.pyplot as plt
from utils.video import load_video
from analyze import analyze_nci_video


def extract_c(y_video_file, downscale_factor=1, normalize_channels=True):
    y, VIDEO_FPS = load_video(y_video_file, downscale_factor=downscale_factor, gamma_correction=1)
    
    if normalize_channels:
        y_red = np.mean(y[:, :, :, 0], axis=(1, 2))
        y_green = np.mean(y[:, :, :, 1], axis=(1, 2))
        y_blue = np.mean(y[:, :, :, 2], axis=(1, 2))
        
        # Normalize each channel
        y_red = (y_red - np.mean(y_red)) / (np.std(y_red) + 1e-6)
        y_green = (y_green - np.mean(y_green)) / (np.std(y_green) + 1e-6)
        y_blue = (y_blue - np.mean(y_blue)) / (np.std(y_blue) + 1e-6)
        
        y_global = (y_red + y_green + y_blue) / 3.0
    else:
        y_global = np.mean(y, axis=(1, 2, 3))
        
        # Normalize
        y_global = (y_global - np.mean(y_global)) / (np.std(y_global) + 1e-6)
    
    return y_global, VIDEO_FPS


def clean(signal):
    return np.array([sample for sample in signal if np.abs(np.mean(signal) - sample) < 5*np.std(signal)])


def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


def align(signal1, signal2):
    correlation = correlate(signal1, signal2, mode='full')
    shift = correlation.argmax() - (len(signal2)-1)

    if shift > 0:
        aligned_signal1 = signal1[shift:]
        aligned_signal2 = signal2
    else:
        aligned_signal1 = signal1
        aligned_signal2 = signal2[-shift:]
    
    aligned_length = min(len(aligned_signal1), len(aligned_signal2))

    return aligned_signal1[:aligned_length], aligned_signal2[:aligned_length]


def rmse(signal1, signal2):
    return np.sqrt(np.mean((signal1 - signal2)**2))


def extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=1, normalize_channels=True):
    true_c = np.load(c_array_file)

    extracted_c, VIDEO_FPS = extract_c(y_video_file, downscale_factor=downscale_factor, normalize_channels=normalize_channels)
    Y_SAMPLE_RATE = VIDEO_FPS

    # match true_c's sample rate to the sample rate of extracted_c, which is extracted from and thus has the same sample rate as y
    C_SAMPLE_RATE = int(len(true_c)*(Y_SAMPLE_RATE/C_SAMPLE_RATE))/len(true_c)*C_SAMPLE_RATE
    print(f'Resampling true_c to {C_SAMPLE_RATE} Hz')
    true_c = resample(true_c, int(len(true_c)*(Y_SAMPLE_RATE/C_SAMPLE_RATE)))

    extracted_c = clean(extracted_c)

    true_c_normalized = normalize(true_c)
    extracted_c_normalized = normalize(extracted_c)

    aligned_true_c_normalized, aligned_extracted_c_normalized = align(true_c_normalized, extracted_c_normalized)

    print(f'RMSE: {rmse(aligned_true_c_normalized, aligned_extracted_c_normalized)}')
    
    SAMPLE_PERIOD = 1 / Y_SAMPLE_RATE
    time_range = np.arange(0, SAMPLE_PERIOD*50, SAMPLE_PERIOD)
    plt.figure(figsize=(16, 9))
    plt.plot(time_range, aligned_true_c_normalized[:50], label='True C')
    plt.plot(time_range, aligned_extracted_c_normalized[:50], label='Extracted C')
    plt.title('True vs. Extracted C Signal (50 Samples)')
    plt.xlabel('Time (s)')
    plt.ylabel('(Normalized) Amplitude')
    plt.legend()
    plt.savefig('out/compare_extracted_c.png')

    np.save('out/true_c.npy', aligned_true_c_normalized)
    np.save('out/extracted_c.npy', aligned_extracted_c_normalized)
    np.save('out/extracted_c_sample_rate.npy', C_SAMPLE_RATE)
    return aligned_true_c_normalized, aligned_extracted_c_normalized, C_SAMPLE_RATE



def experiment1():
    y_video_file = 'in/irl/c1/c.mp4'
    c_array_file = 'in/irl/c1/c.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file)


def experiment2():
    y_video_file = 'in/irl/c1/iphone/c_paper.MOV'
    c_array_file = 'in/irl/c1/c.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment2_base_alignmat():
    y_video_file = 'in/irl/c1/iphone/c_paper.MOV'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment2_extracted_alignmat():
    y_video_file = 'in/irl/c1/iphone/c_paper.MOV'
    c_array_file = 'out/extracted_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')

    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3():
    y_video_file = 'in/irl/c1/iphone/y2.MOV'
    c_array_file = 'in/irl/c1/c.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment3_base_alignmat():
    y_video_file = 'in/irl/c1/iphone/y2.MOV'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3_extracted_alignmat():
    y_video_file = 'in/irl/c1/iphone/y2.MOV'
    c_array_file = 'out/extracted_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')

    analyze_nci_video(c_array_file, y_video_file, downscale=5, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3_2_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c1/reconstructed/reconstructed_y.mov'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=2, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3_2_2_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c1/reconstructed/reconstructed_y1.mov'
    c_array_file = 'in/irl/c1/c.npy'
    C_SAMPLE_RATE = 30
    
    analyze_nci_video(c_array_file, y_video_file, downscale=2, C_SAMPLE_RATE=C_SAMPLE_RATE)

def experiment3_3_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c1/reconstructed/reconstructed_y2.MOV'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3_4_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c1/reconstructed/reconstructed_y.MOV'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)

def experiment3_4_2_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c1/reconstructed/reconstructed_y.MOV'
    c_array_file = 'in/irl/c1/c.npy'
    C_SAMPLE_RATE = 30
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment4():
    y_video_file = 'in/irl/c2/iphone/38.mov'
    c_array_file = 'in/irl/c2/c.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment4_base_alignmat():
    y_video_file = 'in/irl/c2/iphone/38.mov'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment4_extracted_alignmat():
    y_video_file = 'in/irl/c2/iphone/38.mov'
    c_array_file = 'out/extracted_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')

    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_1():
    y_video_file = 'in/irl/c4/iphone/38_0.5c.mov'
    c_array_file = 'in/irl/c4/c_0.5.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment5_1_base_alignmat():
    y_video_file = 'in/irl/c4/iphone/38_0.5c.mov'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_1_extracted_alignmat():
    y_video_file = 'in/irl/c4/iphone/38_0.5c.mov'
    c_array_file = 'out/extracted_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')

    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_2():
    y_video_file = 'in/irl/c4/iphone/38_0.25c.mov'
    c_array_file = 'in/irl/c4/c_0.25.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment5_2_base_alignmat():
    y_video_file = 'in/irl/c4/iphone/38_0.25c.mov'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_2_extracted_alignmat():
    y_video_file = 'in/irl/c4/iphone/38_0.25c.mov'
    c_array_file = 'out/extracted_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')

    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_3():
    y_video_file = 'in/irl/c4/iphone/38_0.125c.mov'
    c_array_file = 'in/irl/c4/c_0.125.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment5_4():
    y_video_file = 'in/irl/c4/iphone/38_0.0625c.mov'
    c_array_file = 'in/irl/c4/c_0.0625.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment5_4_2_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c4/reconstructed/reconstructed_y.mov'
    c_array_file = 'out/extracted_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_4_3_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c4/reconstructed/reconstructed_y.mov'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_4_4_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c4/reconstructed/reconstructed_y.mov'
    c_array_file = 'in/irl/c4/c_0.0625.npy'
    C_SAMPLE_RATE = 30
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)



if __name__ == '__main__':
    experiment5_4_4_alignmat_on_reconstructed_signal()
