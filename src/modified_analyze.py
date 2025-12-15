import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import torch
from scipy.signal import resample
from utils.filters import apply_temporal_bilateral_filter
from utils.video import load_video, write_video, export_frame
from analyze import get_alignment_matrix, plot_alignment_matrix, calculate_r


def analyze_nci_video(c_array_file = f'in/irl/c3/c.npy', y_video_file = f'in/irl/c3/iphone/71_edited_sampling_mult.mp4', downscale=1, w_a=511, w_r=127, C_SAMPLE_RATE=30):
    ###############################################################################
    # Load data
    ###############################################################################
    
    print(f'Loading NCI array {c_array_file}')
    c = np.load(c_array_file)
    # c = 0.5*c/np.max(c)
    # np.save(f'out/c', c)
    C_SAMPLE_RATE = C_SAMPLE_RATE

    print(f'Loading video file {y_video_file}')
    y, VIDEO_FPS = load_video(y_video_file, downscale_factor=downscale, gamma_correction=2.2)
    Y_SAMPLE_RATE = VIDEO_FPS

    ###############################################################################
    # Pre-processing
    ###############################################################################

    # match c's sample rate to y
    C_SAMPLE_RATE = int(len(c)*(Y_SAMPLE_RATE/C_SAMPLE_RATE))/len(c)*C_SAMPLE_RATE
    print(f'Resampling c to {C_SAMPLE_RATE} Hz')
    c = resample(c, int(len(c)*(Y_SAMPLE_RATE/C_SAMPLE_RATE)))

    # bilateral filter
    print("Applying temporal bilateral filter")
    y = apply_temporal_bilateral_filter(y, spatial_sigma=0.5, range_sigma=0.03, window_size=5)

    print(f"{y.shape=}")
    print(f"{c.shape=}")

    # export pre-processed video
    write_video(y, 'out/y_pre_processed.mp4', VIDEO_FPS)

    ###############################################################################
    # Plots for debugging
    ###############################################################################

    export_frame(y, 0, 'out/y_frame0.png')
    export_frame(y, 1, 'out/y_frame1.png')

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    y_center_pixel = y[:, y.shape[1]//2, y.shape[2]//2, :]
    ax.plot(y_center_pixel[:, 0], '.', color='red')
    ax.plot(y_center_pixel[:, 1], '.', color='green')
    ax.plot(y_center_pixel[:, 2], '.', color='blue')
    ax.set_title(f"Pixel({y.shape[2]//2},{y.shape[1]//2}) Intensity")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pixel Intensity")
    fig.savefig("out/y.png")
    print("Saved plot out/y.png")
    plt.close(fig)

    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot()
    # ax.hist(y.flatten(), bins=100)
    # ax.set_title("Y Distribution")
    # ax.set_xlabel("Pixel Intensity")
    # ax.set_ylabel("Count")
    # fig.savefig("out/y_histogram.png")
    # plt.close(fig)

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

    ###############################################################################
    # Start analysis
    ###############################################################################

    align_mat = get_alignment_matrix(y, c, window_size=w_a)
    y_to_c = align_mat.argmax(axis=0)
    c_index_start = np.min(y_to_c)
    c_index_end = np.max(y_to_c) + 1
    cropped_align_mat = align_mat[c_index_start:c_index_end]
    cropped_extent=[
        (0-0.5)/Y_SAMPLE_RATE, 
        (len(y)-0.5)/Y_SAMPLE_RATE, 
        (c_index_start-0.5)/C_SAMPLE_RATE, 
        (c_index_end-0.5)/C_SAMPLE_RATE
    ]
    plot_alignment_matrix(cropped_align_mat, cropped_extent, output_path="out/align-mat.png")

    r = calculate_r(y, c, y_to_c=y_to_c, r_start=0, r_end=int(30*VIDEO_FPS), window_size=w_r, batch_size=5)
    export_frame(r, 0, "out/r_estimate.png")
    write_video(r, 'out/r_estimate.mp4', VIDEO_FPS, gamma=2.2)

    return align_mat, r
