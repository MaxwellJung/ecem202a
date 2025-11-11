"""Analyze noise coded video (section 4)

Usage:
    python3 ./src/analyze.py
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import torch
import cv2
from scipy.signal import resample

# Check if GPU is available (CUDA for NVIDIA, MPS for Apple Silicon)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def main():
    C_ARRAY_FILE = 'in/irl/c.npy'
    C_SAMPLE_RATE = 30

    Y_VIDEO_FILE = 'in/irl/iphone/c_paper.mov'
    VIDEO_FPS = 30
    Y_SAMPLE_RATE = VIDEO_FPS

    c = np.load(C_ARRAY_FILE)
    y = load_video(Y_VIDEO_FILE)

    # downscale video
    DOWNSCALE_FACTOR = 4
    y = y[:, ::DOWNSCALE_FACTOR, ::DOWNSCALE_FACTOR, :]

    VIDEO_LENGTH = y.shape[0]
    VIDEO_HEIGHT = y.shape[1]
    VIDEO_WIDTH = y.shape[2]

    # match c's sample rate to y
    C_SAMPLE_RATE = int(len(c)*(Y_SAMPLE_RATE/C_SAMPLE_RATE))/len(c)*C_SAMPLE_RATE
    print(f'Resampling c to {C_SAMPLE_RATE} Hz')
    c = resample(c, int(len(c)*(Y_SAMPLE_RATE/C_SAMPLE_RATE)))

    # linear gamma correction
    GAMMA = 2.2
    print(f'Applying gamma={GAMMA} correction')
    y = (y/255)**(1/GAMMA)

    # # Simulate malicious video cut
    # y = np.concatenate((y[:VIDEO_LENGTH//4], y[2*VIDEO_LENGTH//4:]))
    # # Simulate malicious photoshop (edit in a grey square in the middle)
    # y[0:VIDEO_LENGTH//8, VIDEO_WIDTH//4:VIDEO_WIDTH//2, VIDEO_HEIGHT//4:VIDEO_HEIGHT//2] = 0.5

    # export edited video
    # VIDEO_FPS = 30
    # out = cv2.VideoWriter('out/y_edited.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (VIDEO_HEIGHT, VIDEO_WIDTH), True)
    # for frame in (255*y).clip(min=0, max=255):
    #     out.write(frame.astype(np.uint8))
    # out.release()

    print(f"{y.shape=}")
    print(f"{c.shape=}")

    ###############################################################################
    # Plots for debugging
    ###############################################################################

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.imshow(y[0])
    ax.set_title("Y Frame 0")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    fig.savefig("out/y_frame0.png")
    print("Saved plot out/y_frame0.png")
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.imshow(y[1])
    ax.set_title("Y Frame 1")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    fig.savefig("out/y_frame1.png")
    print("Saved plot out/y_frame1.png")
    plt.close(fig)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.plot(y[:, VIDEO_HEIGHT//2, VIDEO_WIDTH//2], '.')
    ax.set_title(f"Pixel({VIDEO_WIDTH//2},{VIDEO_HEIGHT//2}) Intensity")
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

    # fig = plt.figure(figsize=(16, 9))
    # ax = fig.add_subplot()
    # plt.hist(c, bins=100)
    # ax.set_title("C Distribution")
    # ax.set_xlabel("Amplitude")
    # ax.set_ylabel("Count")
    # fig.savefig("out/c_histogram.png")
    # plt.close(fig)

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

    align_mat = get_alignment_matrix(y, c)
    y_to_c = align_mat.argmax(axis=0)
    c_index_start = np.min(y_to_c)
    c_index_end = np.max(y_to_c)+1
    cropped_align_mat = align_mat[c_index_start:c_index_end]

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.matshow(cropped_align_mat, origin='lower', extent=[(0-0.5)/Y_SAMPLE_RATE, (len(y)-0.5)/Y_SAMPLE_RATE, (c_index_start-0.5)/C_SAMPLE_RATE, (c_index_end-0.5)/C_SAMPLE_RATE])
    ax.set_title("Alignment Matrix (Cropped)")
    ax.xaxis.tick_bottom()
    ax.set_xlabel("Y time (second)")
    ax.set_ylabel("C time (second)")
    fig.savefig("out/align-mat.png")
    print("Saved alignment matrix out/align-mat.png")
    plt.close(fig)

    # r = calculate_r(y, c)
    # fig = plt.figure(figsize=(16, 9))
    # ax.imshow(r[0])
    # ax.set_title("Code Image")
    # ax.set_xlabel("Width")
    # ax.set_ylabel("Height")
    # fig.savefig("out/r_estimate.png")
    # plt.close(fig)
    # print("Saved code image to out/r_estimate.png")

    # out = cv2.VideoWriter('out/r_estimate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (VIDEO_HEIGHT, VIDEO_WIDTH), True)
    # for frame in (255*r).clip(min=0, max=255):
    #     out.write(frame.astype(np.uint8))
    # out.release()
    # print("Saved code video to out/r_estimate.mp4")


def load_video(video_path):
    print(f'Loading video file {video_path}')
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    y = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

    for i in range(frame_count):
        try:
            _, y[i] = cap.read()
        except TypeError as e:
            pass

    cap.release()

    return y


def get_alignment_matrix(y, c, window_size=511):
    # make sure window_size is odd number
    # ideally some power of 2 minus 1

    # Create global vector y by reducing each frame to a single value
    y = np.mean(y, axis=(1,2,3))

    padded_c = np.pad(c, (window_size//2, window_size//2))
    padded_y = np.pad(y, (window_size//2, window_size//2))
    c_prime_windows = sliding_window_view(padded_c, window_size, writeable=True)
    y_windows = sliding_window_view(padded_y, window_size, writeable=True)

    # Convert to PyTorch tensors and move to GPU
    c_prime_windows_torch = torch.from_numpy(c_prime_windows).float().to(device)
    y_windows_torch = torch.from_numpy(y_windows).float().to(device)

    # Perform matrix multiplication on GPU/MPS (or by default your CPU)
    align_mat = torch.matmul(y_windows_torch, c_prime_windows_torch.T)
    align_mat = align_mat.T

    # Move result back to CPU and convert to NumPy
    align_mat = align_mat.cpu().numpy()

    return align_mat


def calculate_r(y, c, window_size=511):
    align_mat = get_alignment_matrix(y, c)
    y_to_c = align_mat.argmax(axis=0)

    padded_c = np.pad(c, (window_size//2, window_size//2))
    padded_y = np.pad(y, ((window_size//2, window_size//2), (0, 0), (0, 0), (0, 0)))
    c_prime_windows = sliding_window_view(padded_c, window_size, writeable=True)[y_to_c]
    y_windows = sliding_window_view(padded_y, (window_size, 1, 1, 1), writeable=True).squeeze()
    # rearrange axes to allow broadcasting
    y_windows = y_windows.transpose([1,2,3,0,4])

    # Convert to PyTorch tensors and move to GPU
    c_prime_windows_torch = torch.from_numpy(c_prime_windows).float().to(device)
    y_windows_torch = torch.from_numpy(y_windows).float().to(device)

    # Perform matrix multiplication on GPU/MPS (or by default your CPU)
    r = torch.divide(torch.sum(torch.multiply(c_prime_windows_torch, y_windows_torch), dim=4), 
                     torch.sum(torch.multiply(c_prime_windows_torch, c_prime_windows_torch), dim=1))
    r = r.cpu().numpy()
    r = r.transpose([3,0,1,2])

    return r


if __name__ == '__main__':
    main()
