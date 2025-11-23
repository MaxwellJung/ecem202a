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
from utils.filters import bilateral_filter_1d

# Check if GPU is available (CUDA for NVIDIA, MPS for Apple Silicon)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def main():
    ###############################################################################
    # Load data
    ###############################################################################

    C_ARRAY_FILE = 'in/irl/c.npy'
    print(f'Loading NCI array {C_ARRAY_FILE}')
    c = np.load(C_ARRAY_FILE)
    C_SAMPLE_RATE = 30

    Y_VIDEO_FILE = 'in/irl/iphone/y2.MOV'
    print(f'Loading video file {Y_VIDEO_FILE}')
    y, VIDEO_FPS = load_video(Y_VIDEO_FILE)
    Y_SAMPLE_RATE = VIDEO_FPS

    # Simulate malicious video cut
    y = np.concatenate((y[:y.shape[0]//4], y[2*y.shape[0]//4:]))
    # Simulate malicious photoshop (edit in a grey square in the middle)
    y[0:y.shape[0]//8, y.shape[1]//4:y.shape[1]//2, y.shape[2]//4:y.shape[2]//2] = 128

    # export edited video
    out = cv2.VideoWriter('out/y_edited.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (y.shape[2], y.shape[1]), True)
    for frame in (y).clip(min=0, max=255):
        out.write(frame.astype(np.uint8))
    out.release()

    ###############################################################################
    # Pre-processing
    ###############################################################################

    # match c's sample rate to y
    C_SAMPLE_RATE = int(len(c)*(Y_SAMPLE_RATE/C_SAMPLE_RATE))/len(c)*C_SAMPLE_RATE
    print(f'Resampling c to {C_SAMPLE_RATE} Hz')
    c = resample(c, int(len(c)*(Y_SAMPLE_RATE/C_SAMPLE_RATE)))

    # downscale video
    DOWNSCALE_FACTOR = 4
    y = y[:, ::DOWNSCALE_FACTOR, ::DOWNSCALE_FACTOR, :]

    # linear gamma correction
    GAMMA = 2.2
    print(f'Undoing gamma={GAMMA} correction')
    y = (y/255)**GAMMA

    # bilateral filter
    print("Applying temporal bilateral filter")
    y = apply_temporal_bilateral_filter(y, spatial_sigma=2.0, range_sigma=0.05, window_size=11)

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

    plot_alignment_matrix(y, c, Y_SAMPLE_RATE, C_SAMPLE_RATE, output_path="out/align-mat.png")

    r = calculate_r(y, c, r_start=0, r_end=int(30*VIDEO_FPS), window_size=127, batch_size=5)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot()
    ax.imshow(r[0])
    ax.set_title("Code Image 0")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    fig.savefig("out/r_estimate.png")
    print("Saved code image out/r_estimate.png")
    plt.close(fig)

    out = cv2.VideoWriter('out/r_estimate.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (r.shape[2], r.shape[1]), True)
    for frame in (255*r).clip(min=0, max=255):
        out.write(frame.astype(np.uint8))
    out.release()
    print("Saved code video to out/r_estimate.mp4")


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    y = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))

    for i in range(frame_count):
        try:
            _, frame = cap.read()
            y[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            pass

    cap.release()

    return y, fps


def apply_temporal_bilateral_filter(y, spatial_sigma=2.0, range_sigma=0.05, window_size=11):
    """
    Apply bilateral filter to every pixel in y
    """
    filtered_y = np.apply_along_axis(
        bilateral_filter_1d, 1, 
        y.reshape((y.shape[0],y.shape[1]*y.shape[2]*y.shape[3])),
        spatial_sigma, range_sigma, window_size).reshape((y.shape[0],y.shape[1],y.shape[2],y.shape[3]))

    return filtered_y


def get_alignment_matrix(y, c, window_size=511, use_all_channels=True):
    """
    Improved alignment matrix with bilateral filtering option
    
    Args:
        y: Video frames (T, H, W, C)
        c: Code signal
        window_size: Window size for correlation
        use_all_channels: If True, use all color channels separately
    """
    if use_all_channels:
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
    
    padded_c = np.pad(c, (window_size//2, window_size//2))
    padded_y = np.pad(y_global, (window_size//2, window_size//2))
    
    c_windows = sliding_window_view(padded_c, window_size, writeable=True)
    y_windows = sliding_window_view(padded_y, window_size, writeable=True)
    
    # Normalize each window
    c_windows = (c_windows - np.mean(c_windows, axis=1, keepdims=True)) / \
                (np.std(c_windows, axis=1, keepdims=True) + 1e-6)
    y_windows = (y_windows - np.mean(y_windows, axis=1, keepdims=True)) / \
                (np.std(y_windows, axis=1, keepdims=True) + 1e-6)
    
    device_local = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    c_windows_torch = torch.from_numpy(c_windows).float().to(device_local)
    y_windows_torch = torch.from_numpy(y_windows).float().to(device_local)
    
    # Normalized cross-correlation
    align_mat = torch.matmul(y_windows_torch, c_windows_torch.T) / window_size
    align_mat = align_mat.T.cpu().numpy()
    
    return align_mat


def plot_alignment_matrix(y, c, Y_SAMPLE_RATE, C_SAMPLE_RATE, 
                                 title="Alignment Matrix",
                                 output_path="out/align-mat-styled.png",
                                 use_log_scale=False,
                                 brighten_path=True):
    """
    Plot alignment matrix with enhanced visualization similar to reference image
    
    Args:
        y: Video frames
        c: Code signal
        Y_SAMPLE_RATE: Video sample rate (fps)
        C_SAMPLE_RATE: Code signal sample rate
        title: Plot title
        output_path: Where to save the plot
        use_log_scale: Apply log scaling to enhance weak signals
        brighten_path: Highlight the strongest alignment path
    """
    align_mat = get_alignment_matrix(y, c)
    y_to_c = align_mat.argmax(axis=0)
    c_index_start = np.min(y_to_c)
    c_index_end = np.max(y_to_c) + 1
    cropped_align_mat = align_mat[c_index_start:c_index_end]
    
    if use_log_scale:
        # Apply log scaling to enhance weaker signals
        cropped_align_mat_vis = np.log10(cropped_align_mat - cropped_align_mat.min() + 1)
    else:
        cropped_align_mat_vis = cropped_align_mat.copy()
    
    # Normalize to 0-1
    mat_min = cropped_align_mat_vis.min()
    mat_max = cropped_align_mat_vis.max()
    cropped_align_mat_vis = (cropped_align_mat_vis - mat_min) / (mat_max - mat_min + 1e-6)
    
    if brighten_path:
        align_mat_image = np.zeros(cropped_align_mat_vis.shape)
        best_indices = np.argmax(cropped_align_mat_vis, axis=0)
        
        # Add the strongest correlation values along the path
        for i in range(cropped_align_mat_vis.shape[1]):
            idx = best_indices[i]
            align_mat_image[idx, i] = cropped_align_mat_vis[idx, i]
            
            if idx > 0:
                align_mat_image[idx-1, i] = cropped_align_mat_vis[idx-1, i] * 0.5
            if idx < cropped_align_mat_vis.shape[0] - 1:
                align_mat_image[idx+1, i] = cropped_align_mat_vis[idx+1, i] * 0.5
        
        mat_to_plot = align_mat_image
    else:
        mat_to_plot = cropped_align_mat_vis
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(
        mat_to_plot,
        origin='lower',
        extent=[
            (0-0.5)/Y_SAMPLE_RATE, 
            (len(y)-0.5)/Y_SAMPLE_RATE, 
            (c_index_start-0.5)/C_SAMPLE_RATE, 
            (c_index_end-0.5)/C_SAMPLE_RATE
        ],
        cmap='hot',
        interpolation='bilinear'
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Video Time (second)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Code Signal Time (second)", fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='Correlation Strength')
    cbar.ax.tick_params(labelsize=10)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved alignment matrix to {output_path}")
    plt.close(fig)


def calculate_r(y, c, y_to_c=None, r_start=0, r_end=None, window_size=127, batch_size=None):
    if y_to_c is None:
        align_mat = get_alignment_matrix(y, c, window_size=window_size)
        y_to_c = align_mat.argmax(axis=0)

    if r_end is None or r_end > len(y):
        r_end = len(y)

    if batch_size is None:
        batch_size = r_end - r_start

    # Create sliding windows
    padded_c = np.pad(c, (window_size//2, window_size//2))
    padded_y = np.pad(y, ((window_size//2, window_size//2), (0, 0), (0, 0), (0, 0)))
    c_prime_windows = sliding_window_view(padded_c, window_size, writeable=True)[y_to_c]
    y_windows = sliding_window_view(padded_y, (window_size, 1, 1, 1), writeable=True).squeeze()

    r = []
    for i in range(r_start, r_end, batch_size):
        r_i_start = i
        r_i_end = (i+batch_size) if (i+batch_size) < r_end else r_end

        # Select windows corresponding to current batch
        c_prime_windows_i = c_prime_windows[r_i_start:r_i_end]
        y_windows_i = y_windows[r_i_start:r_i_end]

        # rearrange axes to allow broadcasting
        y_windows_i = y_windows_i.transpose([1,2,3,0,4])

        # Convert to PyTorch tensors and move to GPU
        c_prime_windows_i = torch.from_numpy(c_prime_windows_i).float().to(device)
        y_windows_i = torch.from_numpy(y_windows_i).float().to(device)

        # Perform matrix multiplication on GPU/MPS (or by default your CPU)
        r_i = torch.divide(torch.sum(torch.multiply(c_prime_windows_i, y_windows_i), dim=4), 
                        torch.sum(torch.multiply(c_prime_windows_i, c_prime_windows_i), dim=1))
        r_i = r_i.cpu().numpy()
        r_i = r_i.transpose([3,0,1,2])

        r.append(r_i)

    r = np.concatenate(r)

    return r


if __name__ == '__main__':
    main()
