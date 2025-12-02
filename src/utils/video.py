"""Video utility functions

Usage:
    from utils.video import load_video, write_video
"""

import cv2
import numpy as np

def load_video(video_path, downscale_factor=1, gamma_correction=1):
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

    # downscale video
    if downscale_factor > 1:
        print(f'Downscaling video by {downscale_factor}')
        y = y[:, ::downscale_factor, ::downscale_factor, :]

    y = y/255

    # linear gamma correction
    if gamma_correction > 1:
        print(f'Undoing gamma={gamma_correction} correction')
        y = y**gamma_correction

    return y, fps


def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255
    
    return image


def write_video(video_array, video_path='out/y.mp4', fps=30, gamma=1):
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (video_array.shape[2], video_array.shape[1]), True)
    for frame in (255*(video_array**(1/gamma))).clip(min=0, max=255):
        rgb_frame = frame.astype(np.uint8)
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()
    print(f'Saved video to {video_path}')


def export_frame(y, frame_index, filename):
    first_frame = (255*y[frame_index]).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    print(f'Saved frame {frame_index} to {filename}')
