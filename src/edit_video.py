"""Edit video using python

Usage:
    python3 ./src/edit_video.py
"""

import numpy as np
import cv2
from utils.video import load_video, write_video, export_frame, load_image

def main():
    Y_VIDEO_FILE = 'in/irl/c2/iphone/38.MOV'
    print(f'Loading video file {Y_VIDEO_FILE}')
    y, VIDEO_FPS = load_video(Y_VIDEO_FILE, downscale_factor=4, gamma_correction=2.2)
    export_frame(y, 0, 'out/true_y_frame_0.png')

    # y_edited = basic_edit(y, reference=load_image('in/irl/c2/iphone/fake_y_frame_0.png'))
    y_edited = scaling_attack(y, reference=load_image('in/irl/c2/iphone/fake_y_frame_0.png'))

    # export edited video
    write_video(y_edited, 'out/fake_y.mp4', VIDEO_FPS, gamma=2.2)


def basic_edit(y, reference=None):
    if reference is None:
        # Simulate malicious video cut
        y_edited = np.concatenate((y[:y.shape[0]//4], y[2*y.shape[0]//4:]))
        # Simulate malicious photoshop (edit in a white square in the middle)
        y_edited[0:y_edited.shape[0]//8, y_edited.shape[1]//4:y_edited.shape[1]//2, y_edited.shape[2]//4:y_edited.shape[2]//2] = 1
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            reference_uint8 = (255*reference).astype(np.uint8)
            y_uint8 = (255*y[0]).astype(np.uint8)
            modified_pixels = np.any(np.abs(reference_uint8-y_uint8) > 0, axis=2)
            print(f'Modified {np.count_nonzero(modified_pixels)} pixels')
            # debug
            print(f'{np.transpose(modified_pixels.nonzero())}')
            y_edited = np.copy(y)
            y_edited[:, modified_pixels] = reference[modified_pixels]

    return y_edited


def scaling_attack(y, reference=None):
    if reference is None:
        y_edited = y
        # Multiply top left quadrant by random alpha
        region = y_edited[:, :y_edited.shape[1]//2, :y_edited.shape[2]//2]
        alpha = np.random.uniform(0/np.max(region), 1/np.max(region), (region.shape[1], region.shape[2], 3))
        y_edited[:, :y_edited.shape[1]//2, :y_edited.shape[2]//2] = \
            alpha*region
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            reference_uint8 = (255*reference).astype(np.uint8)
            y_uint8 = (255*y[0]).astype(np.uint8)
            mask = np.where(y_uint8==0, 1, reference_uint8/y_uint8)
            modified_pixels = np.any(mask!=1, axis=2)
            print(f'Modified {np.count_nonzero(modified_pixels)} pixels')
            # debug
            print(f'{np.transpose(modified_pixels.nonzero())}')
            y_edited = mask*y
    
    return y_edited


if __name__ == '__main__':
    main()
