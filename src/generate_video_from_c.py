#!/usr/bin/env python3

import numpy as np
import cv2


def generate_video_from_c(c_array_file, VIDEO_FPS, ASPECT_RATIO):
    c_normalized = np.load(c_array_file)

    out = cv2.VideoWriter('out/c.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, ASPECT_RATIO, False)
    for sample in c_normalized:
        out.write(np.full(ASPECT_RATIO, np.round(sample*255), dtype=np.uint8).T)
    out.release()


def main():
    VIDEO_FPS = 30
    ASPECT_RATIO = (16, 9)

    c_array_file = 'out/extracted_c.npy'

    generate_video_from_c(c_array_file, VIDEO_FPS, ASPECT_RATIO)


if __name__ == '__main__':
    main()
