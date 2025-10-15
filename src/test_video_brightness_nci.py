#!/usr/bin/env python3

import nci
import numpy as np
import cv2


def main():
    VIDEO_FPS = 30
    VIDEO_DURATION = 10
    ASPECT_RATIO = (16, 9)

    c, y = nci.generate_video(l=1, r=1, noise_variance=0.01, width=1, height=1, fps=VIDEO_FPS, duration=VIDEO_DURATION)

    min_c = np.min(c)
    max_c = np.max(c)
    c_normalized = np.round(((c - min_c) / (max_c - min_c)) * 255)

    out = cv2.VideoWriter('out/code.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, ASPECT_RATIO, False)
    for c_n in c_normalized:
        out.write(np.full(ASPECT_RATIO, c_n, dtype=np.uint8).T)
    out.release()


if __name__ == '__main__':
    main()
