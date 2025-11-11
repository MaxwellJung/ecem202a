#!/usr/bin/env python3

import simulate_nci
import numpy as np
import cv2


def main():
    VIDEO_FPS = 30
    VIDEO_DURATION = 60*5
    ASPECT_RATIO = (16, 9)

    y, c, r, n = simulate_nci.generate_video(l=1, noise_variance=0.01**2,
                          width=1, height=1,
                          fps=VIDEO_FPS, duration=VIDEO_DURATION)
    
    l = 0.5 # base light intensity
    c_normalized = l + c
    print(f'{np.min(c_normalized)=}')
    print(f'{np.max(c_normalized)=}')
    c_normalized = np.clip(255*c_normalized, 0, 255)

    out = cv2.VideoWriter('in/irl/c.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, ASPECT_RATIO, False)
    for c_n in c_normalized[:, 0, 0, 0]:
        out.write(np.full(ASPECT_RATIO, c_n, dtype=np.uint8).T)
    out.release()

    np.save('in/irl/c', c[:, 0, 0, 0])

if __name__ == '__main__':
    main()
