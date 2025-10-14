#!/usr/bin/env python3

import nci
import time
import numpy as np
import screen_brightness_control as sbc


def main():
    VIDEO_FPS = 30
    VIDEO_DURATION = 10

    c, y = nci.generate_video(l=1, r=1, noise_variance=0.01, fps=VIDEO_FPS, duration=VIDEO_DURATION)

    min_c = np.min(c)
    max_c = np.max(c)
    c_normalized = ((c - min_c) / (max_c - min_c)) * 100

    while True:
        start_time = time.perf_counter()
        for c_n in c_normalized:
            sbc.set_brightness(c_n)
        end_time = time.perf_counter()
        print(f'Elapsed time: {end_time-start_time:.6f} s')


if __name__ == '__main__':
    main()
