#!/usr/bin/env python3
"""Generate code signal video that can be played on monitor

Usage:
    python3 ./src/code_signal_video.py
"""

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
    
    # normalize maximum magnitude of c to 1
    c = c/np.max(np.abs(c))

    # change c amplitude
    a = 2**-2
    c = a*c

    # base light intensity
    l = 0.5
    # monitor pixel value
    m = l + c
    print(f'{np.min(m)=}')
    print(f'{np.max(m)=}')
    assert(np.min(m) >= 0 and np.max(m) <= 1)

    out = cv2.VideoWriter(f'out/c_{a}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, ASPECT_RATIO, False)
    for m_sample in m[:, 0, 0, 0]:
        out.write(np.full(ASPECT_RATIO, 255*m_sample, dtype=np.uint8).T)
    out.release()

    np.save(f'out/c_{a}', c[:, 0, 0, 0])

if __name__ == '__main__':
    main()
