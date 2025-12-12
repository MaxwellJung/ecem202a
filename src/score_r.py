""" Compute Root Mean Square Error of reflectance estimate

Usage:
    python3 ./src/score_r.py
"""

import numpy as np
from utils.video import load_video, load_image, export_frame, write_video

def main():
    edited_frame = load_image('in/irl/c3/iphone/fake_y_frame_0.png')
    y, _ = load_video('in/irl/c3/iphone/71.mov', downscale_factor=4, gamma_correction=2.2)
    original_frame = y[0]

    modified_uint8 = (255*edited_frame).astype(np.uint8)
    original_frame_uint8 = (255*original_frame).astype(np.uint8)
    mask = np.any(np.abs(modified_uint8-original_frame_uint8) > 0, axis=2)
    print(f'Modified {np.count_nonzero(mask)} pixels')

    attacks = ['basic', 'mult', 'sampling', 'sampling_mult']

    for a in attacks:
        actual, _ = load_video(f'results/71/{a}/r_estimate.mp4', downscale_factor=1, gamma_correction=2.2)

        expected = np.copy(actual).transpose(1,2,3,0).reshape(actual.shape[1]*actual.shape[2], actual.shape[3], actual.shape[0])
        expected[mask.flatten()] = 0
        expected = expected.reshape(actual.shape[1], actual.shape[2], actual.shape[3], actual.shape[0]).transpose(3, 0, 1, 2)

        # export_frame(expected, 0, f'out/{a}_expected_r.png')
        # write_video(expected, f'out/{a}_expected_r.mp4', 30, 2.2)

        score = rmse(expected, actual)
        print(f'RMSE of {a}: {score:.5f}')


def rmse(expected, actual):
    return np.sqrt(np.mean((expected-actual)**2))


if __name__ == '__main__':
    main()
