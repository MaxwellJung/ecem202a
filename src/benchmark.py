import multiprocessing
import numpy as np
from analyze import analyze_nci_video

def main():
    # window_sizes = [2**i-1 for i in range(5, 10)]

    # with multiprocessing.Pool(processes=1) as pool:
    #     results = pool.map(test_align_mat_win_size, window_sizes)

    # print(zip(window_sizes, results))

    for i in range(100, 700, 100):
        score = test_align_mat_win_size(i+1)
        print(f'{i+1} {score}')


def test_align_mat_win_size(w):
    print(f'Testing Alignment Matrix Window size {w}')
    align_mat, r = analyze_nci_video(
                        c_array_file = f'in/irl/c4/c_0.5.npy', 
                        y_video_file = f'in/irl/c4/iphone/38_0.5c.mov', 
                        downscale = 4,
                        w_a=w,
                        w_r=127)
    
    cor = np.mean(align_mat.max(axis=0))
    
    return cor

if __name__ == '__main__':
    main()
