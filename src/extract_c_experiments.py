#!/usr/bin/env python3

import numpy as np
from extract_c import extract_and_compare_c
from analyze import analyze_nci_video


def experiment1():
    y_video_file = 'in/irl/c1/c.mp4'
    c_array_file = 'in/irl/c1/c.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file)


def experiment2():
    y_video_file = 'in/irl/c1/iphone/c_paper.MOV'
    c_array_file = 'in/irl/c1/c.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment2_base_alignmat():
    y_video_file = 'in/irl/c1/iphone/c_paper.MOV'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3():
    y_video_file = 'in/irl/c1/iphone/y2.MOV'
    c_array_file = 'in/irl/c1/c.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment3_base_alignmat():
    y_video_file = 'in/irl/c1/iphone/y2.MOV'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3_2_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c1/reconstructed/reconstructed_y.mov'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=2, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3_2_2_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c1/reconstructed/reconstructed_y1.mov'
    c_array_file = 'in/irl/c1/c.npy'
    C_SAMPLE_RATE = 30
    
    analyze_nci_video(c_array_file, y_video_file, downscale=2, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3_3_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c1/reconstructed/reconstructed_y2.MOV'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3_4_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c1/reconstructed/reconstructed_y.MOV'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment3_4_2_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c1/reconstructed/reconstructed_y.MOV'
    c_array_file = 'in/irl/c1/c.npy'
    C_SAMPLE_RATE = 30
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment4():
    y_video_file = 'in/irl/c2/iphone/38.mov'
    c_array_file = 'in/irl/c2/c.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment4_base_alignmat():
    y_video_file = 'in/irl/c2/iphone/38.mov'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_1():
    y_video_file = 'in/irl/c4/iphone/38_0.5c.mov'
    c_array_file = 'in/irl/c4/c_0.5.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment5_1_base_alignmat():
    y_video_file = 'in/irl/c4/iphone/38_0.5c.mov'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_2():
    y_video_file = 'in/irl/c4/iphone/38_0.25c.mov'
    c_array_file = 'in/irl/c4/c_0.25.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment5_2_base_alignmat():
    y_video_file = 'in/irl/c4/iphone/38_0.25c.mov'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_3():
    y_video_file = 'in/irl/c4/iphone/38_0.125c.mov'
    c_array_file = 'in/irl/c4/c_0.125.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment5_4():
    y_video_file = 'in/irl/c4/iphone/38_0.0625c.mov'
    c_array_file = 'in/irl/c4/c_0.0625.npy'
    C_SAMPLE_RATE = 30

    extract_and_compare_c(c_array_file, C_SAMPLE_RATE, y_video_file, downscale_factor=4, normalize_channels=False)


def experiment5_4_2_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c4/reconstructed/reconstructed_y.mov'
    c_array_file = 'out/true_c.npy'
    C_SAMPLE_RATE = np.load('out/extracted_c_sample_rate.npy')
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


def experiment5_4_3_alignmat_on_reconstructed_signal():
    y_video_file = 'in/irl/c4/reconstructed/reconstructed_y.mov'
    c_array_file = 'in/irl/c4/c_0.0625.npy'
    C_SAMPLE_RATE = 30
    
    analyze_nci_video(c_array_file, y_video_file, downscale=4, C_SAMPLE_RATE=C_SAMPLE_RATE)


if __name__ == '__main__':
    experiment5_4_3_alignmat_on_reconstructed_signal()
