/* From https://github.com/espressif/esp-dsp/issues/25, provided by user @coderforlife */

#ifndef IFFT_H
#define IFFT_H

#include "esp_dsp.h"

void rfft_init(int n_reals) {
    dsps_fft2r_deinit_fc32();
    dsps_fft4r_deinit_fc32();
    dsps_fft2r_init_fc32(NULL, n_reals >> 1);
    dsps_fft4r_init_fc32(NULL, n_reals >> 1);  // 4r is used for BOTH radix-2 and radix-4 FFTs!
}

void _rfft_core(float *x, int n) {
    if (n & 0xAAAAAAAA) {  // n is a power of 4
        dsps_fft4r_fc32(x, n);
        dsps_bit_rev4r_fc32(x, n);
    } else {
        dsps_fft2r_fc32(x, n);
        dsps_bit_rev2r_fc32(x, n);
    }
}

// complex conjugate of n complex numbers as pairs of floats (real, imag); operates in-place
void _conj(float *x, int n) { for (int i = 0; i < n; i++) { x[2*i+1] = -x[2*i+1]; } }

void rfft(float *x, int n) {
    // x has n real values or n/2 complex values, n must be a power of 2
    _rfft_core(x, n >> 1);
    dsps_cplx2real_fc32(x, n >> 1);
}

void irfft(float *x, int n) {
    // x has n complex values or n*2 real values, n must be a power of 2
    float re = x[0], im = x[1];

    _conj(x, n);               // conjugate the input for IFFT
    dsps_cplx2real_fc32(x, n); // reverse the post-processing (although not quite right)

    // Fix issues with `dsps_cplx2real_fc32()`
    x[0] = (re + im) * 0.5;  // fixes even values (and helps odd values) [DC component]
    x[1] = (im - re) * 0.5;  // fixes odd values [Nyquist component]

    _rfft_core(x, n);  // perform the FFT

    _conj(x, n); // fix the sign of every other real value

    // Scale the output by 1/n
    float scale = 1.0f / n;
    for (int i = 0; i < n * 2; i++) { x[i] *= scale; }
}

#endif // IFFT_H
