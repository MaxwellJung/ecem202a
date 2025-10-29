/* From https://github.com/espressif/esp-dsp/issues/25, provided by user @coderforlife */

#ifndef IFFT_H
#define IFFT_H

#include "esp_dsp.h"

void fft_init(int n) {
    dsps_fft2r_deinit_fc32();
    dsps_fft2r_init_fc32(NULL, n);
}

void fft(float *x, int n) {
    // Use dsps_fft2r_fc32_ansi because dsps_fft2r_fc32 is bugged.

    // dsps_fft2r_fc32 calls assembly version of fft optimized for
    // esp32s3, but the fft outputs are incorrect.
    // Likely bug introduced in one of the past commits
    // https://github.com/espressif/esp-dsp/blob/master/modules/fft/float/dsps_fft2r_fc32_aes3_.S
    dsps_fft2r_fc32_ansi(x, n);
    dsps_bit_rev2r_fc32(x, n);
}

// complex conjugate of n complex numbers as pairs of floats (real, imag); operates in-place
void _conj(float *x, int n) { for (int i = 0; i < n; i++) { x[2*i+1] = -x[2*i+1]; } }

void ifft(float *x, int n) {
    // Compute Inverse FFT using FFT
    // See https://dsp.stackexchange.com/questions/36082/calculate-ifft-using-only-fft

    _conj(x, n); // 1. Complex conjugate the given sequence that we want to inverse DFT
    fft(x, n);   // 2. Calculate its forward DFT
    _conj(x, n); // 3. Calculate complex conjugate of the result

    // Scale the output by 1/n
    float scale = 1.0f / n;
    for (int i = 0; i < n * 2; i++) { x[i] *= scale; }
}

#endif // IFFT_H
