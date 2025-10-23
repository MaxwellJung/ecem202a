#ifndef NCI_H
#define NCI_H

#include "esp_dsp.h"
#include "ifft.h"
#include "esp_random.h"
#include <stdint.h>
#include <float.h>

void generate_code(float* freq_bins, float f_m, float f_s, uint16_t N)
{
    float nyquist_freq = f_s/2;
    uint16_t valid_bins = N/2 * (f_m/nyquist_freq);

    // randomly generate lower half of freq bins
    for (int i = 2; i < (valid_bins*2)+2; i += 2)
    {
        freq_bins[i] = (float) esp_random() / UINT32_MAX;       // real components
        freq_bins[i+1] = (float) esp_random() / UINT32_MAX;     // imaginary components
    }
    // DC component
    freq_bins[0] = 0;
    freq_bins[1] = 0;
    // freq components outside maximum bandwidth
    for (uint16_t i = (valid_bins*2)+2; i < N; ++i)
        freq_bins[i] = 0;

    // set upper half of freq bins as mirrored and conjugate version of lower bins
    for (uint16_t i = 2; i < N; i += 2)
    {
        freq_bins[(N*2)-i] = freq_bins[i];
        freq_bins[(N*2)-i+1] = freq_bins[i+1];
    }
    for (uint16_t i = N+2; i < N*2; i += 2)
    {
        freq_bins[i+1] = -freq_bins[i+1];
    }
    // Nyquist component
    freq_bins[N] = 0;
    freq_bins[N+1] = 0;

    // generate real-valued signal in time domain
    rfft_init(N);
    irfft(freq_bins, N);
    // separate real and imaginary components; real components are elements 0...N-1, imaginary components are elements N...2*N-1
    dsps_cplx2reC_fc32(freq_bins, N);
}

void normalize_code(float* c, uint16_t* c_normalized, uint16_t range, uint16_t N)
{
    float c_max = 0;
    float c_min = FLT_MAX;

    for (uint16_t i = 0; i < N; ++i)
    {
        c_max = c[i] > c_max ? c[i] : c_max;
        c_min = c[i] < c_min ? c[i] : c_min;
    }

    float c_range = c_max - c_min;

    for (uint16_t i = 0; i < N; ++i)
    {
        c_normalized[i] = ((c[i] - c_min) / c_range) * range;
    }
}

#endif // NCI_H
