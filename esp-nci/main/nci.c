#include "nci.h"

#include <float.h>
#include <math.h>
#include "esp_random.h"

#include "ifft.h"

void _normalize_code(float* c);

const char* TAG = "nci";

// input to FFT must be aligned to 64-bit (8 byte) boundary
__attribute__((aligned(8)))
float _code_signal[CODE_SIGNAL_LENGTH*2];

float* generate_random_signal(float f_m, float f_s)
{
    float nyquist_freq = f_s/2;
    uint16_t valid_bins = CODE_SIGNAL_LENGTH/2 * (f_m/nyquist_freq);

    // randomly generate lower half of freq bins
    for (int i = 2; i < (valid_bins*2)+2; i += 2)
    {
        float magnitude = (float) esp_random() / UINT32_MAX;
        float phase = (float) esp_random() / UINT32_MAX * (2*M_PI);
        _code_signal[i] = magnitude * cos(phase);                          // real component
        _code_signal[i+1] = magnitude * sin(phase);                        // imaginary component
    }
    // DC component
    _code_signal[0] = 0;
    _code_signal[1] = 0;
    // freq components outside maximum bandwidth
    for (uint16_t i = (valid_bins*2)+2; i < CODE_SIGNAL_LENGTH; ++i)
        _code_signal[i] = 0;

    // set upper half of freq bins as mirrored and conjugate version of lower bins
    for (uint16_t i = 2; i < CODE_SIGNAL_LENGTH; i += 2)
    {
        _code_signal[(CODE_SIGNAL_LENGTH*2)-i] = _code_signal[i];
        _code_signal[(CODE_SIGNAL_LENGTH*2)-i+1] = _code_signal[i+1];
    }
    for (uint16_t i = CODE_SIGNAL_LENGTH+2; i < CODE_SIGNAL_LENGTH*2; i += 2)
    {
        _code_signal[i+1] = -_code_signal[i+1];
    }
    // Nyquist component
    _code_signal[CODE_SIGNAL_LENGTH] = 0;
    _code_signal[CODE_SIGNAL_LENGTH+1] = 0;

    // generate real-valued signal in time domain
    fft_init(CODE_SIGNAL_LENGTH);
    ifft(_code_signal, CODE_SIGNAL_LENGTH);

    _normalize_code(_code_signal);

    return _code_signal;
}

void _normalize_code(float* c)
{
    float c_max = 0;
    float c_min = FLT_MAX;

    for (size_t i = 0; i < 2*CODE_SIGNAL_LENGTH; ++i)
    {
        c_max = c[i] > c_max ? c[i] : c_max;
        c_min = c[i] < c_min ? c[i] : c_min;
    }

    float c_magnitude_max = fabs(c_max) > fabs(c_min) ? fabs(c_max) : fabs(c_min);

    for (size_t i = 0; i < 2*CODE_SIGNAL_LENGTH; ++i)
    {
        c[i] /= c_magnitude_max;
    }
}
