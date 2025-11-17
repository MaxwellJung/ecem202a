#ifndef NCI_H
#define NCI_H

#define CODE_SIGNAL_LENGTH  1024

#define C_MAX_FREQ_HZ   9
#define C_SAMPLE_RATE   120

float* generate_random_signal(float f_m, float f_s);

// void normalize_code(float* c, uint16_t* c_normalized, uint16_t range, uint16_t N);

#endif // NCI_H
