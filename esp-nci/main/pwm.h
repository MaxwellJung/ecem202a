#ifndef PWM_H
#define PWM_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define PWM_BUFFER_COUNT  2

void setup_pwm(void);

uint16_t* update_pwm_buffer();

uint16_t* get_pwm_buffer(size_t buffer_index);

#endif // PWM_H
