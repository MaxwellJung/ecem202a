#ifndef MAIN_H
#define MAIN_H

#include <stdint.h>
#include <stdbool.h>

#define C_LENGTH 1024
#define C_MAX_FREQ_HZ 9

#define PWM_GPIO            1
#define PWM_FREQUENCY_HZ    30
#define TIMER_RESOLUTION_HZ 1000000                                 // 1 MHz, 1 us per tick
#define TIMER_PERIOD_TICKS  TIMER_RESOLUTION_HZ / PWM_FREQUENCY_HZ  // ~33333 ticks

extern uint16_t c0_normalized[];
extern uint16_t c1_normalized[];

#define C0 0
#define C1 1
extern bool curr_c;

extern uint16_t* c_normalized;
extern uint16_t  c_index;

extern bool generate_new_c;

#endif // MAIN_H
