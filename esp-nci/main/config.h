
#ifndef CONFIG_H
#define CONFIG_H

#define C_LENGTH 1024
#define C_MAX_FREQ_HZ 9

#define PWM_GPIO            1
#define PWM_FREQUENCY_HZ    30
#define TIMER_RESOLUTION_HZ 1000000                                 // 1 MHz, 1 us per tick
#define TIMER_PERIOD_TICKS  TIMER_RESOLUTION_HZ / PWM_FREQUENCY_HZ  // ~33333 ticks

#endif