#include "pwm.h"

#include "driver/mcpwm_prelude.h"
#include "esp_attr.h"

#include "nci.h"

#define PWM_GPIO            1
#define PWM_FREQUENCY_HZ    C_SAMPLE_RATE
#define TIMER_RESOLUTION_HZ 1000000                                 // 1 MHz, 1 us per tick
#define TIMER_PERIOD_TICKS  (TIMER_RESOLUTION_HZ / PWM_FREQUENCY_HZ)  // ~33333 ticks

mcpwm_timer_handle_t    s_timer;
mcpwm_oper_handle_t     s_operator;
mcpwm_cmpr_handle_t     s_comparator;
mcpwm_gen_handle_t      s_generator;

uint16_t _pwm_buffer[PWM_BUFFER_COUNT][CODE_SIGNAL_LENGTH];

// variables for reading buffer
volatile size_t _last_read_buffer_index;
volatile size_t _read_buffer_index;
volatile size_t _read_sample_index;

static bool IRAM_ATTR on_timer_empty_cb(mcpwm_timer_handle_t timer,
                                        const mcpwm_timer_event_data_t* edata,
                                        void* user_ctx)
{
    (void)timer; (void)edata; (void)user_ctx;

    mcpwm_comparator_set_compare_value(s_comparator, _pwm_buffer[_read_buffer_index][(_read_sample_index++)]);

    _last_read_buffer_index = _read_buffer_index;

    if (_read_sample_index == CODE_SIGNAL_LENGTH) {
        _read_buffer_index = (_read_buffer_index+1)%PWM_BUFFER_COUNT;
        _read_sample_index = 0;
    }

    return false;
}

uint16_t* _overwrite_pwm_buffer(size_t buffer_index) {
    float* code_signal = generate_random_signal(C_MAX_FREQ_HZ, C_SAMPLE_RATE);
    for (int i = 0; i < CODE_SIGNAL_LENGTH; ++i) {
        _pwm_buffer[buffer_index][i] = (uint16_t) (TIMER_PERIOD_TICKS*(code_signal[2*i]+1)/2.0);
    }

    return _pwm_buffer[buffer_index];
}

void setup_pwm(void)
{
    for (int i = 0; i < PWM_BUFFER_COUNT; ++i) {
        _overwrite_pwm_buffer(i);
    }

    // reset read indices
    _last_read_buffer_index = 0;
    _read_buffer_index = 0;
    _read_sample_index = 0;

    /* Create timer and operator */
    mcpwm_timer_config_t timer_config = {
        .group_id = 0,
        .clk_src = MCPWM_TIMER_CLK_SRC_DEFAULT,
        .resolution_hz = TIMER_RESOLUTION_HZ,
        .period_ticks = TIMER_PERIOD_TICKS,
        .count_mode = MCPWM_TIMER_COUNT_MODE_UP,
    };
    ESP_ERROR_CHECK(mcpwm_new_timer(&timer_config, &s_timer));

    mcpwm_operator_config_t operator_config = {
        .group_id = 0, // operator must be in the same group to the timer
    };
    ESP_ERROR_CHECK(mcpwm_new_operator(&operator_config, &s_operator));

    /* Connect timer and operator */
    ESP_ERROR_CHECK(mcpwm_operator_connect_timer(s_operator, s_timer));

    /* Create comparator and generator from the operator */
    mcpwm_comparator_config_t comparator_config = {
        .flags.update_cmp_on_tez = true,
    };
    ESP_ERROR_CHECK(mcpwm_new_comparator(s_operator, &comparator_config, &s_comparator));

    mcpwm_generator_config_t generator_config = {
        .gen_gpio_num = PWM_GPIO,
    };
    ESP_ERROR_CHECK(mcpwm_new_generator(s_operator, &generator_config, &s_generator));

    // set the initial compare value
    ESP_ERROR_CHECK(mcpwm_comparator_set_compare_value(s_comparator, 0));

    /* Set generator action on timer and compare event */
    // go high on counter empty
    ESP_ERROR_CHECK(mcpwm_generator_set_action_on_timer_event(s_generator,
                                                              MCPWM_GEN_TIMER_EVENT_ACTION(MCPWM_TIMER_DIRECTION_UP, MCPWM_TIMER_EVENT_EMPTY, MCPWM_GEN_ACTION_HIGH)));
    // go low on compare threshold
    ESP_ERROR_CHECK(mcpwm_generator_set_action_on_compare_event(s_generator,
                                                                MCPWM_GEN_COMPARE_EVENT_ACTION(MCPWM_TIMER_DIRECTION_UP, s_comparator, MCPWM_GEN_ACTION_LOW)));

    /* Register callback for when timer is empty */
    mcpwm_timer_event_callbacks_t cbs = {
        .on_empty = on_timer_empty_cb,
    };
    ESP_ERROR_CHECK(mcpwm_timer_register_event_callbacks(s_timer, &cbs, NULL));

    /* Enable and start timer */
    ESP_ERROR_CHECK(mcpwm_timer_enable(s_timer));
    ESP_ERROR_CHECK(mcpwm_timer_start_stop(s_timer, MCPWM_TIMER_START_NO_STOP));
}

uint16_t* update_pwm_buffer() {
    if (_read_buffer_index != _last_read_buffer_index) {
        return _overwrite_pwm_buffer(_last_read_buffer_index);
    } else {
        return NULL;
    }
}

uint16_t* get_pwm_buffer(size_t buffer_index) {
    return _pwm_buffer[buffer_index];
}
