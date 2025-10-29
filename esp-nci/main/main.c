#include "driver/mcpwm_prelude.h"
#include "esp_log.h"
#include "nci.h"

static const char* TAG = "esp-nci";

#define C_LENGTH 1024
#define C_MAX_FREQ_HZ 9

#define PWM_GPIO            1
#define PWM_FREQUENCY_HZ    30
#define TIMER_RESOLUTION_HZ 1000000                                 // 1 MHz, 1 us per tick
#define TIMER_PERIOD_TICKS  TIMER_RESOLUTION_HZ / PWM_FREQUENCY_HZ  // ~33333 ticks

static float    c[C_LENGTH*2];                                      // first half real, second half imaginary
static uint16_t c_normalized[C_LENGTH];
static uint16_t c_index;

static mcpwm_timer_handle_t s_timer;
static mcpwm_oper_handle_t  s_operator;
static mcpwm_cmpr_handle_t  s_comparator;
static mcpwm_gen_handle_t   s_generator;

static inline uint16_t get_next_duty(void)
{
    uint16_t c_value = c_normalized[c_index];
    if (++c_index == C_LENGTH)
        c_index = 0;
    return c_value;
}

static bool IRAM_ATTR on_timer_empty_cb(mcpwm_timer_handle_t timer,
                                        const mcpwm_timer_event_data_t* edata,
                                        void* user_ctx)
{
    (void)timer; (void)edata; (void)user_ctx;

    mcpwm_comparator_set_compare_value(s_comparator, get_next_duty());
    return false;
}

void timer_setup(void)
{
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

void app_main(void)
{
    generate_code(c, C_MAX_FREQ_HZ, PWM_FREQUENCY_HZ, C_LENGTH);
    normalize_code(c, c_normalized, TIMER_PERIOD_TICKS, C_LENGTH);
    timer_setup();

    ESP_LOGI(TAG, "Setup complete");
}
