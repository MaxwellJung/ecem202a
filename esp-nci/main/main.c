#include "main.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "pwm.h"
#include "nci.h"
#include "sd.h"

static const char* TAG = "esp-nci";
static char c_file_path[] = "/c00_normalized.bin";

#define C_FILE_PATH_LENGTH 19
#define C_NUMBER_INDEX     3
#define MOUNT_POINT_LENGTH 4

#define C_NORMALIZED_SIZE_BYTES C_LENGTH*2

// input to FFT must be aligned to 64-bit (8 byte) boundary
__attribute__((aligned(8)))
static float c0[C_LENGTH*2];
uint16_t c0_normalized[C_LENGTH];

__attribute__((aligned(8)))
static float c1[C_LENGTH*2];
uint16_t c1_normalized[C_LENGTH];

bool curr_c;

uint16_t* c_normalized;
uint16_t  c_index;

bool generate_new_c;

static inline void increment_file_number(void)
{
    if (c_file_path[C_NUMBER_INDEX] == '9') {
        c_file_path[C_NUMBER_INDEX] = '0';
        ++c_file_path[C_NUMBER_INDEX-1];
    } else {
        ++c_file_path[C_NUMBER_INDEX];
    }
}

static inline void write_file(uint16_t* c)
{
    char write_path[MOUNT_POINT_LENGTH+C_FILE_PATH_LENGTH];
    strcpy(write_path, MOUNT_POINT);
    strcat(write_path, c_file_path);

    ESP_LOGI(TAG, "c_normalized: [%d %d %d %d ...]", c[0], c[1], c[2], c[3]);
    export_binary(write_path, (char*)c, C_NORMALIZED_SIZE_BYTES);
    
    increment_file_number();
}

void setup(void)
{
    generate_code(c0, C_MAX_FREQ_HZ, PWM_FREQUENCY_HZ, C_LENGTH);
    normalize_code(c0, c0_normalized, TIMER_PERIOD_TICKS, C_LENGTH);

    generate_code(c1, C_MAX_FREQ_HZ, PWM_FREQUENCY_HZ, C_LENGTH);
    normalize_code(c1, c1_normalized, TIMER_PERIOD_TICKS, C_LENGTH);

    setup_sd();
    write_file(c0_normalized);
    write_file(c1_normalized);
    cleanup_sd();

    curr_c = C0;
    c_normalized = c0_normalized;
    c_index = 0;
    generate_new_c = false;

    pwm_timer_setup();
}

void main_loop(void* pvParameters)
{
    (void)pvParameters;

    float* new_c;
    uint16_t* new_c_normalized;

    while (1) {
        if (generate_new_c) {
            if (curr_c == C1) {
                new_c = c0;
                new_c_normalized = c0_normalized;
            } else {
                new_c = c1;
                new_c_normalized = c1_normalized;
            }
            generate_code(new_c, C_MAX_FREQ_HZ, PWM_FREQUENCY_HZ, C_LENGTH);
            normalize_code(new_c, new_c_normalized, TIMER_PERIOD_TICKS, C_LENGTH);

            setup_sd();
            write_file(new_c_normalized);
            cleanup_sd();

            generate_new_c = false;
        }

        vTaskDelay(pdMS_TO_TICKS(10));  // 10 ms = 10000 us; watchdog will complain if too short
    }
}

void app_main(void)
{
    setup();
    xTaskCreate(main_loop, "Main Loop", 4096, NULL, 5, NULL);
    ESP_LOGI(TAG, "Initialization complete");
}
