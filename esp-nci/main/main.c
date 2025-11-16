#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "sd.h"
#include "nci.h"
#include "pwm.h"

static const char* TAG = "main";

void export_code_signal(uint16_t* c) {
    // write to disk
    ESP_LOGI(TAG, "code_signal: [%d %d %d ... %d %d %d]", c[0], c[1], c[2], c[CODE_SIGNAL_LENGTH-3], c[CODE_SIGNAL_LENGTH-2], c[CODE_SIGNAL_LENGTH-1]);
    export_binary(MOUNT_POINT"/c.bin", (char*)c, CODE_SIGNAL_LENGTH);
}

void setup(void) {
    setup_pwm();
    setup_sd();
    for (int i = 0; i < PWM_BUFFER_COUNT; ++i) {
        export_code_signal(get_pwm_buffer(i));
    }
    cleanup_sd();
}

void loop() {
    uint16_t* new_c = update_pwm_buffer();
    if (new_c) {
        setup_sd();
        export_code_signal(new_c);
        cleanup_sd();
    }

    vTaskDelay(pdMS_TO_TICKS(10));  // 10 ms = 10000 us; watchdog will complain if too short
}

void app_main(void) {
    setup();
    ESP_LOGI(TAG, "Setup complete");

    ESP_LOGI(TAG, "Entering main loop");
    while (true) {
        loop();
    }
}
