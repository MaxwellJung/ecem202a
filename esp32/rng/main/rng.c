#include <stdio.h>
#include "esp_system.h"
#include "esp_random.h"

void app_main(void) {
    
    // Method 1: Simple random number
    printf("Random 32-bit: 0x%08lX\n", esp_random());
    
    // Method 2: Fill buffer with random data
    uint8_t random_buffer[16];
    esp_fill_random(random_buffer, sizeof(random_buffer));
    
    printf("Random bytes: ");
    for (int i = 0; i < sizeof(random_buffer); i++) {
        printf("%02X ", random_buffer[i]);
    }
    printf("\n");
}