
#ifndef FIFO_H
#define FIFO_H

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "config.h"
#include "string.h"

#define FIFO_DEPTH 4
#define SAMPLES_PER_BLOCK C_LENGTH
#define FIFO_RETRY_DELAY_MS 10000

static const char* FIFO_TAG = "fifo";
bool fifo_verbose = false;

typedef struct {
    uint16_t samples[FIFO_DEPTH][SAMPLES_PER_BLOCK];
    bool valid;
    uint32_t block_id;
} block_t;

typedef struct {
    block_t blocks[FIFO_DEPTH];
    uint32_t read_index;
    uint32_t write_index;
    uint32_t available_blocks;
    SemaphoreHandle_t mutex;
} fifo_t;

static fifo_t g_fifo;

// Initialize the FIFO buffer
void fifo_init(void){

    g_fifo.read_index = 0;
    g_fifo.write_index = 0;
    g_fifo.available_blocks = 0;
    g_fifo.mutex = xSemaphoreCreateMutex();

    for(int i=0; i<FIFO_DEPTH; i++){
        g_fifo.blocks[i].block_id = i;
        g_fifo.blocks[i].valid = false;
        memset(g_fifo.blocks[i].samples, 0, sizeof(uint16_t)*SAMPLES_PER_BLOCK);
    }
}

// Push block into the FIFO buffer
bool fifo_write(uint16_t *sample_write){

    if(xSemaphoreTake(g_fifo.mutex, portMAX_DELAY) == pdTRUE){

        if(g_fifo.blocks[g_fifo.write_index].valid){
            // Block is already valid, cannot overwrite, wait for it to be consumed by pwm
            if(fifo_verbose){
                ESP_LOGW(FIFO_TAG, "Write failed: block at index %u is still valid (FIFO full)", g_fifo.write_index);
            }
            xSemaphoreGive(g_fifo.mutex);
            return false;
        }

        memcpy(g_fifo.blocks[g_fifo.write_index].samples, sample_write, sizeof(uint16_t)*SAMPLES_PER_BLOCK);
        g_fifo.blocks[g_fifo.write_index].valid = true;

        g_fifo.write_index = (g_fifo.write_index + 1) % FIFO_DEPTH;
        g_fifo.available_blocks++;

        if(fifo_verbose){
            ESP_LOGI(FIFO_TAG, "Wrote block at index %u (available: %u, free: %u)", 
                     g_fifo.write_index, g_fifo.available_blocks + 1, FIFO_DEPTH - g_fifo.available_blocks);
        }

        xSemaphoreGive(g_fifo.mutex);
        return true;
    }

    return false;
}

// Get next block from the FIFO buffer
bool fifo_read(uint16_t *sample_read){

    if(xSemaphoreTake(g_fifo.mutex, portMAX_DELAY) == pdTRUE){

        if(!g_fifo.blocks[g_fifo.read_index].valid){
            // Block is not valid, nothing to read, wait for it to be generated
            if(fifo_verbose){
                ESP_LOGW(FIFO_TAG, "Read failed: block at index %u is not valid (FIFO empty)", g_fifo.read_index);
            }
            xSemaphoreGive(g_fifo.mutex);
            return false;
        }

        memcpy(sample_read, g_fifo.blocks[g_fifo.read_index].samples, sizeof(uint16_t)*SAMPLES_PER_BLOCK);

        g_fifo.blocks[g_fifo.read_index].valid = false;
        g_fifo.read_index = (g_fifo.read_index + 1) % FIFO_DEPTH;
        g_fifo.available_blocks--;

        if(fifo_verbose){
            ESP_LOGI(FIFO_TAG, "Read block at index %u (available: %u, free: %u)", 
                     g_fifo.read_index, g_fifo.available_blocks - 1, FIFO_DEPTH - g_fifo.available_blocks);
        }

        xSemaphoreGive(g_fifo.mutex);
        return true;
    }

    return false;

}

uint32_t fifo_availble_blocks(void){
    return g_fifo.available_blocks;
}

uint32_t fifo_free_blocks(void){
    return FIFO_DEPTH - g_fifo.available_blocks;
}

#endif