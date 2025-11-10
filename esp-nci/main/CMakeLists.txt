set(srcs "main.c" "sd.c" "nci.h" "ifft.h" "sd.h")

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_driver_mcpwm
    REQUIRES esp-dsp esp_timer fatfs sd_card
)