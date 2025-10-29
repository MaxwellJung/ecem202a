idf_component_register(
    SRCS "main.c" "nci.h" "ifft.h"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_driver_mcpwm
    REQUIRES esp-dsp
)