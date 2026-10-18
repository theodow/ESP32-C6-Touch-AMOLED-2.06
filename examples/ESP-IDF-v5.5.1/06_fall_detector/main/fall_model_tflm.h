#pragma once

#include <stddef.h>
#include "esp_err.h"
#include "imu_sample.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Loads the generated model and allocates the tensor arena. */
esp_err_t fall_model_init(void);
/* Runs the model on one window (oldest first). Returns the fall probability in [0, 1]. */
float fall_model_score(const imu_sample_t *window, size_t len);

#ifdef __cplusplus
}
#endif
