#pragma once

#include <stdbool.h>
#include <stdint.h>
#include "esp_err.h"
#include "imu_sample.h"

typedef struct {
    bool fall;          /* score crossed the threshold outside the refractory time */
    float score;        /* backend output in [0, 1] */
    uint32_t infer_us;  /* time spent in the backend */
} fall_result_t;

esp_err_t fall_detector_init(void);
/*
 * Adds one sample to the sliding window. Every CONFIG_FALL_STRIDE_SAMPLES
 * samples, once the window is full, the backend evaluates it; the function
 * then returns true and fills *out.
 */
bool fall_detector_push(const imu_sample_t *sample, fall_result_t *out);
const char *fall_detector_backend_name(void);
