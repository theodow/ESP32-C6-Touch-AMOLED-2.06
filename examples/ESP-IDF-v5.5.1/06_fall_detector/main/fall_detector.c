#include "fall_config.h"
#include "fall_detector.h"
#include "imu_window.h"
#include "fall_heuristic.h"
#if CONFIG_FALL_BACKEND_TFLM
#include "fall_model_tflm.h"
#endif

#ifdef ESP_PLATFORM
#include "esp_timer.h"
#define NOW_US() ((uint32_t)esp_timer_get_time())
#else
#define NOW_US() 0u
#endif

#define REFRACTORY_SAMPLES FALL_MS_TO_SAMPLES(CONFIG_FALL_REFRACTORY_MS)

static imu_sample_t s_storage[CONFIG_FALL_WINDOW_SAMPLES];
static imu_sample_t s_linear[CONFIG_FALL_WINDOW_SAMPLES];
static imu_window_t s_window;
static uint32_t s_since_eval;
static uint32_t s_refractory;

esp_err_t fall_detector_init(void)
{
    imu_window_init(&s_window, s_storage, CONFIG_FALL_WINDOW_SAMPLES);
    s_since_eval = 0;
    s_refractory = 0;
#if CONFIG_FALL_BACKEND_TFLM
    return fall_model_init();
#else
    return ESP_OK;
#endif
}

const char *fall_detector_backend_name(void)
{
#if CONFIG_FALL_BACKEND_TFLM
    return "TFLite Micro";
#else
    return "heuristic";
#endif
}

bool fall_detector_push(const imu_sample_t *sample, fall_result_t *out)
{
    imu_window_push(&s_window, sample);
    if (s_refractory > 0) {
        s_refractory--;
    }
    if (!imu_window_full(&s_window) || ++s_since_eval < CONFIG_FALL_STRIDE_SAMPLES) {
        return false;
    }
    s_since_eval = 0;

    size_t n = imu_window_copy(&s_window, s_linear);
    uint32_t start = NOW_US();
#if CONFIG_FALL_BACKEND_TFLM
    float score = fall_model_score(s_linear, n);
    float threshold = CONFIG_FALL_TFLM_THRESHOLD_PCT / 100.0f;
#else
    float score = fall_heuristic_score(s_linear, n);
    float threshold = 0.5f;
#endif
    out->infer_us = NOW_US() - start;
    out->score = score;
    out->fall = score >= threshold && s_refractory == 0;
    if (out->fall) {
        s_refractory = REFRACTORY_SAMPLES;
    }
    return true;
}
