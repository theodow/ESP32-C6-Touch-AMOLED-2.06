#include <math.h>
#include "fall_config.h"
#include "fall_heuristic.h"

#define FREEFALL_SEARCH_SAMPLES FALL_MS_TO_SAMPLES(500)
#define POST_IMPACT_SAMPLES     FALL_MS_TO_SAMPLES(CONFIG_FALL_POST_IMPACT_MS)
#define SETTLE_SAMPLES          FALL_MS_TO_SAMPLES(500)

static float magnitude(const imu_sample_t *s)
{
    return sqrtf(s->ax * s->ax + s->ay * s->ay + s->az * s->az);
}

float fall_heuristic_score(const imu_sample_t *window, size_t len)
{
    if (len < FREEFALL_SEARCH_SAMPLES + POST_IMPACT_SAMPLES + 1) {
        return 0.0f;
    }

    /* Only peaks with enough history before and enough data after them are judged. */
    size_t first = FREEFALL_SEARCH_SAMPLES;
    size_t last = len - 1 - POST_IMPACT_SAMPLES;
    size_t peak = first;
    float peak_mag = 0.0f;
    for (size_t i = first; i <= last; i++) {
        float m = magnitude(&window[i]);
        if (m > peak_mag) {
            peak_mag = m;
            peak = i;
        }
    }
    if (peak_mag < CONFIG_FALL_IMPACT_MG / 1000.0f) {
        return 0.0f;
    }

#ifdef CONFIG_FALL_FREEFALL_REQUIRED
    float min_mag = INFINITY;
    for (size_t i = peak - FREEFALL_SEARCH_SAMPLES; i < peak; i++) {
        float m = magnitude(&window[i]);
        if (m < min_mag) {
            min_mag = m;
        }
    }
    if (min_mag > CONFIG_FALL_FREEFALL_MG / 1000.0f) {
        return 0.0f;
    }
#endif

    /* Skip the rebound right after the impact, then require little movement. */
    size_t still_start = peak + SETTLE_SAMPLES;
    if (still_start >= len) {
        return 0.0f;
    }
    size_t n = len - still_start;
    float sum = 0.0f, sum_sq = 0.0f;
    for (size_t i = still_start; i < len; i++) {
        float m = magnitude(&window[i]);
        sum += m;
        sum_sq += m * m;
    }
    float mean = sum / n;
    float var = sum_sq / n - mean * mean;
    float std = var > 0.0f ? sqrtf(var) : 0.0f;
    return std < CONFIG_FALL_STILL_STD_MG / 1000.0f ? 1.0f : 0.0f;
}
