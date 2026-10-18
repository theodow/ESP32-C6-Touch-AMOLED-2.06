#pragma once

/*
 * Detector settings. On the device they come from Kconfig ("Fall detector"
 * menu); the defaults below let the detector build on a host for tests.
 */
#ifdef ESP_PLATFORM
#include "sdkconfig.h"
#endif

#ifndef CONFIG_FALL_SAMPLE_RATE_HZ
#define CONFIG_FALL_SAMPLE_RATE_HZ 50
#endif
#ifndef CONFIG_FALL_WINDOW_SAMPLES
#define CONFIG_FALL_WINDOW_SAMPLES 128
#endif
#ifndef CONFIG_FALL_STRIDE_SAMPLES
#define CONFIG_FALL_STRIDE_SAMPLES 25
#endif
#ifndef CONFIG_FALL_REFRACTORY_MS
#define CONFIG_FALL_REFRACTORY_MS 5000
#endif
#if !defined(CONFIG_FALL_BACKEND_HEURISTIC) && !defined(CONFIG_FALL_BACKEND_TFLM)
#define CONFIG_FALL_BACKEND_HEURISTIC 1
#endif
#ifndef CONFIG_FALL_IMPACT_MG
#define CONFIG_FALL_IMPACT_MG 2500
#endif
#if !defined(ESP_PLATFORM) && !defined(CONFIG_FALL_FREEFALL_REQUIRED)
#define CONFIG_FALL_FREEFALL_REQUIRED 1
#endif
#ifndef CONFIG_FALL_FREEFALL_MG
#define CONFIG_FALL_FREEFALL_MG 600
#endif
#ifndef CONFIG_FALL_POST_IMPACT_MS
#define CONFIG_FALL_POST_IMPACT_MS 1000
#endif
#ifndef CONFIG_FALL_STILL_STD_MG
#define CONFIG_FALL_STILL_STD_MG 150
#endif
#ifndef CONFIG_FALL_TFLM_THRESHOLD_PCT
#define CONFIG_FALL_TFLM_THRESHOLD_PCT 70
#endif
#ifndef CONFIG_FALL_TFLM_ARENA_KB
#define CONFIG_FALL_TFLM_ARENA_KB 24
#endif

#define FALL_MS_TO_SAMPLES(ms) (((ms) * CONFIG_FALL_SAMPLE_RATE_HZ + 999) / 1000)
