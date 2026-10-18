#pragma once

#include <stddef.h>
#include "imu_sample.h"

/*
 * Classic three-phase rule on the acceleration magnitude: an impact peak,
 * optionally preceded by free fall, followed by stillness. Returns 1.0 for
 * a fall and 0.0 otherwise, so it plugs into the same score threshold as a
 * model. The window is oldest first.
 */
float fall_heuristic_score(const imu_sample_t *window, size_t len);
