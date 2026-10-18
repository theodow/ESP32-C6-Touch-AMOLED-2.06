#pragma once

#include <stdbool.h>
#include <stddef.h>
#include "imu_sample.h"

/* Fixed-size ring buffer holding the most recent IMU samples. */
typedef struct {
    imu_sample_t *buf;
    size_t len;
    size_t head;  /* index the next sample is written to */
    size_t count; /* samples stored, saturates at len */
} imu_window_t;

void imu_window_init(imu_window_t *w, imu_sample_t *storage, size_t len);
void imu_window_push(imu_window_t *w, const imu_sample_t *s);
bool imu_window_full(const imu_window_t *w);
/* Copies the stored samples into dst, oldest first. Returns the number copied. */
size_t imu_window_copy(const imu_window_t *w, imu_sample_t *dst);
