#pragma once

/* One IMU reading: acceleration in g, angular rate in deg/s. */
typedef struct {
    float ax, ay, az;
    float gx, gy, gz;
} imu_sample_t;

#define IMU_SAMPLE_CHANNELS 6
