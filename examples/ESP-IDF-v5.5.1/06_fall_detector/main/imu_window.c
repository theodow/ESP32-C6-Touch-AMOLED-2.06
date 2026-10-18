#include <string.h>
#include "imu_window.h"

void imu_window_init(imu_window_t *w, imu_sample_t *storage, size_t len)
{
    w->buf = storage;
    w->len = len;
    w->head = 0;
    w->count = 0;
}

void imu_window_push(imu_window_t *w, const imu_sample_t *s)
{
    w->buf[w->head] = *s;
    w->head = (w->head + 1) % w->len;
    if (w->count < w->len) {
        w->count++;
    }
}

bool imu_window_full(const imu_window_t *w)
{
    return w->count == w->len;
}

size_t imu_window_copy(const imu_window_t *w, imu_sample_t *dst)
{
    size_t start = (w->head + w->len - w->count) % w->len;
    size_t first = w->len - start;
    if (first > w->count) {
        first = w->count;
    }
    memcpy(dst, &w->buf[start], first * sizeof(imu_sample_t));
    memcpy(dst + first, w->buf, (w->count - first) * sizeof(imu_sample_t));
    return w->count;
}
