/*
 * Host tests for the sliding window and the heuristic backend, driven by
 * synthetic 50 Hz traces. Build and run with `make` in this directory.
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "fall_config.h"
#include "fall_detector.h"
#include "imu_window.h"

static imu_sample_t still(void)
{
    imu_sample_t s = {0};
    s.az = 1.0f;
    return s;
}

static imu_sample_t mag(float g)
{
    imu_sample_t s = {0};
    s.az = g;
    return s;
}

/* Feeds n samples and returns how many evaluations flagged a fall. */
static int feed(imu_sample_t (*gen)(int), int n)
{
    int falls = 0;
    for (int i = 0; i < n; i++) {
        imu_sample_t s = gen(i);
        fall_result_t r;
        if (fall_detector_push(&s, &r) && r.fall) {
            falls++;
        }
    }
    return falls;
}

/* 3 s standing, 0.3 s free fall, impact, then lying still. */
static imu_sample_t gen_fall(int i)
{
    if (i < 150) return still();
    if (i < 165) return mag(0.2f);
    if (i < 168) return mag(4.0f);
    if (i < 175) return mag(i % 2 ? 1.6f : 0.5f);
    imu_sample_t s = {0};
    s.ax = 1.0f; /* lying on the side */
    return s;
}

/* Same impact, but the wearer keeps moving vigorously afterwards (e.g. clapping). */
static imu_sample_t gen_impact_then_moving(int i)
{
    if (i < 150) return still();
    if (i < 165) return mag(0.2f);
    if (i < 168) return mag(4.0f);
    return mag(1.0f + 0.8f * sinf(i * 1.3f));
}

/* Walking: periodic 0.6–1.6 g, never a free fall or a big impact. */
static imu_sample_t gen_walking(int i)
{
    return mag(1.1f + 0.5f * sinf(i * 2.0f * 3.14159f / 25.0f));
}

/* Sitting down hard: impact without free fall. */
static imu_sample_t gen_sit_down(int i)
{
    if (i < 150) return still();
    if (i < 153) return mag(2.8f);
    return still();
}

/* Falls repeating every 4 s (inside the refractory time) or every 8 s. */
static imu_sample_t gen_falls_4s(int i)
{
    return gen_fall(i % 200);
}

static imu_sample_t gen_falls_8s(int i)
{
    return gen_fall(i % 400);
}

static void test_window_order(void)
{
    imu_sample_t storage[4], out[4];
    imu_window_t w;
    imu_window_init(&w, storage, 4);
    for (int i = 0; i < 6; i++) {
        imu_sample_t s = mag((float)i);
        imu_window_push(&w, &s);
    }
    assert(imu_window_full(&w));
    assert(imu_window_copy(&w, out) == 4);
    for (int i = 0; i < 4; i++) {
        assert(out[i].az == (float)(i + 2));
    }
    printf("ok  window keeps the newest samples, oldest first\n");
}

static void expect(const char *name, imu_sample_t (*gen)(int), int n, int want)
{
    fall_detector_init();
    int got = feed(gen, n);
    printf("%s  %-36s falls=%d (want %d)\n", got == want ? "ok " : "FAIL", name, got, want);
    assert(got == want);
}

int main(void)
{
    test_window_order();
    expect("fall with free fall and stillness", gen_fall, 400, 1);
    expect("impact followed by movement", gen_impact_then_moving, 400, 0);
    expect("walking", gen_walking, 1000, 0);
    expect("sitting down hard", gen_sit_down, 400, 0);
    expect("two falls 4 s apart", gen_falls_4s, 600, 1);
    expect("two falls 8 s apart", gen_falls_8s, 1000, 2);
    return 0;
}
