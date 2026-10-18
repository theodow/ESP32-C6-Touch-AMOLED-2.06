#include <math.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "lvgl.h"
#include "bsp/esp-bsp.h"
#include "bsp/display.h"
#include "qmi8658.h"
#include "fall_config.h"
#include "fall_detector.h"

static const char *TAG = "FALL_DETECTOR";

#define STANDARD_GRAVITY 9.80665f
#define RAD_TO_DEG       57.29578f
#define UI_UPDATE_EVERY  (CONFIG_FALL_SAMPLE_RATE_HZ / 5)

static lv_obj_t *s_status_label;
static lv_obj_t *s_alert;

static void alert_dismiss_cb(lv_event_t *e)
{
    lv_obj_add_flag(s_alert, LV_OBJ_FLAG_HIDDEN);
}

static void ui_create(void)
{
    lv_obj_t *scr = lv_screen_active();
    lv_obj_set_style_bg_color(scr, lv_color_black(), 0);

    lv_obj_t *title = lv_label_create(scr);
    lv_obj_set_style_text_font(title, &lv_font_montserrat_24, 0);
    lv_obj_set_style_text_color(title, lv_color_white(), 0);
    lv_label_set_text(title, "Fall Guardian");
    lv_obj_align(title, LV_ALIGN_TOP_MID, 0, 40);

    s_status_label = lv_label_create(scr);
    lv_obj_set_style_text_font(s_status_label, &lv_font_montserrat_16, 0);
    lv_obj_set_style_text_color(s_status_label, lv_color_white(), 0);
    lv_obj_set_style_text_align(s_status_label, LV_TEXT_ALIGN_CENTER, 0);
    lv_label_set_text(s_status_label, "Initializing QMI8658...");
    lv_obj_align(s_status_label, LV_ALIGN_CENTER, 0, 0);

    /* Full-screen alert, hidden until a fall is detected; a tap dismisses it. */
    s_alert = lv_obj_create(scr);
    lv_obj_remove_style_all(s_alert);
    lv_obj_set_size(s_alert, LV_PCT(100), LV_PCT(100));
    lv_obj_set_style_bg_color(s_alert, lv_palette_main(LV_PALETTE_RED), 0);
    lv_obj_set_style_bg_opa(s_alert, LV_OPA_COVER, 0);
    lv_obj_add_flag(s_alert, LV_OBJ_FLAG_CLICKABLE | LV_OBJ_FLAG_HIDDEN);
    lv_obj_add_event_cb(s_alert, alert_dismiss_cb, LV_EVENT_CLICKED, NULL);

    lv_obj_t *alert_title = lv_label_create(s_alert);
    lv_obj_set_style_text_font(alert_title, &lv_font_montserrat_40, 0);
    lv_obj_set_style_text_color(alert_title, lv_color_white(), 0);
    lv_obj_set_style_text_align(alert_title, LV_TEXT_ALIGN_CENTER, 0);
    lv_label_set_text(alert_title, "FALL\nDETECTED");
    lv_obj_align(alert_title, LV_ALIGN_CENTER, 0, -30);

    lv_obj_t *alert_hint = lv_label_create(s_alert);
    lv_obj_set_style_text_font(alert_hint, &lv_font_montserrat_24, 0);
    lv_obj_set_style_text_color(alert_hint, lv_color_white(), 0);
    lv_label_set_text(alert_hint, "Tap to dismiss");
    lv_obj_align(alert_hint, LV_ALIGN_BOTTOM_MID, 0, -60);
}

static void ui_show_alert(void)
{
    bsp_display_lock(0);
    lv_obj_remove_flag(s_alert, LV_OBJ_FLAG_HIDDEN);
    lv_obj_move_foreground(s_alert);
    bsp_display_unlock();
}

static void ui_update_status(float mag_g, const fall_result_t *last, uint32_t falls)
{
    char buf[160];
    snprintf(buf, sizeof(buf),
             "Monitoring (%s)\n\n"
             "|a| = %.2f g\n"
             "score = %.2f\n"
             "inference = %lu us\n"
             "falls = %lu",
             fall_detector_backend_name(), mag_g, last->score,
             (unsigned long)last->infer_us, (unsigned long)falls);
    if (bsp_display_lock(pdMS_TO_TICKS(20))) {
        lv_label_set_text(s_status_label, buf);
        bsp_display_unlock();
    }
}

static void imu_init(qmi8658_dev_t *dev)
{
    ESP_ERROR_CHECK(qmi8658_init(dev, bsp_i2c_get_handle(), QMI8658_ADDRESS_HIGH));
    qmi8658_set_accel_range(dev, QMI8658_ACCEL_RANGE_8G);
    qmi8658_set_accel_odr(dev, QMI8658_ACCEL_ODR_500HZ);
    qmi8658_set_accel_unit_mps2(dev, true);
    qmi8658_set_gyro_range(dev, QMI8658_GYRO_RANGE_512DPS);
    qmi8658_set_gyro_odr(dev, QMI8658_GYRO_ODR_1000HZ);
    qmi8658_set_gyro_unit_rads(dev, true);
    /* Accelerometer LPF as in 04_qmi8658 (0x03), plus the gyroscope LPF (bit 4). */
    qmi8658_write_register(dev, QMI8658_CTRL5, 0x13);
}

static void fall_detect_task(void *arg)
{
    qmi8658_dev_t *dev = (qmi8658_dev_t *)arg;
    qmi8658_data_t data;
    fall_result_t last = {0};
    uint32_t falls = 0;
    uint32_t n = 0;
    const TickType_t period = pdMS_TO_TICKS(1000 / CONFIG_FALL_SAMPLE_RATE_HZ);
    TickType_t wake = xTaskGetTickCount();

    while (1) {
        vTaskDelayUntil(&wake, period);
        if (qmi8658_read_sensor_data(dev, &data) != ESP_OK) {
            ESP_LOGW(TAG, "Failed to read sensor data");
            continue;
        }

        imu_sample_t s = {
            .ax = data.accelX / STANDARD_GRAVITY,
            .ay = data.accelY / STANDARD_GRAVITY,
            .az = data.accelZ / STANDARD_GRAVITY,
            .gx = data.gyroX * RAD_TO_DEG,
            .gy = data.gyroY * RAD_TO_DEG,
            .gz = data.gyroZ * RAD_TO_DEG,
        };
#if CONFIG_FALL_LOG_SAMPLES
        printf("IMU,%lu,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f\n",
               (unsigned long)(esp_timer_get_time() / 1000),
               s.ax, s.ay, s.az, s.gx, s.gy, s.gz);
#endif

        fall_result_t r;
        if (fall_detector_push(&s, &r)) {
            last = r;
            if (r.fall) {
                falls++;
                ESP_LOGW(TAG, "Fall detected (score %.2f, %lu us)", r.score, (unsigned long)r.infer_us);
                ui_show_alert();
            }
        }

        if (++n % UI_UPDATE_EVERY == 0) {
            ui_update_status(sqrtf(s.ax * s.ax + s.ay * s.ay + s.az * s.az), &last, falls);
        }
    }
}

void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    bsp_display_start();
    bsp_display_backlight_on();

    bsp_display_lock(0);
    ui_create();
    bsp_display_unlock();

    static qmi8658_dev_t dev;
    imu_init(&dev);
    ESP_ERROR_CHECK(fall_detector_init());
    ESP_LOGI(TAG, "Fall detector running: %s backend, %d Hz, window %d, stride %d",
             fall_detector_backend_name(), CONFIG_FALL_SAMPLE_RATE_HZ,
             CONFIG_FALL_WINDOW_SAMPLES, CONFIG_FALL_STRIDE_SAMPLES);

    xTaskCreate(fall_detect_task, "fall_detect", 6144, &dev, 5, NULL);
}
