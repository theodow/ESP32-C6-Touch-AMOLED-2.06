#include <cmath>
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "fall_config.h"
#include "fall_model_tflm.h"
#include "fall_model_data.h"
#include "fall_model_config.h"

static const char *TAG = "FALL_TFLM";

static_assert(FALL_MODEL_WINDOW_SAMPLES == CONFIG_FALL_WINDOW_SAMPLES,
              "Model window length differs from CONFIG_FALL_WINDOW_SAMPLES");
static_assert(FALL_MODEL_SAMPLE_RATE_HZ == CONFIG_FALL_SAMPLE_RATE_HZ,
              "Model sample rate differs from CONFIG_FALL_SAMPLE_RATE_HZ");

namespace {
constexpr size_t kArenaSize = CONFIG_FALL_TFLM_ARENA_KB * 1024;
uint8_t *s_arena;
tflite::MicroInterpreter *s_interpreter;
TfLiteTensor *s_input;
TfLiteTensor *s_output;

// Ops produced by the Keras Conv1D model in training/train_fall_model.py.
using OpResolver = tflite::MicroMutableOpResolver<10>;

float read_output(const TfLiteTensor *t, int index)
{
    if (t->type == kTfLiteInt8) {
        return (t->data.int8[index] - t->params.zero_point) * t->params.scale;
    }
    return t->data.f[index];
}
}  // namespace

esp_err_t fall_model_init(void)
{
    const tflite::Model *model = tflite::GetModel(g_fall_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema %lu, expected %d", (unsigned long)model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_ERR_INVALID_VERSION;
    }

    static OpResolver resolver;
    resolver.AddConv2D();
    resolver.AddExpandDims();
    resolver.AddReshape();
    resolver.AddMaxPool2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddLogistic();
    resolver.AddSoftmax();
    resolver.AddQuantize();
    resolver.AddDequantize();

    s_arena = static_cast<uint8_t *>(heap_caps_aligned_alloc(16, kArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (s_arena == nullptr) {
        ESP_LOGE(TAG, "Could not allocate %u byte tensor arena", (unsigned)kArenaSize);
        return ESP_ERR_NO_MEM;
    }

    static tflite::MicroInterpreter interpreter(model, resolver, s_arena, kArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "AllocateTensors failed, raise CONFIG_FALL_TFLM_ARENA_KB");
        return ESP_FAIL;
    }
    s_interpreter = &interpreter;
    s_input = interpreter.input(0);
    s_output = interpreter.output(0);

    if (s_input->bytes / (s_input->type == kTfLiteInt8 ? 1 : sizeof(float)) !=
        (size_t)FALL_MODEL_WINDOW_SAMPLES * IMU_SAMPLE_CHANNELS) {
        ESP_LOGE(TAG, "Unexpected input size %u bytes", (unsigned)s_input->bytes);
        return ESP_ERR_INVALID_SIZE;
    }
    ESP_LOGI(TAG, "Model ready, arena used %u of %u bytes",
             (unsigned)interpreter.arena_used_bytes(), (unsigned)kArenaSize);
    return ESP_OK;
}

float fall_model_score(const imu_sample_t *window, size_t len)
{
    if (s_interpreter == nullptr || len != FALL_MODEL_WINDOW_SAMPLES) {
        return 0.0f;
    }

    for (size_t i = 0; i < len; i++) {
        const float raw[IMU_SAMPLE_CHANNELS] = {
            window[i].ax, window[i].ay, window[i].az,
            window[i].gx, window[i].gy, window[i].gz,
        };
        for (int c = 0; c < IMU_SAMPLE_CHANNELS; c++) {
            float x = (raw[c] - kFallModelMean[c]) / kFallModelStd[c];
            size_t idx = i * IMU_SAMPLE_CHANNELS + c;
            if (s_input->type == kTfLiteInt8) {
                long q = lroundf(x / s_input->params.scale) + s_input->params.zero_point;
                s_input->data.int8[idx] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
            } else {
                s_input->data.f[idx] = x;
            }
        }
    }

    if (s_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGW(TAG, "Invoke failed");
        return 0.0f;
    }

    // One sigmoid output, or two softmax outputs [not fall, fall].
    int outputs = s_output->dims->data[s_output->dims->size - 1];
    return read_output(s_output, outputs > 1 ? 1 : 0);
}
