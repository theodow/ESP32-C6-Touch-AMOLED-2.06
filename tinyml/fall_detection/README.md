# Fall detection model (TinyML)

Training pipeline for an on-device fall detector for the ESP32-C6 watch.
It loads public fall datasets, cuts 2 s IMU windows, trains a small 1D-CNN,
quantizes it to full int8 TFLite and writes a C array for TensorFlow Lite Micro.

## Setup

```bash
cd tinyml/fall_detection
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download and extract one or both datasets (they require accepting the authors' terms):

| Dataset | Sensors used | Rate | Labels |
|---|---|---|---|
| [SisFall](https://www.mdpi.com/1424-8220/17/1/198) | ADXL345 accel + ITG3200 gyro, waist | 200 Hz | `F01..F15` falls, `D01..D19` ADLs |
| MobiFall v2.0 (BMI lab, Hellenic Mediterranean University) | phone accel + gyro, trouser pocket | ~87 / 200 Hz, irregular | `FALLS/` vs `ADL/` folders |

Both are converted to the watch's units: acceleration in g, angular rate in deg/s,
resampled to 50 Hz.

## Train

```bash
# Smoke test without any download (synthetic data, the model is not useful)
python train.py --synthetic --epochs 5

# Real run
python train.py --sisfall /path/to/SisFall_dataset --mobifall /path/to/MobiFall_Dataset_v2.0
```

Useful flags: `--fs 50 --window-s 2` (window = 100 samples), `--accel-only` (3 channels, if the
gyro stays off on the watch), `--fall-offsets 8` (windows taken around each fall impact),
`--max-model-kb 100` (the script fails if the int8 model is bigger).

How windows are labelled: ADL trials are cut with a 1 s stride and labelled 0. In fall trials
the impact is the peak of |acc| and the windows are placed so the impact sits between 30% and
80% of the window; the rest of a fall trial is dropped. Train/val/test are split by subject
(per dataset), so no person is in more than one split. Training adds rotated/scaled/noisy
copies so the model tolerates how the watch sits on the wrist. The decision threshold is the
best-F1 one on the validation set.

## Outputs (`output/`)

- `fall_model_int8.tflite`: the quantized model (~17 KB, ~7k parameters)
- `fall_model.h` / `fall_model.cc`: the model as a 16-byte aligned C array, plus sample rate,
  window length, per-channel mean/std, input/output quantization and the threshold
- `fall_model_float.keras`: the float model
- `metrics.json`: sensitivity, specificity, precision and F1 on the test subjects for the
  float model and for the int8 model as it will run on the device

## Using it on the ESP32-C6

Ops to register in the TFLM `MicroMutableOpResolver`: `Conv2D`, `MaxPool2D`, `Mean`,
`FullyConnected`, `Logistic`, `Reshape`, `ExpandDims` (7 ops). With the
`espressif/esp-tflite-micro` component the conv/pool kernels use ESP-NN.

Feeding a window (per sample, QMI8658 at >= 50 Hz, accel in m/s^2):

```c
float v[6] = {ax / 9.80665f, ay / 9.80665f, az / 9.80665f, gx, gy, gz};
for (int c = 0; c < FALL_MODEL_NUM_CHANNELS; c++) {
    float xn = (v[c] - fall_model_channel_mean[c]) / fall_model_channel_std[c];
    int q = (int)lrintf(xn / FALL_MODEL_INPUT_SCALE) + FALL_MODEL_INPUT_ZERO_POINT;
    input->data.int8[t * FALL_MODEL_NUM_CHANNELS + c] = q < -128 ? -128 : q > 127 ? 127 : q;
}
// after invoke():
float p = (output->data.int8[0] - FALL_MODEL_OUTPUT_ZERO_POINT) * FALL_MODEL_OUTPUT_SCALE;
bool fall = p >= FALL_MODEL_THRESHOLD;
```

Note the datasets were recorded at the waist (SisFall) and in a pocket (MobiFall), not on the
wrist, so expect to fine-tune on watch recordings before trusting the numbers.
