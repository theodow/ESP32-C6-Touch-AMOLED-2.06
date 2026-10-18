# 06_fall_detector

Fall detection on the ESP32-C6-Touch-AMOLED-2.06, built on the `04_qmi8658`
example. The QMI8658 is sampled at 50 Hz, samples go into a sliding window
(128 samples, 2.56 s), and every 25 samples (0.5 s) the window is classified.
A detected fall turns the whole screen red with "FALL DETECTED"; tapping the
screen dismisses it.

```
QMI8658 ──50 Hz──▶ ring buffer (128 × 6) ──every 25 samples──▶ backend ──score ≥ threshold──▶ alert
  ax ay az [g]                                                  heuristic      (5 s refractory)
  gx gy gz [deg/s]                                              or TFLM CNN
```

## Backends

Pick one under `idf.py menuconfig` → **Fall detector** → **Detection backend**.

**Threshold heuristic** (default, works without a model). Inside the window it
looks for an impact peak (|a| > 2.5 g) that has at least 0.5 s of history and
1 s of data after it, requires free fall (|a| < 0.6 g) in the 0.5 s before the
peak, and requires stillness (std of |a| < 0.15 g) from 0.5 s after the peak to
the end of the window. All thresholds are in the same menu.

**TensorFlow Lite Micro.** A small int8 1D CNN (two Conv1D layers, about 4 k
parameters) trained with `training/train_fall_model.py`. The firmware
normalises the window with the mean/std saved at training time, quantises it,
runs the model and compares the fall probability with
`FALL_TFLM_THRESHOLD_PCT`. The status screen shows the inference time in µs.
The exported model is about 11 KB of flash and needs about 7 KB of tensor
arena (24 KB is reserved by default).

## Build

```
idf.py set-target esp32c6
idf.py build flash monitor
```

## Training a model

1. Get data. Either or both of:
   - [SisFall](https://doi.org/10.3390/s17010198): 38 subjects, falls and
     ADLs at 200 Hz. The script converts it to 50 Hz, g and deg/s. SisFall was
     recorded at the waist, so a model trained only on it will be weaker on
     the wrist.
   - Recordings from the watch: enable **Print every IMU sample as CSV** in the
     menu, run `idf.py monitor | tee fall_01.log` while performing one fall
     (or `adl_walk_01.log` for daily activity), and put the files in one folder.
2. Train and export (TensorFlow 2.x):
   ```
   pip install tensorflow numpy
   python training/train_fall_model.py --sisfall ~/data/SisFall_dataset --csv ~/data/watch
   ```
   It holds out whole subjects/recordings for the test set, prints sensitivity
   and specificity for the float and the int8 model, checks that the model only
   uses ops the firmware registers, and writes `main/model/fall_model_data.cc`
   and `main/model/fall_model_config.h`.
3. Select **TensorFlow Lite Micro** in the menu and rebuild.

`--synthetic` trains on generated traces. It only proves the pipeline works;
that model is not a fall detector.

## Tests

The window and the heuristic build on a host:

```
cd test_host && make test
```

## Notes on sdkconfig

The other examples in this repository carry ESP32-S3 options (`CONFIG_SPIRAM*`,
`CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240`, RGB LCD buffers, Brookesia). The ESP32-C6
has no PSRAM and tops out at 160 MHz, so Kconfig ignores them. This example's
`sdkconfig.defaults` sets 160 MHz explicitly and drops the rest, along with the
LVGL demos `04_qmi8658` compiled in but never ran.

The accelerometer stays at ±8 g and the gyroscope at ±512 deg/s, the ranges the
Waveshare driver documents. Hard wrist impacts can saturate them; the driver
may offer wider ranges (±16 g, ±2048 deg/s) if that matters.
