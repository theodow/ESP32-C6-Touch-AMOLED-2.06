#!/usr/bin/env python3
"""Train the fall detection CNN and export it for TensorFlow Lite Micro.

Writes main/model/fall_model_data.cc and main/model/fall_model_config.h, which
the firmware compiles when CONFIG_FALL_BACKEND_TFLM is selected.

Data sources (combine as many as you like):
  --sisfall DIR   SisFall dataset root (folders SA01..SA23, SE01..SE15).
                  Falls are files F01..F15, everything else (D01..D19) is ADL.
  --csv DIR       Recordings from the watch itself: logs captured with
                  CONFIG_FALL_LOG_SAMPLES (lines "IMU,ms,ax,ay,az,gx,gy,gz").
                  Name each file fall_*.txt|csv|log (one fall per file) or
                  adl_*.txt|csv|log.
  --synthetic     Generated traces, only for checking the pipeline end to end.
                  A model trained on them is not a real fall detector.

All data is brought to the firmware's input format: 50 Hz, 128-sample windows,
channels ax, ay, az in g and gx, gy, gz in deg/s.

Example:
  python train_fall_model.py --sisfall ~/data/SisFall_dataset --csv ~/data/watch
"""

import argparse
import re
from pathlib import Path

import numpy as np

SAMPLE_RATE_HZ = 50
WINDOW = 128
STRIDE = 25
# Positions of the impact peak inside a fall window. They match the range the
# firmware evaluates: at least 0.5 s of history and 1 s of data after the peak.
FALL_PEAK_OFFSETS = (30, 45, 60, 75)
# Ops the firmware registers in main/fall_model_tflm.cc.
FIRMWARE_OPS = {
    "CONV_2D", "EXPAND_DIMS", "RESHAPE", "MAX_POOL_2D", "MEAN",
    "FULLY_CONNECTED", "LOGISTIC", "SOFTMAX", "QUANTIZE", "DEQUANTIZE",
}

MODEL_DIR = Path(__file__).resolve().parent.parent / "main" / "model"


# ---------------------------------------------------------------- loading


def resample(x, src_hz):
    """Block-average to SAMPLE_RATE_HZ (src_hz must be an integer multiple)."""
    factor = src_hz // SAMPLE_RATE_HZ
    if factor * SAMPLE_RATE_HZ != src_hz:
        raise ValueError(f"cannot resample {src_hz} Hz to {SAMPLE_RATE_HZ} Hz")
    n = len(x) // factor * factor
    return x[:n].reshape(-1, factor, x.shape[1]).mean(axis=1)


def load_sisfall(root):
    """Yields (trace, is_fall, group). SisFall is 200 Hz, ADXL345 + ITG3200 at the waist."""
    acc_scale = 2 * 16 / 2**13      # ADXL345, +-16 g, 13 bit
    gyro_scale = 2 * 2000 / 2**16   # ITG3200, +-2000 deg/s, 16 bit
    for path in sorted(Path(root).glob("S[AE][0-9][0-9]/*.txt")):
        rows = []
        for line in path.read_text().splitlines():
            vals = [v for v in line.strip().rstrip(";").split(",") if v.strip()]
            if len(vals) >= 6:
                rows.append([float(v) for v in vals[:6]])
        if not rows:
            continue
        raw = np.asarray(rows, dtype=np.float32)
        trace = np.concatenate([raw[:, :3] * acc_scale, raw[:, 3:6] * gyro_scale], axis=1)
        yield resample(trace, 200), path.name.startswith("F"), path.parent.name


def load_csv(root):
    """Yields (trace, is_fall, group) from watch logs."""
    for path in sorted(Path(root).iterdir()):
        name = path.name.lower()
        if not re.match(r"(fall|adl)_.*\.(txt|csv|log)$", name):
            continue
        rows = []
        for line in path.read_text(errors="ignore").splitlines():
            if line.startswith("IMU,"):
                parts = line.strip().split(",")
                if len(parts) == 8:
                    rows.append([float(v) for v in parts[2:]])
        if len(rows) >= WINDOW:
            yield np.asarray(rows, dtype=np.float32), name.startswith("fall_"), path.name


def load_synthetic(n_per_class, rng):
    """Yields toy traces: falls (free fall, impact, lying) and ADLs (walking, still)."""
    t = np.arange(10 * SAMPLE_RATE_HZ)
    for i in range(n_per_class):
        g = rng.normal(0, 0.03, (len(t), 6)).astype(np.float32)
        g[:, 3:] *= 300
        a = g[:, :3]
        a[:, 2] += 1.0
        start = rng.integers(150, 300)
        a[start:start + 15] *= 0.2
        a[start + 15:start + 18, 2] += rng.uniform(2.0, 5.0)
        a[start + 18:, 2] -= 1.0
        a[start + 18:, 0] += 1.0
        g[start:start + 25, 3:] += rng.normal(0, 150, (25, 3))
        yield g, True, f"synthetic{i % 10}"

        g = rng.normal(0, 0.03, (len(t), 6)).astype(np.float32)
        g[:, 3:] *= 300
        g[:, 2] += 1.0
        if i % 2:
            g[:, 2] += rng.uniform(0.2, 0.7) * np.sin(2 * np.pi * t / rng.uniform(20, 30))
        yield g, False, f"synthetic{i % 10}"


# -------------------------------------------------------------- windowing


def windows_for(trace, is_fall):
    """Returns (X, y) windows cut from one trace."""
    xs, ys = [], []
    mag = np.linalg.norm(trace[:, :3], axis=1)
    if is_fall:
        peak = int(np.argmax(mag))
        for off in FALL_PEAK_OFFSETS:
            s = peak - off
            if 0 <= s and s + WINDOW <= len(trace):
                xs.append(trace[s:s + WINDOW])
                ys.append(1)
        # Activity well before the fall is a useful negative.
        for s in range(0, peak - WINDOW - SAMPLE_RATE_HZ, STRIDE):
            xs.append(trace[s:s + WINDOW])
            ys.append(0)
    else:
        for s in range(0, len(trace) - WINDOW + 1, STRIDE):
            xs.append(trace[s:s + WINDOW])
            ys.append(0)
    return xs, ys


def build_dataset(sources):
    X, y, groups = [], [], []
    for trace, is_fall, group in sources:
        xs, ys = windows_for(trace, is_fall)
        X += xs
        y += ys
        groups += [group] * len(xs)
    if not X:
        raise SystemExit("No data loaded. Pass --sisfall, --csv or --synthetic.")
    return np.stack(X).astype(np.float32), np.asarray(y, dtype=np.float32), np.asarray(groups)


def split_by_group(groups, test_fraction, rng):
    """Holds out whole subjects/recordings so the test set is unseen people."""
    unique = np.unique(groups)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test = np.isin(groups, unique[:n_test])
    return ~test, test


# ------------------------------------------------------------ model/export


def build_model():
    import tensorflow as tf

    return tf.keras.Sequential([
        tf.keras.layers.Input((WINDOW, 6)),
        tf.keras.layers.Conv1D(16, 5, activation="relu"),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(32, 5, activation="relu"),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def convert_int8(model, X_rep):
    import tensorflow as tf

    def representative():
        for i in range(0, min(len(X_rep), 500)):
            yield [X_rep[i:i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def tflite_predict(tflite_model, X):
    import tensorflow as tf

    # No XNNPACK delegate, so the op list matches what TFLM will execute.
    interp = tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interp.allocate_tensors()
    inp, out = interp.get_input_details()[0], interp.get_output_details()[0]
    ops = {op["op_name"] for op in interp._get_ops_details()}
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]
    preds = []
    for x in X:
        q = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp["index"], q[None])
        interp.invoke()
        preds.append((interp.get_tensor(out["index"])[0, 0].astype(np.float32) - out_zp) * out_scale)
    return np.asarray(preds), ops


def report(name, y_true, prob, threshold):
    pred = prob >= threshold
    tp = int(np.sum(pred & (y_true == 1)))
    fn = int(np.sum(~pred & (y_true == 1)))
    tn = int(np.sum(~pred & (y_true == 0)))
    fp = int(np.sum(pred & (y_true == 0)))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    print(f"{name}: sensitivity {sens:.3f}  specificity {spec:.3f}  (tp {tp} fn {fn} tn {tn} fp {fp})")


def write_c_files(tflite_model, mean, std, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    body = ",\n".join(
        "    " + ", ".join(f"0x{b:02x}" for b in tflite_model[i:i + 12])
        for i in range(0, len(tflite_model), 12)
    )
    (out_dir / "fall_model_data.cc").write_text(
        "// Generated by training/train_fall_model.py. Do not edit.\n"
        '#include "fall_model_data.h"\n\n'
        f"alignas(16) const unsigned char g_fall_model_data[] = {{\n{body}\n}};\n"
        f"const unsigned int g_fall_model_data_len = {len(tflite_model)};\n"
    )
    fmt = lambda v: ", ".join(f"{x:.6f}f" for x in v)
    (out_dir / "fall_model_config.h").write_text(
        "// Generated by training/train_fall_model.py. Do not edit.\n"
        "#pragma once\n\n"
        f"#define FALL_MODEL_SAMPLE_RATE_HZ {SAMPLE_RATE_HZ}\n"
        f"#define FALL_MODEL_WINDOW_SAMPLES {WINDOW}\n\n"
        "// Per-channel normalisation (ax, ay, az in g; gx, gy, gz in deg/s).\n"
        f"static const float kFallModelMean[6] = {{{fmt(mean)}}};\n"
        f"static const float kFallModelStd[6] = {{{fmt(std)}}};\n"
    )


# ------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sisfall", type=Path)
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.7,
                    help="report metrics at this probability (match CONFIG_FALL_TFLM_THRESHOLD_PCT)")
    ap.add_argument("--out", type=Path, default=MODEL_DIR)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sources = []
    if args.sisfall:
        sources.append(load_sisfall(args.sisfall))
    if args.csv:
        sources.append(load_csv(args.csv))
    if args.synthetic:
        print("WARNING: synthetic data only checks the pipeline; do not deploy this model.")
        sources.append(load_synthetic(100, rng))
    X, y, groups = build_dataset(s for src in sources for s in src)
    print(f"{len(X)} windows, {int(y.sum())} falls, {len(np.unique(groups))} groups")

    train, test = split_by_group(groups, args.test_fraction, rng)
    mean = X[train].reshape(-1, 6).mean(axis=0)
    std = X[train].reshape(-1, 6).std(axis=0) + 1e-6
    Xn = (X - mean) / std

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    model = build_model()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    pos = max(y[train].sum(), 1)
    neg = max(len(y[train]) - pos, 1)
    model.fit(Xn[train], y[train], epochs=args.epochs, batch_size=64, verbose=2,
              validation_data=(Xn[test], y[test]),
              class_weight={0: 1.0, 1: float(neg / pos)})

    report("float test", y[test], model.predict(Xn[test], verbose=0)[:, 0], args.threshold)
    tflite_model = convert_int8(model, Xn[train])
    prob, ops = tflite_predict(tflite_model, Xn[test])
    report("int8  test", y[test], prob, args.threshold)

    missing = ops - FIRMWARE_OPS
    if missing:
        raise SystemExit(f"Model uses ops the firmware does not register: {sorted(missing)}. "
                         "Add them to the resolver in main/fall_model_tflm.cc.")
    write_c_files(tflite_model, mean, std, args.out)
    print(f"Wrote {len(tflite_model)} byte model to {args.out} (ops: {', '.join(sorted(ops))})")


if __name__ == "__main__":
    main()
