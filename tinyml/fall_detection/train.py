"""Train a small 1D-CNN fall detector and export it as int8 TFLite + C array for the ESP32-C6.

Examples:
    # Smoke test, no downloads needed (synthetic data, model is not useful)
    python train.py --synthetic --epochs 5

    # Real training on SisFall and/or MobiFall
    python train.py --sisfall ~/data/SisFall_dataset --mobifall ~/data/MobiFall_Dataset_v2.0

Outputs (in --out, default ./output):
    fall_model_float.keras, fall_model_int8.tflite, fall_model.h, fall_model.cc, metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

import datasets
import export
from windows import augment, make_windows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("data")
    src.add_argument("--sisfall", type=Path, help="Path to the extracted SisFall_dataset folder")
    src.add_argument("--mobifall", type=Path, help="Path to the extracted MobiFall_Dataset_v2.0 folder")
    src.add_argument("--synthetic", action="store_true", help="Use generated data (pipeline smoke test only)")
    src.add_argument("--synthetic-subjects", type=int, default=20)
    win = p.add_argument_group("windowing")
    win.add_argument("--fs", type=float, default=50.0, help="Resample everything to this rate (Hz)")
    win.add_argument("--window-s", type=float, default=2.0, help="Window length in seconds")
    win.add_argument("--adl-stride-s", type=float, default=1.0, help="Stride between ADL windows (s)")
    win.add_argument("--fall-offsets", type=int, default=8, help="Windows taken around each fall impact")
    win.add_argument("--accel-only", action="store_true", help="Use only the 3 accelerometer channels")
    tr = p.add_argument_group("training")
    tr.add_argument("--epochs", type=int, default=40)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--lr", type=float, default=2e-3)
    tr.add_argument("--val-frac", type=float, default=0.15, help="Fraction of subjects for validation")
    tr.add_argument("--test-frac", type=float, default=0.15, help="Fraction of subjects for test")
    tr.add_argument("--no-augment", action="store_true")
    tr.add_argument("--seed", type=int, default=42)
    out = p.add_argument_group("export")
    out.add_argument("--out", type=Path, default=Path(__file__).parent / "output")
    out.add_argument("--name", default="fall_model", help="C symbol / file name prefix")
    out.add_argument("--max-model-kb", type=float, default=100.0, help="Fail if the int8 model is larger")
    out.add_argument("--tensor-arena-kb", type=int, default=32, help="Arena size written to the header")
    return p.parse_args()


def load_recordings(args) -> list[datasets.Recording]:
    recs: list[datasets.Recording] = []
    if args.sisfall:
        recs += list(datasets.load_sisfall(args.sisfall, args.fs))
    if args.mobifall:
        recs += list(datasets.load_mobifall(args.mobifall, args.fs))
    if args.synthetic:
        recs += list(datasets.load_synthetic(args.synthetic_subjects, args.fs, args.seed))
    if not recs:
        raise SystemExit("No data: pass --sisfall, --mobifall and/or --synthetic")
    return recs


def split_subjects(subjects: list[str], val_frac: float, test_frac: float, rng) -> dict[str, str]:
    """Subject-wise split so no person appears in more than one of train/val/test.

    Done per dataset so each split sees every source.
    """
    by_source = defaultdict(list)
    for s in sorted(set(subjects)):
        by_source[s.split("_")[0]].append(s)
    assign = {}
    for subs in by_source.values():
        subs = list(rng.permutation(subs))
        n_test = max(1, round(len(subs) * test_frac))
        n_val = max(1, round(len(subs) * val_frac))
        for i, s in enumerate(subs):
            assign[s] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    return assign


def build_model(window_len: int, channels: int) -> tf.keras.Model:
    """~10k parameters; every op maps to a TFLM builtin (and ESP-NN for conv/pool)."""
    L = tf.keras.layers
    inp = L.Input((window_len, channels))
    x = L.Conv1D(16, 5, padding="same", activation="relu")(inp)
    x = L.MaxPooling1D(2)(x)
    x = L.Conv1D(32, 5, padding="same", activation="relu")(x)
    x = L.MaxPooling1D(2)(x)
    x = L.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(0.3)(x)
    x = L.Dense(16, activation="relu")(x)
    out = L.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(inp, out, name="fall_cnn")


def metrics_at(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * sens / max(prec + sens, 1e-9)
    return dict(threshold=round(thr, 4), sensitivity=sens, specificity=spec, precision=prec, f1=f1,
                tp=tp, tn=tn, fp=fp, fn=fn)


def pick_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Best F1 on validation; ties go to the lower threshold (missing a fall is worse than a false alarm)."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [metrics_at(y, p, t)["f1"] for t in grid]
    return float(grid[int(np.argmax(scores))])


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    window_len = int(round(args.window_s * args.fs))
    recs = load_recordings(args)
    channels = slice(0, 3) if args.accel_only else slice(0, 6)
    assign = split_subjects([r.subject for r in recs], args.val_frac, args.test_frac, rng)

    parts = {k: ([], []) for k in ("train", "val", "test")}
    for r in recs:
        w, y = make_windows(r, window_len, int(args.adl_stride_s * args.fs), args.fall_offsets, rng)
        if len(w):
            parts[assign[r.subject]][0].append(w[:, :, channels])
            parts[assign[r.subject]][1].append(y)
    data = {k: (np.concatenate(xs), np.concatenate(ys)) for k, (xs, ys) in parts.items() if xs}
    for k in ("train", "val", "test"):
        if k not in data:
            raise SystemExit(f"Split '{k}' is empty; need more subjects")
    for k, (x, y) in data.items():
        print(f"{k:5s}: {len(x):6d} windows, {int(y.sum()):5d} falls, "
              f"{len({s for s, v in assign.items() if v == k})} subjects")

    x_train, y_train = data["train"]
    if not args.no_augment:
        x_train = np.concatenate([x_train, augment(x_train, rng)])
        y_train = np.concatenate([y_train, y_train])

    mean = x_train.reshape(-1, x_train.shape[2]).mean(0)
    std = x_train.reshape(-1, x_train.shape[2]).std(0) + 1e-6
    norm = lambda x: ((x - mean) / std).astype(np.float32)  # noqa: E731
    x_train_n, x_val_n, x_test_n = norm(x_train), norm(data["val"][0]), norm(data["test"][0])
    y_val, y_test = data["val"][1], data["test"][1]

    n_pos = max(int(y_train.sum()), 1)
    n_neg = len(y_train) - n_pos
    class_weight = {0: 1.0, 1: n_neg / n_pos}

    model = build_model(window_len, x_train.shape[2])
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")],
    )
    model.fit(
        x_train_n, y_train,
        validation_data=(x_val_n, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        ],
        verbose=2,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / f"{args.name}_float.keras")

    p_val = model.predict(x_val_n, verbose=0).ravel()
    thr = pick_threshold(y_val, p_val)
    p_test = model.predict(x_test_n, verbose=0).ravel()

    rep_idx = rng.choice(len(x_train_n), size=min(500, len(x_train_n)), replace=False)
    tflite_model = export.to_int8_tflite(model, x_train_n[rep_idx])
    (args.out / f"{args.name}_int8.tflite").write_bytes(tflite_model)
    p_test_q = export.run_tflite(tflite_model, x_test_n)

    h, cc = export.write_c_sources(
        args.out, tflite_model,
        name=args.name, sample_rate_hz=args.fs, window_len=window_len,
        channel_mean=mean, channel_std=std, threshold=thr,
        tensor_arena_hint=args.tensor_arena_kb * 1024,
    )

    size_kb = len(tflite_model) / 1024
    report = {
        "sources": sorted({r.source for r in recs}),
        "window_len": window_len, "fs": args.fs, "channels": int(x_train.shape[2]),
        "params": int(model.count_params()),
        "int8_model_kb": round(size_kb, 2),
        "test_float": metrics_at(y_test, p_test, thr),
        "test_int8": metrics_at(y_test, p_test_q, thr),
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {h}, {cc} and {args.name}_int8.tflite ({size_kb:.1f} KB)")
    if size_kb > args.max_model_kb:
        raise SystemExit(f"int8 model is {size_kb:.1f} KB, over the {args.max_model_kb} KB budget")


if __name__ == "__main__":
    main()
