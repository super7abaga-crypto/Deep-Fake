"""
Evaluation harness
==================

Runs the shipped detection pipeline over a labelled folder tree and reports
accuracy, precision, recall, F1, AUC and the confusion matrix, plus per-file
predictions in CSV. Use these numbers in your report: they come from the same
code the app serves, not from a separate notebook.

Expected layout:

    <data-root>/fake/*.jpg     files known to be manipulated
    <data-root>/real/*.jpg     files known to be authentic

Usage:
    python evaluate.py --data ../test_media --out results.csv
    python evaluate.py --data ../test_media --media image
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from app.detectors import audio_detector, image_detector, video_detector

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}


def collect(root: str, media: str) -> list[tuple[int, str]]:
    ext = {"image": IMAGE_EXT, "video": VIDEO_EXT, "audio": AUDIO_EXT}.get(media)
    items: list[tuple[int, str]] = []
    for label, folder in ((1, "fake"), (0, "real")):
        for path in sorted(glob.glob(os.path.join(root, folder, "*"))):
            if ext is None or os.path.splitext(path)[1].lower() in ext:
                items.append((label, path))
    return items


def analyse(path: str, media: str) -> dict:
    if media == "image":
        with open(path, "rb") as fh:
            return image_detector.analyse_image(fh.read())
    if media == "video":
        return video_detector.analyse_video(path, n_frames=8)
    return audio_detector.analyse_audio(path)


def metrics(scores: list[float], labels: list[int], threshold: float) -> dict:
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
    tn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 0)
    fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)
    n = len(labels)
    acc = (tp + tn) / n if n else 0
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if pos and neg:
        wins = sum(1 for a in pos for b in neg if a > b) + 0.5 * sum(
            1 for a in pos for b in neg if a == b)
        auc = wins / (len(pos) * len(neg))
    else:
        auc = float("nan")

    return {"n": n, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "auc": auc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../test_media")
    ap.add_argument("--media", default="image", choices=["image", "video", "audio"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    items = collect(args.data, args.media)
    if not items:
        raise SystemExit(f"No files found under {args.data}/fake and {args.data}/real")

    rows, scores, labels = [], [], []
    t0 = time.time()
    for label, path in items:
        try:
            result = analyse(path, args.media)
            score = float(result["fake_probability"])
        except Exception as exc:
            print(f"  ! {os.path.basename(path)}: {exc}")
            continue
        rows.append({
            "file": os.path.basename(path),
            "true_label": "fake" if label == 1 else "real",
            "predicted": result["classification"],
            "fake_probability": round(score, 4),
            "elapsed_seconds": result["elapsed_seconds"],
        })
        scores.append(score)
        labels.append(label)
        print(f"  {rows[-1]['true_label']:4} -> {result['classification']:10} "
              f"p={score:.3f}  {rows[-1]['file']}")

    m = metrics(scores, labels, args.threshold)

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "media_type": args.media,
        "threshold": args.threshold,
        **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "mean_fake_probability_on_fakes": round(
            float(np.mean([s for s, l in zip(scores, labels) if l == 1])), 4) if 1 in labels else None,
        "mean_fake_probability_on_reals": round(
            float(np.mean([s for s, l in zip(scores, labels) if l == 0])), 4) if 0 in labels else None,
        "total_seconds": round(time.time() - t0, 1),
        "predictions_csv": os.path.abspath(args.out),
    }
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
