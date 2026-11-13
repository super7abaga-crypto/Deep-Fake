"""
Video deepfake detection.

Deepfake video is usually a face swapped across many frames, so the system
samples frames evenly through the clip, scores each one with the image
ensemble, and aggregates. Two aggregate views matter:

    * mean frame score  - overall likelihood the clip is manipulated
    * flagged fraction  - how consistently the manipulation appears

A clip with a handful of odd frames is treated more cautiously than one where
every frame is flagged. Where the video has an audio track, it is scored too and
folded into the final verdict, because lip-sync forgeries are often more obvious
in the audio than in the pixels.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from . import audio_detector
from .models import registry

DEFAULT_FRAMES = 12
MAX_FRAMES = 24
FRAME_MAX_SIDE = 512


def probe(video_path: str) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", video_path]
    out = subprocess.run(cmd, capture_output=True, timeout=120)
    if out.returncode != 0:
        raise ValueError("ffprobe failed: " + out.stderr.decode()[:200])
    data = json.loads(out.stdout or "{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    fps_raw = video.get("avg_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    return {
        "duration_seconds": round(duration, 2),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": round(fps, 2),
        "codec": video.get("codec_name"),
        "has_audio": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
    }


def extract_frames(video_path: str, tmpdir: str, n_frames: int) -> list[tuple[float, str]]:
    """Sample n_frames evenly across the clip. Returns (timestamp, path) pairs."""
    meta = probe(video_path)
    duration = meta["duration_seconds"] or 0
    n_frames = max(1, min(n_frames, MAX_FRAMES))

    if duration <= 0:
        pattern = os.path.join(tmpdir, "frame_%03d.jpg")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                        "-vf", f"scale={FRAME_MAX_SIDE}:-2", "-vframes", str(n_frames),
                        pattern], capture_output=True, timeout=180)
        files = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
        return [(float(i), f) for i, f in enumerate(files)]

    # aim for one frame every (duration / n_frames) seconds
    rate = n_frames / duration
    pattern = os.path.join(tmpdir, "frame_%03d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
         "-vf", f"fps={rate:.6f},scale={FRAME_MAX_SIDE}:-2",
         "-frames:v", str(n_frames), "-q:v", "3", pattern],
        capture_output=True, timeout=300,
    )
    files = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
    if not files:  # extremely short or odd file: fall back to keyframes
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                        "-vf", f"scale={FRAME_MAX_SIDE}:-2", "-vframes", str(n_frames),
                        pattern], capture_output=True, timeout=180)
        files = sorted(glob.glob(os.path.join(tmpdir, "frame_*.jpg")))
    step = duration / max(len(files), 1)
    return [(round(i * step, 2), f) for i, f in enumerate(files)]


def _score_frame(img: Image.Image) -> float:
    faces = registry.face.detect(img)
    crops = [img] + faces
    total, weight_total = 0.0, 0.0
    for model in registry.models:
        verdict = model.predict(crops)
        total += model.weight * verdict.fake_probability
        weight_total += model.weight
    return total / max(weight_total, 1e-9)


def _classify(prob: float, flagged_fraction: float) -> str:
    # Sustained flagging across most frames is stronger evidence than a high
    # average alone, so the fake threshold relaxes slightly when it is consistent.
    if prob >= 0.60 or (prob >= 0.50 and flagged_fraction >= 0.6):
        return "DEEPFAKE"
    if prob >= 0.40:
        return "SUSPICIOUS"
    return "AUTHENTIC"


def analyse_video(video_path: str, n_frames: int = DEFAULT_FRAMES,
                  include_audio: bool = True) -> dict[str, Any]:
    started = time.time()
    registry.load()

    meta = probe(video_path)
    tmpdir = tempfile.mkdtemp(prefix="frames_")
    try:
        frames = extract_frames(video_path, tmpdir, n_frames)
        if not frames:
            raise ValueError("No frames could be extracted from this file")

        timeline = []
        for ts, path in frames:
            img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            score = _score_frame(img)
            timeline.append({"timestamp_seconds": ts, "fake_probability": round(score, 4)})
    finally:
        for f in glob.glob(os.path.join(tmpdir, "*")):
            try:
                os.unlink(f)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    scores = np.array([f["fake_probability"] for f in timeline])
    mean_score = float(scores.mean())
    flagged_fraction = float((scores >= 0.5).mean())

    audio_result = None
    audio_error = None
    if include_audio and meta["has_audio"]:
        try:
            audio_result = audio_detector.analyse_audio(video_path)
        except Exception as exc:
            audio_error = str(exc)[:200]

    # visual verdict first, then audio corroboration
    visual_prob = float(np.clip(0.75 * mean_score + 0.25 * float(scores.max()), 0, 1))
    if audio_result is not None:
        # audio alone can only pull the score towards fake if the frames agree
        final_prob = float(np.clip(0.7 * visual_prob + 0.3 * audio_result["fake_probability"], 0, 1))
    else:
        final_prob = visual_prob

    label = _classify(final_prob, flagged_fraction)

    explanation = [
        f"{len(timeline)} frames sampled across {meta['duration_seconds']}s.",
        f"Mean frame score {mean_score:.0%}; {int((scores >= 0.5).sum())} of {len(scores)} frames flagged.",
        f"Peak frame score {float(scores.max()):.0%}.",
    ]
    if audio_result is not None:
        explanation.append(
            f"Audio track scored {audio_result['fake_probability']:.0%} "
            f"({audio_result['classification']})."
        )
    elif audio_error:
        explanation.append(f"Audio track present but could not be scored: {audio_error}")
    else:
        explanation.append("No audio track in this file.")

    return {
        "media_type": "video",
        "classification": label,
        "is_deepfake": label == "DEEPFAKE",
        "fake_probability": round(final_prob, 4),
        "confidence": round(max(final_prob, 1 - final_prob), 4),
        "visual_score": round(visual_prob, 4),
        "mean_frame_score": round(mean_score, 4),
        "max_frame_score": round(float(scores.max()), 4),
        "flagged_frame_fraction": round(flagged_fraction, 3),
        "frames_analysed": len(timeline),
        "video": meta,
        "timeline": timeline,
        "audio": audio_result,
        "explanation": explanation,
        "elapsed_seconds": round(time.time() - started, 2),
    }
