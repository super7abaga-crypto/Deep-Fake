"""
Audio deepfake detection.

A Wav2Vec2 classifier scores short windows of speech, and the window scores are
aggregated into one verdict. Window-level scoring matters because synthetic
speech often degrades part-way through a clip (voice cloning drifts, TTS
artefacts cluster).

Audio is decoded with ffmpeg first, which means any container ffmpeg can read
works: mp3, wav, m4a, ogg, and the audio track of an mp4/mov/webm video.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000
WINDOW_SECONDS = 5.0
HOP_SECONDS = 4.0
MAX_WINDOWS = 8
MODEL_ID = "mo-thecreator/Deepfake-audio-detection"

_audio_model = None
_audio_processor = None
_load_error: str | None = None


def _ensure_model():
    """Lazily load the Wav2Vec2 audio classifier once per process."""
    global _audio_model, _audio_processor, _load_error
    if _audio_model is not None or _load_error is not None:
        return
    try:
        from transformers import AutoFeatureExtractor

        kwargs = {}
        if os.getenv("HF_HOME"):
            kwargs["cache_dir"] = os.getenv("HF_HOME")

        _audio_processor = AutoFeatureExtractor.from_pretrained(MODEL_ID, **kwargs)

        # transformers renamed this auto-class; try the audio head first, then
        # fall back to the concrete Wav2Vec2 class for older/newer versions.
        try:
            from transformers import AutoModelForAudioClassification as _Auto
        except ImportError:  # pragma: no cover
            _Auto = None
        try:
            if _Auto is not None:
                _audio_model = _Auto.from_pretrained(MODEL_ID, **kwargs)
            else:
                raise RuntimeError("no audio auto-class")
        except Exception:
            from transformers import Wav2Vec2ForSequenceClassification

            _audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID, **kwargs)
        _audio_model.eval()
    except Exception as exc:
        _load_error = f"{MODEL_ID}: {exc}"


def audio_available() -> bool:
    _ensure_model()
    return _audio_model is not None


def decode_to_wav(source_path: str) -> str:
    """Normalise any input container to 16 kHz mono WAV via ffmpeg."""
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    out.close()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", source_path,
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-vn", out.name]
    proc = subprocess.run(cmd, capture_output=True, timeout=180)
    if proc.returncode != 0 or not os.path.getsize(out.name):
        raise ValueError("Could not decode audio track: " + proc.stderr.decode()[:200])
    return out.name


def _read_wav(path: str) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        # linear resample; ffmpeg normally guarantees 16 kHz already
        idx = np.linspace(0, len(data) - 1, int(len(data) * SAMPLE_RATE / sr))
        data = np.interp(idx, np.arange(len(data)), data).astype(np.float32)
    return data


def _windows(signal: np.ndarray) -> list[np.ndarray]:
    win = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop = int(HOP_SECONDS * SAMPLE_RATE)
    if len(signal) <= win:
        return [signal]
    starts = list(range(0, max(len(signal) - win, 0) + 1, hop))
    if len(starts) > MAX_WINDOWS:
        step = len(starts) / MAX_WINDOWS
        starts = [starts[int(i * step)] for i in range(MAX_WINDOWS)]
    return [signal[s: s + win] for s in starts if len(signal[s: s + win]) > SAMPLE_RATE * 0.5]


def _acoustic_features(signal: np.ndarray) -> dict:
    """Cheap, explainable descriptors alongside the neural score."""
    dur = len(signal) / SAMPLE_RATE
    if len(signal) < 1600:
        return {"duration_seconds": round(dur, 2)}
    spec = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) + 1e-9
    freqs = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)
    total = spec.sum()
    centroid = float((freqs * spec).sum() / total)
    flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    # energy above 8 kHz: vocoders usually roll off sharply, so a very low value
    # is a mild synthetic-speech indicator
    hi = float(spec[freqs > 8000].sum() / total)
    return {
        "duration_seconds": round(dur, 2),
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_flatness": round(flatness, 5),
        "energy_above_8khz": round(hi, 5),
        "rms": round(float(np.sqrt(np.mean(signal ** 2))), 5),
    }


def analyse_audio(source_path: str) -> dict[str, Any]:
    """Score an audio file (or the audio track of any media file)."""
    started = time.time()
    _ensure_model()
    if _audio_model is None:
        raise RuntimeError(f"Audio model unavailable ({_load_error})")

    import torch

    wav_path = decode_to_wav(source_path)
    try:
        signal = _read_wav(wav_path)
        chunks = _windows(signal)
        scores = []
        for chunk in chunks:
            inputs = _audio_processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            with torch.inference_mode():
                logits = _audio_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).numpy()[0]
            id2label = {int(k): v for k, v in _audio_model.config.id2label.items()}
            fake_idx = next(
                (i for i in id2label if "fake" in str(id2label[i]).lower()),
                0,
            )
            scores.append(float(probs[fake_idx]))
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    scores_arr = np.array(scores)
    mean_score = float(scores_arr.mean())
    max_score = float(scores_arr.max())
    # A clip is fake-leaning if the average window says so, or if most windows
    # independently agree it is fake.
    fake_fraction = float((scores_arr >= 0.5).mean())
    final = float(np.clip(0.6 * mean_score + 0.4 * max_score, 0, 1))

    if final >= 0.60:
        label = "DEEPFAKE"
    elif final >= 0.40:
        label = "SUSPICIOUS"
    else:
        label = "AUTHENTIC"

    return {
        "media_type": "audio",
        "classification": label,
        "is_deepfake": label == "DEEPFAKE",
        "fake_probability": round(final, 4),
        "confidence": round(max(final, 1 - final), 4),
        "model_score": round(mean_score, 4),
        "window_max_score": round(max_score, 4),
        "windows_scored": len(scores),
        "windows_flagged_fake": int((scores_arr >= 0.5).sum()),
        "fake_window_fraction": round(fake_fraction, 3),
        "window_scores": [round(s, 4) for s in scores],
        "model": MODEL_ID,
        "acoustics": _acoustic_features(signal),
        "explanation": [
            f"Average over {len(scores)} speech window(s): {mean_score:.0%} probability of synthetic speech.",
            f"{int((scores_arr >= 0.5).sum())} of {len(scores)} windows independently flagged as synthetic.",
            f"Peak window score was {max_score:.0%}.",
        ],
        "elapsed_seconds": round(time.time() - started, 2),
    }
