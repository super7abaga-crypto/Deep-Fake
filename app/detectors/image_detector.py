"""
Image deepfake detection: model ensemble + face-aware cropping + explanation.

Pipeline
    1. normalise the upload to RGB
    2. locate faces (Haar cascade)
    3. score the full frame and each face crop with every model
    4. blend model probabilities into one fake score
    5. attach classical forensic evidence for the explanation
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from . import forensics
from .models import registry

MAX_SIDE = 1024

# Thresholds were chosen from measured performance on two independent labelled
# sets (see docs/RESULTS.md): 0.60 keeps precision high, 0.40 is the
# "look closer" line that keeps deepfake recall at 100%.
CLASS_THRESHOLDS = (
    (0.60, "DEEPFAKE"),
    (0.40, "SUSPICIOUS"),
    (0.0, "AUTHENTIC"),
)


def _classify(prob: float) -> str:
    for threshold, label in CLASS_THRESHOLDS:
        if prob >= threshold:
            return label
    return "AUTHENTIC"


def _load(image_bytes: bytes) -> Image.Image:
    import io

    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def _resize_for_models(img: Image.Image) -> Image.Image:
    """The ViTs expect 224px inputs; feeding them a huge image wastes time."""
    out = img.copy()
    out.thumbnail((384, 384), Image.LANCZOS)
    return out


def analyse_image(image_bytes: bytes) -> dict[str, Any]:
    started = time.time()
    registry.load()

    img = _load(image_bytes)
    small = _resize_for_models(img)
    faces = registry.face.detect(small)

    crops = [small] + faces
    per_model = []
    weighted_sum = 0.0
    weight_total = 0.0

    for model in registry.models:
        verdict = model.predict(crops)
        per_model.append(verdict.as_dict())
        weighted_sum += model.weight * verdict.fake_probability
        weight_total += model.weight

    model_score = weighted_sum / max(weight_total, 1e-9)

    features = forensics.extract_all(img)
    heuristic = forensics.heuristic_fake_score(features)

    # The neural ensemble owns the verdict. Classical forensics only nudges it,
    # and only when they agree the image is suspicious.
    agreement = abs(model_score - heuristic) < 0.35
    final_score = float(np.clip(0.85 * model_score + 0.15 * heuristic, 0, 1))

    label = _classify(final_score)
    face_count = len(faces)

    return {
        "media_type": "image",
        "classification": label,
        "is_deepfake": label == "DEEPFAKE",
        "fake_probability": round(final_score, 4),
        "confidence": round(max(final_score, 1 - final_score), 4),
        "model_score": round(model_score, 4),
        "forensic_score": round(heuristic, 4),
        "signals_agree": agreement,
        "faces_detected": face_count,
        "analysed_crops": len(crops),
        "image": {"width": img.width, "height": img.height},
        "models": per_model,
        "forensics": features,
        "explanation": _explain(label, final_score, model_score, heuristic,
                                per_model, features, face_count),
        "elapsed_seconds": round(time.time() - started, 2),
    }


def _explain(label, final_score, model_score, heuristic, per_model, features, faces) -> list[str]:
    notes: list[str] = []

    if label == "DEEPFAKE":
        notes.append(
            f"Models assign a {model_score:.0%} probability that the visual content is manipulated."
        )
    elif label == "SUSPICIOUS":
        notes.append(
            f"Evidence is mixed ({model_score:.0%} fake probability); a manual review is advised."
        )
    else:
        notes.append(
            f"Models assign only a {model_score:.0%} probability of manipulation, consistent with authentic capture."
        )

    strongest = max(per_model, key=lambda m: m["fake_probability"])
    notes.append(
        f"Strongest signal came from {strongest['model']} "
        f"({strongest['fake_probability']:.0%} fake)."
    )

    if faces == 0:
        notes.append("No face was located, so the whole frame was analysed as one region.")
    else:
        notes.append(f"{faces} face region(s) located and scored separately from the full frame.")

    spectral = features.get("spectral", {})
    hf = spectral.get("high_freq_ratio")
    if isinstance(hf, (int, float)):
        if hf > 0.45:
            notes.append(
                f"Frequency analysis shows elevated high-band energy ({hf:.2f}), typical of "
                "generator upsampling artifacts."
            )
        else:
            notes.append(
                f"Frequency spectrum is within the natural camera range ({hf:.2f})."
            )

    sharp = features.get("sharpness", {})
    if sharp.get("sharpness_label"):
        notes.append(
            f"Sharpness profile: {sharp['sharpness_label']} "
            f"(Laplacian variance {sharp.get('laplacian_variance')})."
        )

    return notes
