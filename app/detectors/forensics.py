"""
Classical (non-neural) forensic feature extraction.

These signals do not decide the verdict on their own. They exist so the system
can *explain* its answer: a deepfake usually leaves traces in the frequency
domain, in JPEG compression history, and in local noise consistency.

Reference ideas:
  - FFT spectral analysis of GAN fingerprints (Zhang et al., CVPR 2019)
  - Error level analysis for resampling / splicing (Krawetz, 2007)
  - Noise / blur consistency across face and background regions
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _to_gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def spectral_artifacts(img: Image.Image) -> dict:
    """High-frequency energy profile. GAN/diffusion upsampling leaves a
    characteristic fall-off and periodic peaks in the power spectrum."""
    gray = _to_gray_array(img)
    # normalise to a fixed power-of-two grid so results are comparable
    size = 256
    g = np.asarray(img.convert("L").resize((size, size), Image.BILINEAR), dtype=np.float32)
    g = g - g.mean()
    spec = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    total = spec.sum() + 1e-12
    bands = {
        "low": (r < 0.10 * size),
        "mid": (r >= 0.10 * size) & (r < 0.30 * size),
        "high": (r >= 0.30 * size) & (r < 0.50 * size),
        "very_high": (r >= 0.50 * size),
    }
    energies = {k: float(spec[m].sum() / total) for k, m in bands.items()}

    # ratio of high band to the smooth mid band: synthetic upsampling tends to
    # leave this elevated relative to camera-captured content
    hf_ratio = energies["high"] / (energies["mid"] + 1e-12)
    peak_ratio = float(np.percentile(spec, 99.9) / (np.median(spec) + 1e-12))

    return {
        "band_energy": energies,
        "high_freq_ratio": round(float(hf_ratio), 4),
        "spectral_peak_ratio": round(peak_ratio, 2),
        "mean_gray": round(float(gray.mean()), 2),
    }


def error_level_analysis(img: Image.Image, quality: int = 90) -> dict:
    """Re-save at a known JPEG quality and measure residual differences.
    Regions spliced or generated separately re-compress differently."""
    from io import BytesIO

    rgb = img.convert("RGB")
    buf = BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    a = np.asarray(rgb, dtype=np.float32)
    b = np.asarray(recompressed, dtype=np.float32)
    diff = np.abs(a - b)
    ela = diff.mean(axis=2)

    h, w = ela.shape
    cy, cx = h // 2, w // 2
    ch, cw = max(h // 4, 1), max(w // 4, 1)
    central = ela[cy - ch: cy + ch, cx - cw: cx + cw]

    return {
        "ela_mean": round(float(ela.mean()), 3),
        "ela_std": round(float(ela.std()), 3),
        "ela_center_vs_edge": round(
            float(central.mean() / (ela.mean() + 1e-6)), 3
        ),
    }


def sharpness_and_noise(img: Image.Image) -> dict:
    """Laplacian variance (blur) and residual noise level."""
    import cv2

    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur = float(lap.var())

    # high-pass residual as a proxy for sensor noise
    blur_img = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    residual = gray.astype(np.float32) - blur_img
    noise = float(residual.std())

    return {
        "laplacian_variance": round(blur, 2),
        "noise_sigma": round(noise, 3),
        "sharpness_label": (
            "sharp" if blur > 400 else "moderate" if blur > 80 else "soft"
        ),
    }


def colour_consistency(img: Image.Image) -> dict:
    """Channel correlation and saturation statistics. Face-swap pipelines often
    leave the blended region with mismatched chroma statistics."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    def corr(x, y):
        x = x.ravel() - x.mean()
        y = y.ravel() - y.mean()
        denom = (np.linalg.norm(x) * np.linalg.norm(y)) + 1e-9
        return float((x @ y) / denom)

    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)

    return {
        "rg_correlation": round(corr(r, g), 4),
        "gb_correlation": round(corr(g, b), 4),
        "mean_saturation": round(float(sat.mean()), 4),
        "saturation_std": round(float(sat.std()), 4),
    }


def extract_all(img: Image.Image) -> dict:
    """Run every classical check and return one explainable bundle."""
    features: dict = {}
    for name, fn in (
        ("spectral", spectral_artifacts),
        ("ela", error_level_analysis),
        ("sharpness", sharpness_and_noise),
        ("colour", colour_consistency),
    ):
        try:
            features[name] = fn(img)
        except Exception as exc:  # never let a diagnostic break detection
            features[name] = {"error": str(exc)}
    return features


def heuristic_fake_score(features: dict) -> float:
    """Very rough 0..1 prior from the classical features.

    Deliberately conservative: it is used as a *supporting* signal and to
    break ties when the neural models disagree, never as the primary verdict.
    """
    score = 0.0
    spectral = features.get("spectral", {})
    sharp = features.get("sharpness", {})
    ela = features.get("ela", {})

    hf = spectral.get("high_freq_ratio")
    if isinstance(hf, (int, float)):
        # natural photos typically sit around 0.05-0.35 in this band
        score += 0.4 * float(np.clip((hf - 0.10) / 0.55, 0, 1))

    lv = sharp.get("laplacian_variance")
    if isinstance(lv, (int, float)):
        # very low blur variance alongside strong high-freq energy is a tell
        if lv < 40:
            score += 0.25
        elif lv < 120:
            score += 0.12

    ela_ratio = ela.get("ela_center_vs_edge")
    if isinstance(ela_ratio, (int, float)):
        score += 0.2 * float(np.clip((ela_ratio - 1.0) / 1.5, 0, 1))

    return float(np.clip(score, 0, 1))
