"""
Model registry.

Loads the neural detectors once, lazily, and keeps them resident. Models are
pulled from the Hugging Face cache / local `models/` directory, so after the
first provisioning run the system works fully offline.

Ensemble
    CommunityForensics-DeepfakeDet-ViT   primary   ViT trained across many
                                                   generator families (the
                                                   "community forensics" mix)
    Wvolf/ViT_Deepfake_Detection         secondary ViT fine-tuned on
                                                   Celeb-DF style face swaps
    Ateeqq/ai-vs-human-image-detector    supporting SigLIP AI-vs-human head,
                                                   catches fully synthetic frames

Each member reports P(fake) in [0,1]; the registry blends them with the weights
below. Weights were set by measuring single-model AUC on held-out sets and are
re-normalised over whichever models actually load.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

DEVICE = torch.device("cpu")

ENSEMBLE = [
    {"repo": "buildborderless/CommunityForensics-DeepfakeDet-ViT",
     "name": "CommunityForensics ViT", "weight": 0.70, "outputs": "sigmoid"},
    {"repo": "Wvolf/ViT_Deepfake_Detection",
     "name": "ViT Deepfake (Celeb-DF)", "weight": 0.20, "outputs": "softmax"},
    {"repo": "Ateeqq/ai-vs-human-image-detector",
     "name": "SigLIP AI-vs-Human", "weight": 0.10, "outputs": "softmax"},
]

FAKE_TOKENS = ("deepfake", "deep fake", "fake", "ai", "artificial", "synthetic",
               "generated", "manipulated", "forged")


@dataclass
class ModelVerdict:
    name: str
    repo: str
    fake_probability: float
    label: str
    raw_scores: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "model": self.name,
            "repo": self.repo,
            "fake_probability": round(self.fake_probability, 4),
            "predicted_label": self.label,
            "scores": {k: round(float(v), 4) for k, v in self.raw_scores.items()},
        }


def _fake_index(id2label: dict) -> tuple[int, str]:
    """Which output index means 'manipulated' for a softmax classifier?"""
    for idx in sorted(id2label):
        low = str(id2label[idx]).lower()
        if any(tok in low for tok in FAKE_TOKENS):
            return idx, str(id2label[idx])
    return 1, str(id2label.get(1, "fake"))


class ImageModel:
    def __init__(self, repo: str, name: str, weight: float, outputs: str):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.repo, self.name, self.weight, self.outputs = repo, name, weight, outputs
        kwargs = {}
        if os.getenv("HF_HOME"):
            kwargs["cache_dir"] = os.getenv("HF_HOME")

        self.processor = AutoImageProcessor.from_pretrained(repo, **kwargs)
        self.model = AutoModelForImageClassification.from_pretrained(repo, **kwargs)
        self.model.eval().to(DEVICE)

        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.num_labels = int(self.model.config.num_labels or 1)
        self.sigmoid = self.num_labels == 1 or outputs == "sigmoid"
        self.fake_idx, self.fake_label = (0, "deepfake") if self.sigmoid else _fake_index(self.id2label)

    @torch.inference_mode()
    def predict(self, images: Iterable[Image.Image]) -> ModelVerdict:
        images = list(images)
        inputs = self.processor(images=images, return_tensors="pt")
        logits = self.model(**inputs).logits

        if self.sigmoid:
            probs = torch.sigmoid(logits).reshape(-1).numpy()
            fake_probs = probs
            raw = {"fake": float(np.mean(probs)), "real": float(1 - np.mean(probs))}
        else:
            probs = torch.softmax(logits, dim=-1).numpy()
            fake_probs = probs[:, self.fake_idx]
            mean_scores = probs.mean(axis=0)
            raw = {self.id2label[i]: float(mean_scores[i]) for i in range(len(mean_scores))}

        fake_prob = float(np.mean(fake_probs))
        label = self.fake_label if fake_prob >= 0.5 else "authentic"
        return ModelVerdict(self.name, self.repo, fake_prob, label, raw)


class FaceCropper:
    """Haar cascade face detection: a light, dependency-free way to give the
    models a tight view of the subject alongside the full frame."""

    def __init__(self):
        self.available = False
        try:
            import cv2

            self.cv2 = cv2
            cascade = getattr(cv2, "CascadeClassifier", None)
            if cascade is None:
                log.warning("cv2 build has no CascadeClassifier; face crops disabled")
                return
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cascade(path)
            self.available = not self.cascade.empty()
        except Exception as exc:
            log.warning("face detector unavailable: %s", exc)

    def detect(self, img: Image.Image, max_faces: int = 3) -> list[Image.Image]:
        if not self.available:
            return []
        gray = np.asarray(img.convert("L"))
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        if len(faces) == 0:
            return []
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:max_faces]
        crops, (w_img, h_img) = [], img.size
        for (x, y, w, h) in faces:
            pad_w, pad_h = int(w * 0.35), int(h * 0.35)
            box = (max(0, x - pad_w), max(0, y - pad_h),
                   min(w_img, x + w + pad_w), min(h_img, y + h + pad_h))
            crops.append(img.crop(box))
        return crops


class ModelRegistry:
    def __init__(self):
        self.models: list[ImageModel] = []
        self.loaded = False
        self.errors: list[str] = []
        self._face = None

    @property
    def face(self) -> FaceCropper:
        if self._face is None:
            self._face = FaceCropper()
        return self._face

    def load(self) -> None:
        if self.loaded:
            return
        for spec in ENSEMBLE:
            try:
                self.models.append(ImageModel(**spec))
                log.info("loaded %s", spec["repo"])
            except Exception as exc:
                self.errors.append(f"{spec['repo']}: {exc}")
                log.warning("load failed %s: %s", spec["repo"], exc)
        if not self.models:
            raise RuntimeError("No detection models could be loaded: " + "; ".join(self.errors))
        total = sum(m.weight for m in self.models)
        for m in self.models:
            m.weight /= total
        self.loaded = True

    @property
    def ready(self) -> bool:
        return self.loaded and bool(self.models)

    def info(self) -> dict:
        return {
            "ready": self.ready,
            "device": str(DEVICE),
            "models": [
                {"name": m.name, "repo": m.repo, "weight": round(m.weight, 3),
                 "outputs": m.outputs}
                for m in self.models
            ],
            "face_detector": "haarcascade_frontalface_default"
            if self.face.available else "unavailable",
            "load_errors": self.errors,
        }


registry = ModelRegistry()
