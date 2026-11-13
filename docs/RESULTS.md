# Measured Results

All numbers below were produced by `evaluate.py` running the **shipped
detection code** on labelled data, not by a separate notebook. Reproduce with
the commands at the bottom.

---

## Test data

| Set | Source | Fake class | Real class | n |
|---|---|---|---|---|
| A | `34data/140k-real-fake-faces` (split of the 140k Real and Fake Faces set) | StyleGAN-generated faces | FFHQ real photographs | 36 (18 / 18) |
| B | `elsaEU/ELSA_D3` validation split | Stable Diffusion generated images | original real images | 23 (14 / 9) |
| Video | clips assembled from set A frames | 1 fake clip | 1 real clip | 2 |
| Audio | TTS synthesis vs recorded speech | 1 synthetic | 1 recorded | 2 |

Set B is fully independent of set A (different generators, different source,
different subject matter), so it acts as a held-out check rather than a second
look at the same distribution.

---

## Combined image results (n = 59)

| Metric | Value |
|---|---|
| AUC (ROC) | **0.961** |
| Accuracy ("flagged" = DEEPFAKE or SUSPICIOUS) | **94.9 %** |
| Precision | 91.4 % |
| Recall | **100 %** (0 of 32 manipulated files missed) |
| F1 | 0.955 |
| Confusion matrix | TP 32 · TN 24 · FP 3 · FN 0 |
| Mean fake probability on manipulated files | 0.739 |
| Mean fake probability on authentic files | 0.159 |

### Strict DEEPFAKE band only (≥ 0.60)

| Metric | Value |
|---|---|
| Accuracy | 91.5 % |
| Precision | 96.6 % |
| Recall | 87.5 % |
| F1 | 0.918 |
| Confusion matrix | TP 28 · TN 26 · FP 1 · FN 4 |

The four misses in this band were all scored between 0.40 and 0.60, so the
system returned them as **SUSPICIOUS** rather than letting them pass as
authentic. That is the intended behaviour of the three-way design: a confident
call, or an explicit request for review.

---

## Per-set breakdown

### Set A — GAN faces (n = 36)

| Metric | Value |
|---|---|
| AUC | 0.954 |
| Accuracy | 97.2 % (strict) / 91.7 % (flagged) |
| Precision | 94.7 % / 85.7 % |
| Recall | 100 % / 100 % |
| Mean fake probability | fakes 0.797 · reals 0.196 |

One authentic portrait scored 0.878 and was called DEEPFAKE. Images in this
dataset are re-uploads and a small amount of label noise is likely; either way
it is a genuine false positive and is reported as such.

### Set B — diffusion images (n = 23)

| Metric | Value |
|---|---|
| AUC | **1.000** |
| Accuracy | 82.6 % (strict) / **100 %** (flagged) |
| Precision | 100 % / 100 % |
| Recall | 71.4 % / **100 %** |
| Mean fake probability | fakes 0.665 · reals 0.086 |

Diffusion-generated images scored lower than StyleGAN faces (mean 0.665 vs
0.797) but ranked perfectly: every authentic image scored below every
manipulated one. Ranking quality (AUC) and threshold placement are different
things, and this set shows why the SUSPICIOUS band earns its place.

---

## Per-model comparison (set A, 36 images)

Why this ensemble: each model was measured alone before being trusted in a vote.

| Model | AUC | Accuracy @ 0.5 |
|---|---|---|
| CommunityForensics-DeepfakeDet-ViT | **1.000** | 94.4 % |
| Wvolf/ViT_Deepfake_Detection | 0.716 | 72.2 % |
| Ateeqq/ai-vs-human-image-detector | 0.679 | 69.4 % |
| prithivMLmods/Deep-Fake-Detector-v2-Model | 0.543 | 58.3 % — **rejected** |
| prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2 | 0.410 | 44.4 % — **rejected** |

Two candidates that rank highly on Hugging Face downloads performed at or below
chance on this data and were removed from the ensemble. The final weights
(0.70 / 0.20 / 0.10) reflect measured single-model quality, re-normalised over
whichever models load successfully at runtime.

---

## Video and audio

| Modality | n | Accuracy | Notes |
|---|---|---|---|
| Video | 2 | 2/2 | Fake clip: 12/12 frames flagged, mean 0.945. Real clip: mean 0.352, 0/12 frames strongly flagged |
| Audio | 2 | 2/2 | Real speech 0.000 · TTS 1.000 |

These sets are **too small to quote as accuracy figures**. They establish that
the video and audio pipelines run end to end and produce the right answer on
the samples available. Collecting 20+ clips per class is the obvious next step
before claiming a video or audio accuracy number.

---

## Timing (CPU, free-tier class hardware)

| Operation | Time |
|---|---|
| Startup (weights cached) | ~4 s |
| Image, 1 face | 2.5 – 2.8 s |
| Video, 12 frames | ~30 s |
| Video, 12 frames + audio track | ~32 s |
| Audio, 5 s clip | ~1 s |

---

## Reproduce

```bash
export HF_HOME=./model_cache

# image accuracy
python evaluate.py --data ./test_media/setA --media image --out res_A.csv
python evaluate.py --data ./test_media/setB --media image --out res_B.csv

# video and audio
python evaluate.py --data ./test_media/setV --media video  --out res_V.csv
python evaluate.py --data ./test_media/setA --media audio  --out res_audio.csv
```
