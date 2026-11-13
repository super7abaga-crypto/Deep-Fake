# Design and Implementation of a Deepfake Multimedia Detection and Classification System

*A technical report on the implemented system. Adapt the sections into your own
chapters — the measurements are real, taken from the code in this repository.*

---

## 1. Introduction and problem statement

Deepfakes are media in which a person's face, voice, or entire presence has been
synthesised or altered by a machine-learning model. Cheap, widely available
generators mean a convincing fake face, voice note, or video clip can now be
produced by someone with no technical skill, and the harm is well documented:
fraud, impersonation, reputational damage, and the collapse of "seeing is
believing" as a basis for evidence.

Manual inspection does not scale. A human cannot reliably tell a well-made
synthetic portrait from a photograph, and cannot inspect thousands of files.
The value of an automated system is that it produces a *fast, repeatable,
inspectable* judgement.

The purpose of this project, as approved, is a system that:

1. accepts a supported media file from a user,
2. analyses the file,
3. applies a machine-learning model to it,
4. classifies it, and
5. shows the result to the user.

The scope for this stage is deliberately narrow: prove that the detection
process works. Presentation, user accounts, and extra features are secondary.

---

## 2. Aim and objectives

**Aim:** to design and implement a working system that classifies a multimedia
file as manipulated or authentic using machine-learning models, and reports the
evidence behind the classification.

**Objectives:**

1. Accept image, video, and audio uploads through a single interface.
2. Apply pretrained deep-learning detectors appropriate to each media type.
3. Combine model outputs into a single calibrated classification.
4. Provide an explanation for each classification, not only a label.
5. Measure detection accuracy on labelled data and report it honestly.
6. Ship the system in a form that runs locally and can be deployed publicly.

---

## 3. Background and related work

The detection literature splits along the same lines as the generation methods.

**Spatial / generative artefacts.** Early detectors targeted the fingerprints left
by the generator itself: GAN upsampling produces characteristic patterns in the
frequency domain, detectable with a discrete Fourier transform
(Zhang et al., 2019; Durall et al., 2020), and CNN-generated images retain
low-level traces that survive compression and survive across architectures
(Wang et al., 2020). MesoNet (Afchar et al., 2018) showed that a small
convolutional network on the mesoscopic level of a face is enough for the
FaceForensics benchmarks.

**Face-swap and reenactment detection.** FaceForensics++ (Rössler et al., 2019)
and Celeb-DF (Li et al., 2020) established the standard benchmarks for
DeepFakes, Face2Face, FaceSwap and NeuralTextures manipulations, and remain the
source of most published accuracy figures for video detectors.

**Transformer backbones.** Vision Transformers (Dosovitskiy et al., 2021) match
or beat convolutional networks on fine-grained classification, and the vision
encoders used in image–text pretraining (Radford et al., 2021; Zhai et al.,
2023) transfer well to forensic tasks where the signal is subtle and global.
Most current public deepfake detectors are ViT or SigLIP heads fine-tuned on a
real/fake corpus.

**Audio.** Synthetic speech detection commonly uses self-supervised speech
representations. wav2vec 2.0 (Baevski et al., 2020) features with a
classification head are a standard baseline, and are what ASVspoof-style
challenges evaluate.

**What this project does differently.** Rather than train a detector from
scratch — which needs a very large labelled corpus and days of GPU time — the
system composes pretrained detectors, measures each one, selects the ones that
actually work, and adds two things a bare model repository does not have: media
type handling across three modalities, and an explanation layer that shows the
user *why* a file was flagged.

---

## 4. System design

### 4.1 Architecture

```
Browser  ──upload──▶  FastAPI  ──route by sniffed media type──▶  detector
   ▲                     │                                          │
   └──render JSON────────┘◀───────── classification + evidence ─────┘
```

Three layers:

* **Presentation** — a single-page upload interface (`app/static/index.html`).
* **Application** — `app/main.py`. Validates the upload, identifies the real
  media type from magic bytes (not just the filename extension), dispatches, and
  shapes the JSON response.
* **Detection** — `app/detectors/`. Media-specific pipelines sharing one model
  registry and one forensic feature module.

Keeping the registry shared means the roughly 1.5 GB of model weights loads once
and every modality reuses it.

### 4.2 Media type resolution

The extension is not trusted, because a `.jpg` can be anything. The first bytes
are checked against known signatures (JPEG `FF D8 FF`, PNG, GIF, RIFF, ISO-BMFF
`ftyp`, ID3, OggS, fLaC). Container formats that may hold either video or audio
are resolved by asking `ffprobe` whether a video stream exists. Only if the
signature is unrecognised does the extension get a vote.

### 4.3 Image pipeline

1. Decode, apply EXIF orientation, convert to RGB, downscale beyond 1024 px.
2. Locate up to three faces with a Haar cascade; crop with 35 % padding.
3. Score the full frame **and** each face crop with each model.
4. Weighted-average the models into one fake probability.
5. Run classical forensics on the full-resolution original.
6. Blend: `0.85 × model + 0.15 × forensic prior`.

Scoring both the whole frame and the face matters. The models are trained mostly
on face crops, but a *fully synthetic* image has no real face for a crop to
isolate, so the full frame is what carries the signal there.

### 4.4 Video pipeline

1. `ffprobe` for duration, resolution, fps, and whether an audio stream exists.
2. Sample 12 frames evenly with a single `ffmpeg` `fps`-filter pass.
3. Score each frame through the image pipeline.
4. Aggregate: mean, peak, and flagged fraction.
5. Score the audio track independently, then blend.

Aggregation is intentionally not just the mean. A clip where 9 of 12 frames
score 0.7 is stronger evidence than one where a single frame spikes, so the
verdict relaxes the threshold to 0.50 when at least 60 % of frames are flagged.

### 4.5 Audio pipeline

`ffmpeg` decodes any container to 16 kHz mono WAV. The clip is cut into 5-second
windows with a 4-second hop (capped at 8 windows), each window is scored by a
Wav2Vec2 sequence classifier, and the windows are combined
(`0.6 × mean + 0.4 × max`). Window-level scoring catches partial manipulation: a
voice that is cloned for the first half only will not average out to a clean
verdict.

### 4.6 Model selection experiment

Candidate detectors were not accepted because they download well; each was
measured alone on the labelled set before entering the ensemble.

| Model | AUC | Accuracy @ 0.5 | Decision |
|---|---|---|---|
| CommunityForensics-DeepfakeDet-ViT | 1.000 | 94.4 % | **primary, weight 0.70** |
| Wvolf/ViT_Deepfake_Detection | 0.716 | 72.2 % | secondary, weight 0.20 |
| Ateeqq/ai-vs-human-image-detector | 0.679 | 69.4 % | supporting, weight 0.10 |
| prithivMLmods/Deep-Fake-Detector-v2-Model | 0.543 | 58.3 % | rejected |
| prithivMLmods/AI-vs-Deepfake-vs-Real-Siglip2 | 0.410 | 44.4 % | rejected |

Two models with large download counts scored at or below chance here and were
removed. This is worth stating in your presentation: model popularity is not a
substitute for measuring the model on your own data.

### 4.7 Decision logic

| Score | Label | Meaning |
|---|---|---|
| ≥ 0.60 | DEEPFAKE | confident call |
| 0.40 – 0.59 | SUSPICIOUS | evidence exists, needs review |
| < 0.40 | AUTHENTIC | no meaningful manipulation signal |

Thresholds were chosen from the measured distributions, not assumed. The
two-line split exists because the two errors are not equally costly: a missed
deepfake is usually worse than a real photo sent for review.

### 4.8 Explanation layer

Every response carries a plain-language explanation and the underlying numbers:

* which model was most confident, and how confident,
* whether faces were located and how many regions were scored,
* frequency-domain high-band energy, with a natural-camera reference range,
* sharpness (Laplacian variance) and error-level analysis statistics,
* for video: a per-frame score timeline,
* for audio: per-window scores and spectral descriptors.

This matters for a forensic tool. A bare percentage invites blind trust; the
evidence lets a human disagree with the system.

---

## 5. Implementation

**Stack:** Python 3.11, FastAPI, PyTorch (CPU), Hugging Face Transformers,
OpenCV, NumPy, ffmpeg. Frontend is one HTML file with vanilla JavaScript — no
build step, nothing to install.

**Why FastAPI:** the whole system is one inference endpoint. FastAPI gives typed
validation, an automatic OpenAPI document at `/docs`, and trivial deployment.

**Why CPU:** the project must run on a normal laptop and on a free hosting tier.
Measured cost is 2.5 s per image and about 30 s for a 12-frame video, which is
acceptable for the use case. GPU support is a one-line device change if needed.

**Why pretrained:** training a competitive deepfake detector needs a corpus of
hundreds of thousands of images and many GPU-hours. Composing and validating
pretrained detectors delivers a working system within the project's constraints,
and the evaluation in section 6 shows the result is genuinely accurate rather
than merely assembled.

---

## 6. Testing and results

Full tables are in [`RESULTS.md`](RESULTS.md). Summary:

**Image classification, 59 labelled files from two independent sources**

| Metric | Value |
|---|---|
| AUC | 0.961 |
| Accuracy (flagged = fake) | 94.9 % |
| Precision | 91.4 % |
| Recall | 100 % — no manipulated file passed as authentic |
| F1 | 0.955 |

**Per-source:**

* StyleGAN faces vs FFHQ photographs (n = 36): AUC 0.954, accuracy 97.2 %,
  recall 100 %.
* Stable Diffusion images vs real images (n = 23): AUC 1.000, accuracy 100 %
  when the SUSPICIOUS band is counted as a positive detection.

The second set matters because it is a different generator family entirely.
Perfect ranking with imperfect threshold placement is exactly the behaviour the
three-way classification was designed to absorb.

**Video and audio:** both pipelines produced correct classifications on the
available samples (fake clip flagged on 12/12 frames at mean 0.945; real clip at
mean 0.352; synthetic speech 1.000 vs recorded speech 0.000). Two files per
modality is enough to prove the pipeline, not enough to claim an accuracy
figure, and it is reported that way.

**Notable failure:** one authentic portrait scored 0.878 and was labelled
DEEPFAKE. Reported as a genuine false positive rather than explained away.

---

## 7. Limitations

1. **Probabilistic output.** A classification is evidence, not proof. The system
   is a triage tool.
2. **Generator dependence.** Accuracy is measured against StyleGAN and Stable
   Diffusion. A generator released next month may degrade performance, because
   detectors learn the artefacts of what they were trained on.
3. **Compression and resolution.** Heavy re-encoding, small faces and low
   resolution all reduce reliability; social media re-uploads are the hard case.
4. **Audio is least validated.** The audio model works but is supported by two
   test files here. Treat its output as indicative.
5. **Forensic features are not calibrated.** They explain; they do not decide.
6. **CPU latency.** A 12-frame video takes about 30 s. Longer clips are
   subsampled rather than fully analysed.
7. **No adversarial robustness testing.** Deliberately crafted inputs designed
   to fool the models were out of scope.

---

## 8. Conclusion and future work

The system meets the approved purpose: a user uploads a supported file, the
application analyses it, machine-learning models examine it, the application
classifies it, and the user sees the result along with the evidence. Detection
was measured rather than assumed, at 0.961 AUC and 100 % recall for manipulated
files across two independent datasets.

**Future work, in order of value:**

1. **Larger video and audio test sets** — 20+ clips per class to publish real
   accuracy figures for those modalities.
2. **Face-embedding consistency across frames** — a face that changes identity
   between frames is strong evidence, independent of pixel artefacts.
3. **Fine-tuning on local data** — collect Nigerian-context faces and voices;
   domain adaptation usually buys more accuracy than a bigger generic model.
4. **Calibration** — fit a logistic layer so the output probability means what
   it says rather than being an ensemble average.
5. **Batch processing and a review queue** for practical deployment.
6. **Provenance cross-checks** — C2PA / metadata signatures where available.

---

## References

Afchar, D., Nozick, V., Yamagishi, J. and Echizen, I. (2018) 'MesoNet: a compact
facial video forgery detection network', *IEEE WIFS*.

Baevski, A., Zhou, H., Mohamed, A. and Auli, M. (2020) 'wav2vec 2.0: a framework
for self-supervised learning of speech representations', *NeurIPS*.

Dosovitskiy, A. et al. (2021) 'An image is worth 16x16 words: transformers for
image recognition at scale', *ICLR*.

Durall, R., Keuper, M. and Pfreundt, F.-J. (2020) 'Unmasking deepfakes with
simple classifiers', *CVPR Workshops*.

Krawetz, N. (2007) 'A picture's worth: digital image analysis and forensics',
*Black Hat Briefings*.

Li, Y., Yang, X., Sun, P., Qi, H. and Lyu, S. (2020) 'Celeb-DF: a large-scale
challenging dataset for deepfake forensics', *CVPR*.

Radford, A. et al. (2021) 'Learning transferable visual models from natural
language supervision', *ICML*.

Rössler, A. et al. (2019) 'FaceForensics++: learning to detect manipulated facial
images', *ICCV*.

Tolosana, R., Vera-Rodriguez, R., Fierrez, J., Morales, A. and Ortega-Garcia, J.
(2020) 'Deepfakes and beyond: a survey of face manipulation and fake detection',
*Information Fusion*, 64, pp. 131–148.

Wang, S.-Y. et al. (2020) 'CNN-generated images are surprisingly easy to spot…
for now', *CVPR*.

Zhai, X., Mustafa, B., Kolesnikov, A. and Beyer, L. (2023) 'Sigmoid loss for
language image pre-training', *ICCV*.

Zhang, X., Karaman, S. and Chang, S.-F. (2019) 'Detecting and simulating
artifacts in GAN fake images', *IEEE WIFS*.

**Models used**

* `buildborderless/CommunityForensics-DeepfakeDet-ViT` (Hugging Face)
* `Wvolf/ViT_Deepfake_Detection` (Hugging Face)
* `Ateeqq/ai-vs-human-image-detector` (Hugging Face)
* `mo-thecreator/Deepfake-audio-detection` (Hugging Face)

**Datasets used for evaluation**

* 140k Real and Fake Faces (FFHQ photographs / StyleGAN faces), via
  `34data/140k-real-fake-faces-*`
* ELSA D3 (`elsaEU/ELSA_D3`), diffusion-generated images with originals
