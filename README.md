# Deepfake Multimedia Detection and Classification System

Upload a supported media file (image, video, or audio). The system analyses it,
runs machine-learning models over it, classifies it as **DEEPFAKE**,
**SUSPICIOUS** or **AUTHENTIC**, and shows the evidence behind that decision.

This is the working core of the project: the detection process, not the styling.

---

## What it does

| Step | Where it happens |
|---|---|
| 1. User uploads a media file | `app/static/index.html` (drag & drop) → `POST /api/analyze` |
| 2. Application analyses the file | `app/main.py` sniffs the real media type, decodes it, routes it |
| 3. A machine-learning model examines it | `app/detectors/*` (ViT / SigLIP / Wav2Vec2 models) |
| 4. Application displays classification | JSON response, rendered in the UI |
| 5. User sees the result | Verdict badge, probability, model scores, explanation, frame timeline |

Supported input:

* **Images** — jpg, jpeg, png, bmp, webp, tif, tiff, gif
* **Video** — mp4, mov, avi, mkv, webm, m4v, mpeg, 3gp
* **Audio** — wav, mp3, m4a, aac, flac, ogg, opus, wma
* Upload limit: 80 MB (configurable)

---

## Quick start

```bash
# 1. install
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. let the models download once (needs one-time internet access)
export HF_HOME=./model_cache                            # Windows: set HF_HOME=./model_cache
python -c "from app.detectors.models import registry; registry.load(); print('ready')"
python -c "from app.detectors.audio_detector import audio_available; print(audio_available())"

# 3. run
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000> and drop a file in.

`ffmpeg` must be installed and on your PATH for video and audio files
(`apt install ffmpeg`, `brew install ffmpeg`, or on Windows download the build
from ffmpeg.org and add its `bin` folder to PATH).

### First-run download

The weights total roughly 1.5 GB and are cached under `HF_HOME`. After that the
system runs **fully offline** — useful when you demo without reliable internet.

| Model | Task | Size |
|---|---|---|
| `buildborderless/CommunityForensics-DeepfakeDet-ViT` | primary deepfake detector | ~350 MB |
| `Wvolf/ViT_Deepfake_Detection` | face-swap detector (Celeb-DF style) | ~330 MB |
| `Ateeqq/ai-vs-human-image-detector` | AI-generated vs human image | ~370 MB |
| `mo-thecreator/Deepfake-audio-detection` | synthetic speech detector | ~380 MB |

---

## API

### `POST /api/analyze`

```bash
curl -F "file=@some_face.jpg" http://localhost:8000/api/analyze
```

```json
{
  "media_type": "image",
  "classification": "DEEPFAKE",
  "fake_probability": 0.8792,
  "confidence": 0.8792,
  "is_deepfake": true,
  "faces_detected": 1,
  "models": [
    {"model": "CommunityForensics ViT", "fake_probability": 1.0},
    {"model": "ViT Deepfake (Celeb-DF)", "fake_probability": 0.9161},
    {"model": "SigLIP AI-vs-Human", "fake_probability": 0.9958}
  ],
  "forensics": { "spectral": {...}, "ela": {...}, "sharpness": {...}, "colour": {...} },
  "explanation": ["...", "..."],
  "elapsed_seconds": 2.52
}
```

Video responses add `timeline` (per-frame scores), `visual_score`,
`flagged_frame_fraction` and an `audio` sub-result when the clip has sound.
Audio responses add `window_scores` (per 5-second window), `windows_scored`
and acoustic descriptors.

### Other endpoints

* `GET /api/health` — status plus supported extensions
* `GET /api/models` — loaded models, weights, device
* `GET /api/demo/{file}` — score a bundled sample from `samples/`
* `GET /docs` — auto-generated Swagger UI (FastAPI)

---

## How detection works

Full reasoning, model choices and measured results are in
[`docs/REPORT.md`](docs/REPORT.md) and [`docs/RESULTS.md`](docs/RESULTS.md).

Short version:

1. **Image** — the frame and up to three detected face crops are scored by three
   transformer models; the weighted ensemble score is blended with classical
   forensic measurements (frequency spectrum, compression history, sharpness).
2. **Video** — 12 frames sampled evenly, each scored as an image, then
   aggregated by mean/peak/flagged-fraction; the audio track is analysed too.
3. **Audio** — Wav2Vec2 scores 5-second windows, aggregated over the clip.

`DEEPFAKE` ≥ 0.60, `SUSPICIOUS` ≥ 0.40, otherwise `AUTHENTIC`.

---

## Measuring accuracy

```bash
# folder layout: data/fake/*.jpg and data/real/*.jpg
python evaluate.py --data ./my_dataset --media image --out results.csv
python evaluate.py --data ./my_dataset --media video
python evaluate.py --data ./my_dataset --media audio
```

This prints accuracy, precision, recall, F1, AUC and the confusion matrix, and
writes per-file predictions to CSV. Use it to produce the numbers in your own
report rather than quoting the ones here.

---

## Deploying

A `Dockerfile` and `render.yaml` are included.

**Render (free tier works):**

1. Push this folder to a GitHub repo.
2. Render → New → Web Service → connect the repo. Render reads `render.yaml`,
   or set manually:
   * Build: `pip install -r requirements.txt`
   * Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   * Health check path: `/api/health`
3. Add a **persistent disk** mounted at `/app/model_cache` and set
   `HF_HOME=/app/model_cache`, otherwise the 1.5 GB of weights re-download on
   every deploy.

**Notes for a live demo:** CPU inference is ~2.5 s per image and ~30 s per
video clip on a free instance. Warm the app once (open the page and score a
sample) before you present.

---

## Project layout

```
deepfake-detector/
├── app/
│   ├── main.py                  FastAPI app, upload endpoint, media routing
│   ├── detectors/
│   │   ├── models.py            model registry, ensemble weights, face cropper
│   │   ├── image_detector.py    image pipeline + explanation builder
│   │   ├── video_detector.py    frame sampling, aggregation, audio track
│   │   ├── audio_detector.py    Wav2Vec2 windowed speech detector
│   │   └── forensics.py         FFT / ELA / sharpness / colour measurements
│   └── static/index.html        upload UI
├── samples/                     bundled demo files
├── docs/REPORT.md               design + evaluation write-up
├── docs/RESULTS.md              measured results tables
├── evaluate.py                  accuracy harness
├── requirements.txt
├── Dockerfile
└── render.yaml
```

---

## Limitations (state these in your report)

* Detection is **probabilistic**. A verdict is evidence, not proof.
* Accuracy depends heavily on the generator. Results measured here are on GAN
  faces and diffusion-generated images; a brand-new generator will degrade.
* Heavy compression, low resolution and small faces all reduce reliability.
* Audio detection is scored on short clips and is the least validated part of
  the system.
* The classical forensic features are explainability aids, not calibrated
  classifiers.
