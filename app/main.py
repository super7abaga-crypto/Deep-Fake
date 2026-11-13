"""
Deepfake Multimedia Detection and Classification System
=======================================================

FastAPI entry point. One endpoint does the work:

    POST /api/analyze   multipart file -> classification result

The media type is decided from the file itself (magic bytes first, extension as
a fallback) and routed to the image, video or audio analyser.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .detectors import audio_detector, image_detector, video_detector
from .detectors.models import registry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deepfake")

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "80")) * 1024 * 1024

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".3gp"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}

app = FastAPI(
    title="Deepfake Multimedia Detection and Classification System",
    description="Upload an image, video or audio file. The system classifies it as "
                "DEEPFAKE, SUSPICIOUS or AUTHENTIC and explains why.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def sniff_media_type(filename: str, head: bytes) -> str:
    """Decide image / video / audio from the file signature, then the extension."""
    if head.startswith(b"\xff\xd8\xff"):
        return "image"
    if head.startswith(b"\x89PNG\r\n\x1a\n") or head[:4] in (b"RIFF", b"BM", b"II*\x00", b"MM\x00*"):
        return "image" if head[:4] != b"RIFF" else "video_or_audio"
    if head[4:8] == b"ftyp":
        return "video_or_audio"
    if head.startswith(b"GIF8"):
        return "image"
    if head.startswith(b"ID3") or head.startswith(b"OggS") or head.startswith(b"fLaC"):
        return "audio"

    ext = os.path.splitext(filename.lower())[1]
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    if ext in AUDIO_EXT:
        return "audio"
    return "unknown"


def resolve_container(path: str) -> str:
    """For containers that can hold either video or audio, ask ffprobe."""
    try:
        meta = video_detector.probe(path)
    except Exception:
        return "unsupported"
    return "video" if meta.get("width") else "audio"


@app.on_event("startup")
def warm_models() -> None:
    """Load the detection models at boot so the first upload is not penalised."""
    t0 = time.time()
    try:
        registry.load()
        log.info("image models ready in %.1fs", time.time() - t0)
    except Exception as exc:
        log.error("model warm-up failed: %s", exc)
    audio_detector.audio_available()
    log.info("startup complete in %.1fs", time.time() - t0)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health():
    return {
        "status": "ok" if registry.ready else "degraded",
        "image_models_ready": registry.ready,
        "audio_model_ready": audio_detector.audio_available(),
        "supported": {
            "image": sorted(IMAGE_EXT),
            "video": sorted(VIDEO_EXT),
            "audio": sorted(AUDIO_EXT),
        },
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@app.get("/api/models")
def models():
    info = registry.info()
    info["audio_model"] = {
        "model": audio_detector.MODEL_ID,
        "ready": audio_detector.audio_available(),
        "labels": {"0": "fake", "1": "real"},
        "window_seconds": audio_detector.WINDOW_SECONDS,
    }
    return info


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file supplied")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is larger than the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        )

    media_type = sniff_media_type(file.filename, content[:32])
    log.info("analysing %s (%s, %.1f MB)", file.filename, media_type, len(content) / 1e6)

    suffix = os.path.splitext(file.filename)[1] or ""
    tmp_path = None
    try:
        if media_type == "image":
            result = image_detector.analyse_image(content)
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(content)
            tmp.close()
            tmp_path = tmp.name

            if media_type in ("video_or_audio", "unknown"):
                media_type = resolve_container(tmp_path)

            if media_type == "unsupported":
                raise HTTPException(
                    status_code=415,
                    detail="This file could not be read as an image, video or audio "
                           "file. Allowed: jpg, png, webp, bmp, gif, mp4, mov, avi, "
                           "mkv, webm, wav, mp3, m4a, ogg, flac.",
                )

            if media_type == "video":
                result = video_detector.analyse_video(tmp_path)
            elif media_type == "audio":
                result = audio_detector.analyse_audio(tmp_path)
            else:
                raise HTTPException(
                    status_code=415,
                    detail="Unsupported file type. Upload an image, video or audio file.",
                )
    except HTTPException:
        raise
    except Exception as exc:  # surface a clean error instead of a stack trace
        log.exception("analysis failed")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    result["filename"] = file.filename
    result["file_size_bytes"] = len(content)
    result["media_type"] = result.get("media_type", media_type)
    return JSONResponse(result)


@app.get("/api/demo/{sample}")
def demo(sample: str):
    """Scoring helper for the evaluation script and for quick sanity checks."""
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples")
    path = os.path.join(base, sample)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Sample not found")
    with open(path, "rb") as fh:
        content = fh.read()
    media_type = sniff_media_type(sample, content[:32])
    if media_type == "image":
        return image_detector.analyse_image(content)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(sample)[1])
    tmp.write(content)
    tmp.close()
    try:
        if media_type == "video":
            return video_detector.analyse_video(tmp.name)
        return audio_detector.analyse_audio(tmp.name)
    finally:
        os.unlink(tmp.name)
