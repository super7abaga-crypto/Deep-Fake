FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/model_cache \
    MAX_UPLOAD_MB=80

# ffmpeg is required for video and audio decoding
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
 && apt-get clean

WORKDIR /app

# CPU-only torch keeps the image near 2 GB instead of 8 GB+
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
 && pip install -r requirements.txt

COPY app ./app
COPY samples ./samples
COPY evaluate.py ./
COPY docs ./docs

# weights download on first boot; mount a disk here in production
RUN mkdir -p /app/model_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/api/health',timeout=5).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
