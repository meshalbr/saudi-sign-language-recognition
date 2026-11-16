# Saudi Sign Language recognition: the browser app and its API in one CPU-only image.
#
#   docker run -p 7860:7860 ghcr.io/meshalbr/saudi-sign-language-recognition
#   open http://localhost:7860
#
# Public mode is on by default (SSL_PUBLIC=1): visitors are isolated, nothing is stored, and
# teach mode and phone pairing are off. Pass -e SSL_PUBLIC=0 to get them back on your own machine.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SSL_PUBLIC=1 \
    SSL_MAX_SERVER_SESSIONS=2

# OpenCV and MediaPipe need these shared libraries even without a display.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libegl1 libgles2 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# Dependencies first, so code changes do not reinstall them. CPU-only torch is ~200 MB,
# the default CUDA build would be ~2.5 GB.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[app]"

COPY app ./app
COPY scripts ./scripts
# MediaPipe models (Google) + trained model, sign names and reference skeletons (GitHub release)
RUN python scripts/download_models.py && chown -R app:app /app

USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/labels', timeout=4)"
CMD ["python", "-m", "app.server", "--run", "runs/final_v3_face", "--no-https", "--http-port", "7860"]
