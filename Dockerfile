FROM python:3.12.7-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/seriesrag/.cache/huggingface

WORKDIR /app

# Install only the declared runtime project and its application package.
COPY pyproject.toml README.md ./
COPY app ./app
RUN python -m pip install --no-cache-dir .

# Keep the writable model cache owned by an unprivileged runtime user.
RUN groupadd --system seriesrag \
    && useradd --system --gid seriesrag --create-home seriesrag \
    && mkdir -p "${HF_HOME}" \
    && chown -R seriesrag:seriesrag /home/seriesrag

USER seriesrag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
