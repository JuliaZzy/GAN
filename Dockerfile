FROM docker.m.daocloud.io/pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# OpenCV runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt excludes torch (already in base image)
COPY requirements.txt .
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=10
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir -r requirements.txt

COPY enhance.py setup_weights.py upscale_sd.py ./

# weights / input / output are mounted at runtime (see docker-compose.yml)
VOLUME ["/app/weights", "/app/input", "/app/output"]

ENTRYPOINT ["python", "enhance.py"]
# Default: batch-enhance everything in /app/input -> /app/output
CMD ["-i", "/app/input", "-o", "/app/output"]
