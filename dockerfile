# GPU-ready: Python 3.10 + CUDA 12.1 + cuDNN 8
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Basic envs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System libs needed by OpenSlide/OpenCV
RUN apt-get update && apt-get install -y \
    openslide-tools libgl1 libglib2.0-0 git wget curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Make writable caches for non-root runs
ENV HOME=/home/app \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -p /home/app /tmp/matplotlib /tmp/numba_cache && chmod -R 777 /home/app /tmp/matplotlib /tmp/numba_cache

# Install Python deps from requirements.txt (best layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --root-user-action=ignore -r requirements.txt

# Install your package (pyproject makes repo importable)
COPY pyproject.toml .
COPY src/ ./src/
COPY main.py .
RUN pip install --no-deps .

# Bring the rest (configs, scripts, etc.)
COPY data/ ./data/
COPY hibou ./hibou/

# Define the entrypoint and CMD
ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
