FROM python:3.10-slim

LABEL maintainer="AI Assistant"
LABEL description="Chinese STT and TTS API using Whisper and Coqui TTS."

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 
# Ensures that Python output (like print statements or logs) is sent straight to stdout 
# without being buffered first, which is useful for Docker logging.

# Install system dependencies
# git might be needed by TTS or transformers for fetching certain model components.
# ffmpeg is essential for audio processing by Whisper and other libraries.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch for CPU first.
# Check https://pytorch.org/get-started/locally/ for the latest CPU-only pip install commands.
# Using a known working version range for stability.
RUN pip install --no-cache-dir torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file and install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the image
COPY ./main.py .
# If you have other local modules, copy them as well e.g. COPY ./app_utils ./app_utils

# Set up a non-root user for security (optional but good practice)
# RUN useradd -ms /bin/bash appuser
# USER appuser
# WORKDIR /app/app # If using a subdirectory for the app

# Expose the port the app runs on
EXPOSE 8000

# Healthcheck (optional, but good for orchestration systems)
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#   CMD curl -f http://localhost:8000/ || exit 1

# Command to run the application
# Using --workers 1 as these ML models are resource-intensive. 
# Multiple workers might lead to OOM errors if each loads the models separately on CPU.
# For production, consider a more robust setup (e.g., Gunicorn managing Uvicorn workers).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
