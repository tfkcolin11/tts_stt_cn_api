# Dockerfile
FROM python:3.10-slim

LABEL maintainer="AI Assistant"
LABEL description="Chinese STT and TTS API using Whisper and Coqui TTS."

# Set environment variables for cleaner pip installs and Docker logging
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1 
ENV PIP_NO_CACHE_DIR=off
ENV PIP_NO_CACHE_DIR=yes
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip, setuptools, and wheel first
RUN pip install --upgrade pip setuptools wheel

# Clear pip cache before installing critical libraries like PyTorch
# This is a bit redundant if PIP_NO_CACHE_DIR=yes, but harmless.
RUN pip cache purge || true # Allow to fail if cache doesn't exist

# Install PyTorch, Torchaudio, and Torchvision for CPU.
# Using specific CPU versions from PyTorch's official download server.
# This command should be robust for getting the correct CPU versions.
RUN pip install torch==2.1.2 torchaudio==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file
COPY requirements.txt .

# Install other Python dependencies AFTER PyTorch, to ensure PyTorch version isn't overridden
# by a sub-dependency from requirements.txt.
RUN pip install -r requirements.txt

# Copy the rest of the application code into the image
COPY ./main.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
