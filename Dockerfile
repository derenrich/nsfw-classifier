# syntax=docker/dockerfile:1
# Use official PyTorch runtime image from GHCR (PyTorch 2.12.0, CUDA 12.6, cuDNN 9)
FROM ghcr.io/pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# PYTHONUNBUFFERED: Prevents Python from buffering stdout/stderr (crucial for docker logs)
# HF_HOME: Defines the cache directory for Hugging Face models
# PIP_BREAK_SYSTEM_PACKAGES: Allows pip installs in PEP 668 managed container environments
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache \
    PIP_BREAK_SYSTEM_PACKAGES=1

# Set working directory inside the container
WORKDIR /app

# Copy python dependencies list
COPY requirements.txt .

# Install dependencies
# We use --no-cache-dir to minimize final image size. 
# PyTorch is already pre-installed in the base image, so pip will skip downloading it.
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache Hugging Face model weights during build phase.
# We mount the HF_TOKEN secret securely so it is not leaked in the image metadata/history.
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) python -c "\
from transformers import pipeline; \
pipeline('image-classification', model='Falconsai/nsfw_image_detection_26'); \
pipeline('image-classification', model='Freepik/nsfw_image_detector'); \
pipeline('image-classification', model='derenrich/private_detector_hf', trust_remote_code=True)\
"

# Copy the application code
COPY main.py .

# Expose port 8000
EXPOSE 8000

# Start the application using the Python entrypoint (reads PORT from environment)
CMD ["python", "main.py"]
