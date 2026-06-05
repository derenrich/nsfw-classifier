# Use official PyTorch runtime image with CUDA 12.1 and cuDNN 8
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# PYTHONUNBUFFERED: Prevents Python from buffering stdout/stderr (crucial for docker logs)
# HF_HOME: Defines the cache directory for Hugging Face models
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/hf_cache

# Set working directory inside the container
WORKDIR /app

# Copy python dependencies list
COPY requirements.txt .

# Install dependencies
# We use --no-cache-dir to minimize final image size. 
# PyTorch is already pre-installed in the base image, so pip will skip downloading it.
RUN pip install --no-cache-dir -r requirements.txt

# Add build argument for Hugging Face authentication (required for gated/restricted models)
ARG HF_TOKEN

# Pre-download and cache Hugging Face model weights during build phase.
# We pass the HF_TOKEN to this step so we can access restricted models.
# By setting it only for this RUN step, we avoid baking the token into the image's runtime env.
RUN HF_TOKEN=$HF_TOKEN python -c "from transformers import pipeline; pipeline('image-classification', model='Falconsai/nsfw_image_detection_26')"

# Copy the application code
COPY main.py .

# Expose port 8000
EXPOSE 8000

# Start the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
