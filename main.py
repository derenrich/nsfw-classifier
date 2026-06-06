import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nsfw-classifier")

app = FastAPI(
    title="NSFW Image Classification Server",
    description="A high-performance GPU-optimized server for batch NSFW image classification",
    version="1.0.0"
)

# Configuration for host to container path translation
# E.g. HOST_DATASET_PATH="/home/user/my_datasets" maps to CONTAINER_DATASET_PATH="/datasets"
HOST_DATASET_PATH = os.getenv("HOST_DATASET_PATH", "/home/user/my_datasets")
CONTAINER_DATASET_PATH = os.getenv("CONTAINER_DATASET_PATH", "/datasets")

logger.info(f"Path translation configured: Host prefix '{HOST_DATASET_PATH}' -> Container prefix '{CONTAINER_DATASET_PATH}'")

# Device configuration (0 = CUDA GPU, -1 = CPU)
device = 0 if torch.cuda.is_available() else -1
logger.info(f"CUDA availability: {torch.cuda.is_available()}")
if device == 0:
    logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA GPU not available. Running on CPU.")

# Initialize Hugging Face pipeline
# Falconsai/nsfw_image_detection_26 is a ViT-based image classification model
MODEL_NAME = "Falconsai/nsfw_image_detection_26"
logger.info(f"Loading pipeline for model: {MODEL_NAME}...")
try:
    classifier = pipeline("image-classification", model=MODEL_NAME, device=device)
    logger.info("Pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Error loading pipeline: {e}")
    raise e

class ClassificationResult(BaseModel):
    file_path: str
    predictions: List[Dict[str, Any]] = []
    error: str = None

def translate_path(host_path: str) -> str:
    """
    Translates an absolute file path on the host system to its corresponding
    path inside the Docker container based on volume mapping configuration.
    """
    # Clean up paths to ensure matching works properly
    host_path_clean = os.path.normpath(host_path)
    host_prefix_clean = os.path.normpath(HOST_DATASET_PATH)
    
    if host_path_clean.startswith(host_prefix_clean):
        relative_path = os.path.relpath(host_path_clean, host_prefix_clean)
        # Avoid joining with '..' if path somehow escaped prefix
        if not relative_path.startswith(".."):
            container_path = os.path.join(CONTAINER_DATASET_PATH, relative_path)
            return os.path.normpath(container_path)
            
    # Fallback to the original path if it doesn't match the expected prefix
    return host_path_clean

@app.post("/classify-batch", response_model=List[ClassificationResult])
async def classify_batch(file_paths: List[str]):
    """
    Accepts a list of absolute file paths on the host system,
    translates them to container paths, loads valid images,
    runs batched GPU inference, and returns predictions.
    """
    results = [ClassificationResult(file_path=path) for path in file_paths]
    
    # Store images and map them back to their original indexes in the request list
    valid_images = []
    valid_indices = []
    
    for idx, host_path in enumerate(file_paths):
        container_path = translate_path(host_path)
        
        if not os.path.exists(container_path):
            error_msg = f"File not found in container. Host path: {host_path} -> Translated container path: {container_path}"
            logger.warning(error_msg)
            results[idx].error = error_msg
            continue
            
        try:
            # Load and convert image to RGB (pipeline requires RGB)
            img = Image.open(container_path).convert("RGB")
            valid_images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            error_msg = f"Error loading image: {str(e)}"
            logger.error(f"{error_msg} (Path: {container_path})")
            results[idx].error = error_msg

    # If we have valid images, run batched inference
    if valid_images:
        try:
            # We use a batch size of up to 16 to optimize GPU throughput
            batch_size = min(len(valid_images), 16)
            logger.info(f"Running inference on a batch of {len(valid_images)} images with batch_size={batch_size}")
            
            # The Hugging Face pipeline handles batching automatically when passed a list of PIL Images
            predictions = classifier(valid_images, batch_size=batch_size)
            
            # If batch_size=1 and only 1 image, some pipeline versions return a dict instead of list of lists
            if len(valid_images) == 1 and not isinstance(predictions, list):
                predictions = [predictions]
                
            # Populate predictions back to the correct original indices
            for valid_idx, pred in zip(valid_indices, predictions):
                results[valid_idx].predictions = pred
        except Exception as e:
            error_msg = f"Inference failure: {str(e)}"
            logger.critical(error_msg)
            for valid_idx in valid_indices:
                results[valid_idx].error = error_msg

    return results

@app.get("/health")
async def health():
    """
    Health check endpoint returning system status and GPU details.
    """
    gpu_available = torch.cuda.is_available()
    return {
        "status": "healthy",
        "gpu_available": gpu_available,
        "gpu_name": torch.cuda.get_device_name(0) if gpu_available else None,
        "device_allocated": "cuda:0" if device == 0 else "cpu",
        "host_dataset_path": HOST_DATASET_PATH,
        "container_dataset_path": CONTAINER_DATASET_PATH
    }

@app.get("/benchmark")
async def run_benchmark():
    """
    Downloads a random sample image from Picsum, and benchmarks inference 
    speeds across different batch sizes (1, 2, 4, 8, 16, 32).
    """
    import io
    import time
    import urllib.request
    
    url = "https://picsum.photos/200/300"
    logger.info(f"Downloading benchmark image from {url}...")
    
    try:
        # Download image using standard library urllib
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            img_data = response.read()
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to download benchmark image: {e}")
        return {"error": f"Failed to download benchmark image: {str(e)}"}

    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = []
    
    # Warmup run to compile graph / initialize CUDA context
    try:
        classifier([img], batch_size=1)
    except Exception as e:
        logger.error(f"Warmup inference failed: {e}")
        return {"error": f"Warmup inference failed: {str(e)}"}

    logger.info("Starting GPU batch size benchmark runs...")
    for size in batch_sizes:
        # Replicate image to build the batch
        batch = [img] * size
        
        start_time = time.perf_counter()
        try:
            # Run inference
            classifier(batch, batch_size=size)
            end_time = time.perf_counter()
            
            elapsed = end_time - start_time
            time_per_image = elapsed / size
            images_per_sec = size / elapsed
            
            results.append({
                "batch_size": size,
                "total_time_seconds": round(elapsed, 4),
                "time_per_image_seconds": round(time_per_image, 4),
                "images_per_second": round(images_per_sec, 2)
            })
            logger.info(f"Batch size {size} benchmark completed in {elapsed:.4f}s ({images_per_sec:.2f} img/sec)")
        except Exception as e:
            logger.error(f"Benchmark run failed for batch size {size}: {e}")
            results.append({
                "batch_size": size,
                "error": f"Inference failed: {str(e)}"
            })

    return {
        "device": "cuda:0" if device == 0 else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "image_size": f"{img.width}x{img.height}",
        "benchmark_results": results
    }

if __name__ == "__main__":
    import uvicorn
    # Read the PORT environment variable, defaulting to 8000
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
