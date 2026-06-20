import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
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

# Initialize Hugging Face pipelines
# Model 1: Falconsai/nsfw_image_detection_26
MODEL_FALCONSAI = "Falconsai/nsfw_image_detection_26"
# Model 2: Freepik/nsfw_image_detector
MODEL_FREEPIK = "Freepik/nsfw_image_detector"
# Model 3: Ported Private Detector model
MODEL_PRIVATE_DETECTOR = "derenrich/private_detector_hf"

logger.info(f"Loading pipeline for model: {MODEL_FALCONSAI}...")
try:
    classifier_falconsai = pipeline("image-classification", model=MODEL_FALCONSAI, device=device)
    logger.info("Falconsai pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Falconsai pipeline: {e}")
    raise e

logger.info(f"Loading pipeline for model: {MODEL_FREEPIK}...")
try:
    classifier_freepik = pipeline("image-classification", model=MODEL_FREEPIK, device=device)
    logger.info("Freepik pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Freepik pipeline: {e}")
    raise e

logger.info(f"Loading pipeline for model: {MODEL_PRIVATE_DETECTOR}...")
try:
    classifier_private_detector = pipeline(
        "image-classification",
        model=MODEL_PRIVATE_DETECTOR,
        device=device,
        trust_remote_code=True
    )
    logger.info("Private Detector pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Private Detector pipeline: {e}")
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

def translate_path_back(container_path: str) -> str:
    """
    Translates an absolute file path inside the container back to its corresponding
    path on the host system based on volume mapping configuration.
    """
    container_path_clean = os.path.normpath(container_path)
    container_prefix_clean = os.path.normpath(CONTAINER_DATASET_PATH)
    
    if container_path_clean.startswith(container_prefix_clean):
        relative_path = os.path.relpath(container_path_clean, container_prefix_clean)
        if not relative_path.startswith(".."):
            host_path = os.path.join(HOST_DATASET_PATH, relative_path)
            return os.path.normpath(host_path)
            
    return container_path_clean

def expand_paths(paths: List[str]) -> List[str]:
    """
    Expands any local paths containing wildcards (globbing) inside the container
    and translates them back to host paths.
    """
    import glob
    expanded = []
    
    for path in paths:
        if path.startswith(("http://", "https://")):
            expanded.append(path)
            continue
            
        # Check if the path contains wildcard characters
        if any(char in path for char in ("*", "?", "[")):
            container_pattern = translate_path(path)
            matches = glob.glob(container_pattern, recursive=True)
            if matches:
                matches.sort()
                for match in matches:
                    # Only include files (ignore directories found by globbing)
                    if os.path.isfile(match):
                        expanded.append(translate_path_back(match))
            else:
                # If no matches found, keep the original path so it generates a failure in the response
                expanded.append(path)
        else:
            # No wildcards, keep the original path
            expanded.append(path)
            
    return expanded

import time
import threading

class RateLimiter:
    """
    A thread-safe sliding window rate limiter that blocks (waiting/retrying)
    until a request slot is available for external resources.
    """
    def __init__(self, max_requests: int, period_seconds: float):
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.timestamps = []
        self.lock = threading.Lock()

    def acquire(self) -> None:
        """
        Acquires a request slot, blocking/sleeping if the rate limit is exceeded
        until a slot opens up.
        """
        with self.lock:
            while True:
                now = time.time()
                self.timestamps = [t for t in self.timestamps if now - t < self.period_seconds]
                
                if len(self.timestamps) < self.max_requests:
                    self.timestamps.append(now)
                    return
                
                sleep_time = self.timestamps[0] + self.period_seconds - now
                if sleep_time > 0:
                    logger.info(f"Rate limit for external images reached. Waiting {sleep_time:.2f}s to retry...")
                    self.lock.release()
                    try:
                        time.sleep(sleep_time)
                    finally:
                        self.lock.acquire()

# Rate limit external image downloads to 20 requests per 30 seconds
external_image_limiter = RateLimiter(max_requests=20, period_seconds=40.0)

def load_image(path_or_url: str) -> Image.Image:
    """
    Loads an image from a local host path (translated to container path) or fetches it from a remote URL.
    """
    import io
    import urllib.request

    if path_or_url.startswith(("http://", "https://")):
        # Blocks and waits if rate limit is reached
        external_image_limiter.acquire()
        
        logger.info(f"Fetching remote image from URL: {path_or_url}")
        req = urllib.request.Request(
            path_or_url,
            headers={'User-Agent': 'NSFWClassifierBot/1.0 (https://github.com/derenrich/nsfw-classifier; info@nsfw-classifier.local)'}
        )
        with urllib.request.urlopen(req, timeout=15) as response:
            img_data = response.read()
        return Image.open(io.BytesIO(img_data)).convert("RGB")
    else:
        container_path = translate_path(path_or_url)
        if not os.path.exists(container_path):
            raise FileNotFoundError(f"File not found on container system (mapped from: {path_or_url})")
        return Image.open(container_path).convert("RGB")

async def classify_batch_generic(path_or_urls: List[str], classifier_pipeline) -> List[ClassificationResult]:
    """
    Generic runner for loading images/URLs and executing batched pipeline classification.
    """
    # Expand any glob patterns in the path list
    expanded_paths = expand_paths(path_or_urls)
    
    results = [ClassificationResult(file_path=path) for path in expanded_paths]
    
    valid_images = []
    valid_indices = []
    
    import asyncio
    loop = asyncio.get_running_loop()
    
    for idx, path_or_url in enumerate(expanded_paths):
        try:
            # Run blocking load_image in a separate thread so it doesn't freeze the main event loop
            img = await loop.run_in_executor(None, load_image, path_or_url)
            valid_images.append(img)
            valid_indices.append(idx)
        except Exception as e:
            error_msg = f"Failed to load image: {str(e)}"
            logger.error(f"{error_msg} (Input: {path_or_url})")
            results[idx].error = error_msg

    if valid_images:
        try:
            # We use a batch size of up to 16 to optimize GPU throughput
            batch_size = min(len(valid_images), 16)
            logger.info(f"Running inference on a batch of {len(valid_images)} images with batch_size={batch_size}")
            
            # The Hugging Face pipeline handles batching automatically when passed a list of PIL Images
            predictions = classifier_pipeline(valid_images, batch_size=batch_size)
            
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

@app.post("/classify-batch", response_model=List[ClassificationResult])
async def classify_batch_default(file_paths: List[str]):
    """
    Accepts a list of absolute host file paths or URLs, runs batched inference
    using the default model (Falconsai/nsfw_image_detection_26).
    """
    return await classify_batch_generic(file_paths, classifier_falconsai)

@app.post("/classify-batch/falconsai", response_model=List[ClassificationResult])
async def classify_batch_falconsai(file_paths: List[str]):
    """
    Accepts a list of absolute host file paths or URLs, runs batched inference
    using Falconsai/nsfw_image_detection_26.
    """
    return await classify_batch_generic(file_paths, classifier_falconsai)

@app.post("/classify-batch/freepik", response_model=List[ClassificationResult])
async def classify_batch_freepik(file_paths: List[str]):
    """
    Accepts a list of absolute host file paths or URLs, runs batched inference
    using Freepik/nsfw_image_detector.
    """
    return await classify_batch_generic(file_paths, classifier_freepik)

@app.post("/classify-batch/private-detector", response_model=List[ClassificationResult])
async def classify_batch_private_detector(file_paths: List[str]):
    """
    Accepts a list of absolute host file paths or URLs, runs batched inference
    using the ported Private Detector model.
    """
    return await classify_batch_generic(file_paths, classifier_private_detector)

def fetch_category_images(category_name: str, limit: int, thumb_width: int = 400) -> List[Dict[str, str]]:
    """
    Queries the Wikimedia Commons API to get file titles and thumbnail URLs.
    Uses a policy-compliant User-Agent.
    """
    import json
    import urllib.parse
    import urllib.request

    # Standardize category name structure
    if not category_name.startswith("Category:"):
        category_name = f"Category:{category_name}"

    params = {
        "action": "query",
        "generator": "categorymembers",
        "gcmtitle": category_name,
        "gcmtype": "file",
        "gcmlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url",
        "iiurlwidth": str(thumb_width),
        "format": "json"
    }
    
    query_string = urllib.parse.urlencode(params)
    url = f"https://commons.wikimedia.org/w/api.php?{query_string}"
    
    logger.info(f"Querying Wikimedia Commons Category API: {url}")
    
    req = urllib.request.Request(
        url,
        headers={
            # Policy-compliant User-Agent including contact details and github URL:
            # https://meta.wikimedia.org/wiki/User-Agent_policy
            'User-Agent': 'NSFWClassifierBot/1.0 (https://github.com/derenrich/nsfw-classifier; info@nsfw-classifier.local) Python-urllib/3'
        }
    )
    
    with urllib.request.urlopen(req, timeout=15) as response:
        data = json.loads(response.read().decode("utf-8"))
        
    pages = data.get("query", {}).get("pages", {})
    results = []
    
    for page_id, page_info in pages.items():
        title = page_info.get("title", "")
        imageinfo = page_info.get("imageinfo", [])
        if imageinfo:
            # Optimally fetch thumbnail URL
            thumb_url = imageinfo[0].get("thumburl")
            original_url = imageinfo[0].get("url")
            url_to_use = thumb_url if thumb_url else original_url
            if title and url_to_use:
                results.append({
                    "title": title,
                    "url": url_to_use
                })
                    
    # API query results can sometimes exceed our limit if generator returns slightly more items
    return results[:limit]

@app.get("/classify-category", response_class=PlainTextResponse)
async def classify_category(category: str, limit: int = 10, model: str = "all"):
    """
    Fetches images from a specified Wikimedia Commons category, runs batch inference
    on Falconsai, Freepik, or both models, and returns the results formatted
    as a sortable MediaWiki wikitext table.
    """
    model_lower = model.lower()
    if model_lower not in ("all", "falconsai", "freepik", "private-detector"):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model '{model}'. Supported options: 'all', 'falconsai', 'freepik', 'private-detector'."
        )

    try:
        # Fetch metadata and thumbnail URLs from Wikimedia Commons
        category_files = fetch_category_images(category, limit)
    except Exception as e:
        logger.error(f"Wikimedia API query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch category images from Wikimedia: {str(e)}")

    if not category_files:
        return f"No files found in category: {category}"

    # Extract URLs to pass to batch classification
    urls = [file_info["url"] for file_info in category_files]

    # Run classification for models based on selection
    falconsai_results = None
    freepik_results = None
    private_detector_results = None

    if model_lower in ("all", "falconsai"):
        logger.info("Running Falconsai model classification...")
        falconsai_results = await classify_batch_generic(urls, classifier_falconsai)

    if model_lower in ("all", "freepik"):
        logger.info("Running Freepik model classification...")
        freepik_results = await classify_batch_generic(urls, classifier_freepik)

    if model_lower in ("all", "private-detector"):
        logger.info("Running Private Detector model classification...")
        private_detector_results = await classify_batch_generic(urls, classifier_private_detector)

    # Helper function to format predictions into a MediaWiki table cell markup
    def get_cell_markup(result) -> str:
        if not result or result.error:
            err_msg = f"Error: {result.error}" if result else "N/A"
            return f'style="background-color: #f6f8fa; color: #6a737d;" data-sort-value="-1.00000" | {err_msg}'
        if not result.predictions:
            return 'style="background-color: #f6f8fa; color: #6a737d;" data-sort-value="-1.00000" | No predictions'
            
        predictions = result.predictions
        if hasattr(predictions, "dict"):
            predictions = predictions.dict()
            
        top_pred = max(predictions, key=lambda x: x.get('score', 0.0) if isinstance(x, dict) else getattr(x, 'score', 0.0))
        label = top_pred.get('label', 'unknown') if isinstance(top_pred, dict) else getattr(top_pred, 'label', 'unknown')
        score = top_pred.get('score', 0.0) if isinstance(top_pred, dict) else getattr(top_pred, 'score', 0.0)
        display_text = f"{label} ({score * 100:.1f}%)"
        
        # Calculate NSFW score for sorting (higher score = more NSFW)
        nsfw_score = None
        for p in predictions:
            p_label = p.get('label', '') if isinstance(p, dict) else getattr(p, 'label', '')
            p_score = p.get('score', 0.0) if isinstance(p, dict) else getattr(p, 'score', 0.0)
            if "nsfw" in p_label.lower():
                nsfw_score = p_score
                break
        if nsfw_score is None:
            if label.lower() in ('normal', 'sfw', 'safe', 's'):
                nsfw_score = 1.0 - score
            else:
                nsfw_score = score
                
        is_nsfw = label.lower() == 'nsfw' or ('nsfw' in label.lower() and label.lower() != 'sfw')
        
        if is_nsfw:
            style = 'style="background-color: #ffeef0; color: #d73a49; font-weight: bold;"'
        else:
            style = 'style="background-color: #e6ffed; color: #22863a;"'
            
        return f'{style} data-sort-value="{nsfw_score:.5f}" | {display_text}'

    # Construct sortable wikitext table
    wikitext = []
    wikitext.append('{| class="wikitable sortable"')
    wikitext.append(f'|+ NSFW Classification Comparison for {category}')
    wikitext.append('|-')
    
    # Header row (Image column is unsortable to make the table clean)
    if model_lower == "all":
        wikitext.append('! class="unsortable" | Image !! Falconsai Prediction !! Freepik Prediction !! Private Detector Prediction')
    elif model_lower == "falconsai":
        wikitext.append('! class="unsortable" | Image !! Falconsai Prediction')
    elif model_lower == "freepik":
        wikitext.append('! class="unsortable" | Image !! Freepik Prediction')
    elif model_lower == "private-detector":
        wikitext.append('! class="unsortable" | Image !! Private Detector Prediction')

    # Data rows
    for i, file_info in enumerate(category_files):
        wikitext.append('|-')
        # Cell 1: MediaWiki image markup using file title
        wikitext.append(f'| [[{file_info["title"]}|100px]]')
        
        if falconsai_results is not None:
            wikitext.append(f'| {get_cell_markup(falconsai_results[i])}')
            
        if freepik_results is not None:
            wikitext.append(f'| {get_cell_markup(freepik_results[i])}')

        if private_detector_results is not None:
            wikitext.append(f'| {get_cell_markup(private_detector_results[i])}')

    wikitext.append('|}')
    
    return '\n'.join(wikitext)

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
async def run_benchmark(width: int = 200, height: int = 300, model: str = "falconsai"):
    """
    Downloads a random sample image from Picsum with specified dimensions,
    and benchmarks inference speeds across different batch sizes (up to 1024)
    using the specified model ('falconsai' or 'freepik').
    """
    import io
    import time
    import urllib.request
    
    # Select target classifier pipeline
    model_lower = model.lower()
    if model_lower == "falconsai":
        classifier_pipeline = classifier_falconsai
        model_name = MODEL_FALCONSAI
    elif model_lower == "freepik":
        classifier_pipeline = classifier_freepik
        model_name = MODEL_FREEPIK
    elif model_lower == "private-detector":
        classifier_pipeline = classifier_private_detector
        model_name = MODEL_PRIVATE_DETECTOR
    else:
        return {"error": f"Invalid model '{model}'. Supported options: 'falconsai', 'freepik', 'private-detector'."}

    url = f"https://picsum.photos/{width}/{height}"
    logger.info(f"Downloading benchmark image ({width}x{height}) for model {model_name} from {url}...")
    
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        img = await loop.run_in_executor(None, load_image, url)
    except Exception as e:
        logger.error(f"Failed to download benchmark image: {e}")
        return {"error": f"Failed to download benchmark image: {str(e)}"}

    # Scan batch sizes up to 1024
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    results = []
    
    # Warmup run to compile graph / initialize CUDA context
    try:
        classifier_pipeline([img], batch_size=1)
    except Exception as e:
        logger.error(f"Warmup inference failed: {e}")
        return {"error": f"Warmup inference failed: {str(e)}"}

    logger.info(f"Starting GPU batch size benchmark runs for {model_name}...")
    for size in batch_sizes:
        # Replicate image to build the batch
        batch = [img] * size
        
        start_time = time.perf_counter()
        try:
            # Run inference
            classifier_pipeline(batch, batch_size=size)
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
            # Clean up GPU memory immediately if we hit an Out-Of-Memory (OOM) or other runtime error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return {
        "device": "cuda:0" if device == 0 else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "model_benchmarked": model_name,
        "image_size": f"{img.width}x{img.height}",
        "benchmark_results": results
    }

if __name__ == "__main__":
    import uvicorn
    # Read the PORT environment variable, defaulting to 8000
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on port {port}...")
    uvicorn.run("main:app", host="0.0.0.0", port=port)
