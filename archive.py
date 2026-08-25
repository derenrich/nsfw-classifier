"""
Image archive for downloading and storing Wikipedia article images with
attribution metadata.

Stores images using Wikimedia's MD5 hash-based directory structure so files
are distributed across ~4096 directories.  Attribution data from the
Wikipedia Attribution API is saved as gzipped JSON alongside each image.
Media-list API responses are persisted with timestamps, also using hash-based
directory bucketing.

Controlled by the ``IMAGE_ARCHIVE_PATH`` environment variable.  If the path
is non-writable the archive disables itself gracefully (logged, no crash).
"""

import gzip
import hashlib
import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError

logger = logging.getLogger("nsfw-classifier")

# Policy-compliant User-Agent for all Wikimedia API requests
_USER_AGENT = (
    "NSFWClassifierBot/1.0 "
    "(https://github.com/derenrich/nsfw-classifier; "
    "info@nsfw-classifier.local) Python-urllib/3"
)

# ---------------------------------------------------------------------------
# Rate limiter (reused from main.py pattern but kept self-contained)
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Sliding-window rate limiter that blocks until a slot opens."""

    def __init__(self, max_requests: int, period_seconds: float) -> None:
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.timestamps: List[float] = []
        self.lock = threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            while True:
                now = time.time()
                self.timestamps = [
                    t for t in self.timestamps
                    if now - t < self.period_seconds
                ]
                if len(self.timestamps) < self.max_requests:
                    self.timestamps.append(now)
                    return
                sleep_time = self.timestamps[0] + self.period_seconds - now
                if sleep_time > 0:
                    self.lock.release()
                    try:
                        time.sleep(sleep_time)
                    finally:
                        self.lock.acquire()


# Conservative: 10 requests per second for the attribution API
_attribution_limiter = _RateLimiter(max_requests=10, period_seconds=1.0)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _md5_hash_dirs(name: str) -> str:
    """Return the ``a/ab`` hash-bucket prefix for *name* (Wikimedia style).

    >>> _md5_hash_dirs("Google_2026_logo.svg")
    'e/ee'
    """
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"{h[0]}/{h[:2]}"


def _url_to_relative_path(url: str) -> str:
    """Convert a Wikimedia upload URL to a relative filesystem path.

    E.g. ``https://upload.wikimedia.org/wikipedia/commons/thumb/e/ee/Foo.svg/250px-Foo.svg.png``
    → ``wikipedia/commons/thumb/e/ee/Foo.svg/250px-Foo.svg.png``
    """
    parsed = urllib.parse.urlparse(url)
    # Strip leading slash
    path = parsed.path.lstrip("/")
    # URL-decode (e.g. %28 → '(')
    return urllib.parse.unquote(path)


def _safe_filename(title: str) -> str:
    """Sanitise a Wikipedia article title for use as a filename."""
    return title.replace("/", "_").replace("\\", "_").replace("\x00", "_")


def _fetch_with_retries(
    url: str,
    max_retries: int = 3,
    timeout: int = 15,
) -> Optional[bytes]:
    """GET *url* with exponential backoff on 429 / 5xx errors.

    Returns the response body bytes, or ``None`` on persistent failure.
    """
    for attempt in range(max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    f"HTTP {exc.code} from {url} — retrying in {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
            else:
                logger.error(f"HTTP {exc.code} from {url} — giving up")
                return None
        except (URLError, OSError) as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    f"Network error fetching {url}: {exc} — retrying in {wait}s"
                )
                time.sleep(wait)
            else:
                logger.error(f"Network error fetching {url}: {exc} — giving up")
                return None
    return None


# ---------------------------------------------------------------------------
# Attribution API
# ---------------------------------------------------------------------------

def fetch_attribution(file_title: str) -> Optional[Dict[str, Any]]:
    """Fetch attribution signals for a Wikimedia ``File:`` page.

    Parameters
    ----------
    file_title:
        The full ``File:…`` title (e.g. ``File:Clapton_is_God.jpg``).

    Returns the parsed JSON dict, or ``None`` on failure.
    """
    _attribution_limiter.acquire()

    encoded = urllib.parse.quote(file_title, safe="")
    url = (
        f"https://en.wikipedia.org/w/rest.php/attribution/"
        f"v0-beta/pages/{encoded}/signals"
    )
    body = _fetch_with_retries(url)
    if body is None:
        return None
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"Attribution JSON decode error for {file_title}: {exc}")
        return None


# ---------------------------------------------------------------------------
# ImageArchive
# ---------------------------------------------------------------------------

class ImageArchive:
    """On-disk archive of Wikipedia images with attribution metadata.

    Directory layout::

        {base_path}/
        ├── media-lists/{a}/{ab}/{Article_Title}.json
        └── images/wikipedia/commons/thumb/{a}/{ab}/…
    """

    def __init__(self, base_path: str) -> None:
        self.base_path = base_path
        self._disabled = False

        # Counters for /health
        self.archived = 0
        self.skipped = 0
        self.failed = 0
        self._stats_lock = threading.Lock()

        try:
            os.makedirs(os.path.join(base_path, "images"), exist_ok=True)
            os.makedirs(os.path.join(base_path, "media-lists"), exist_ok=True)
            # Quick write test
            test_file = os.path.join(base_path, ".write_test")
            with open(test_file, "w") as f:
                f.write("ok")
            os.remove(test_file)
            logger.info(f"Image archive initialized at {base_path}")
        except Exception as exc:
            logger.warning(
                f"Image archive disabled — cannot write to '{base_path}': {exc}"
            )
            self._disabled = True

    # ------------------------------------------------------------------
    # Media-list persistence
    # ------------------------------------------------------------------

    def _media_list_path(self, article_title: str) -> str:
        safe = _safe_filename(article_title)
        bucket = _md5_hash_dirs(safe)
        return os.path.join(
            self.base_path, "media-lists", bucket, f"{safe}.json"
        )

    def save_media_list(
        self,
        article_title: str,
        response: Dict[str, Any],
    ) -> None:
        """Persist a media-list API response with a fetch timestamp."""
        if self._disabled:
            return
        enriched = {
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **response,
        }
        path = self._media_list_path(article_title)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(enriched, f, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"Failed to save media-list for '{article_title}': {exc}")

    def get_media_list(self, article_title: str) -> Optional[Dict[str, Any]]:
        """Return a previously saved media-list, or ``None``."""
        path = self._media_list_path(article_title)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Image storage
    # ------------------------------------------------------------------

    def _image_path(self, url: str) -> str:
        rel = _url_to_relative_path(url)
        return os.path.join(self.base_path, "images", rel)

    def has_image(self, url: str) -> bool:
        """Check whether this image URL has already been archived."""
        return os.path.exists(self._image_path(url))

    def save_image(self, url: str, data: bytes) -> str:
        """Write raw image bytes; returns the absolute path written."""
        path = self._image_path(url)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    # ------------------------------------------------------------------
    # Attribution storage
    # ------------------------------------------------------------------

    def _attribution_path(self, url: str) -> str:
        img_path = self._image_path(url)
        return img_path + ".attribution.json.gz"

    def has_attribution(self, url: str) -> bool:
        return os.path.exists(self._attribution_path(url))

    def save_attribution(
        self, url: str, attribution: Dict[str, Any]
    ) -> None:
        """Write gzipped attribution JSON alongside the image."""
        path = self._attribution_path(url)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as f:
            json.dump(attribution, f, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Stats helpers
    # ------------------------------------------------------------------

    def record_archived(self) -> None:
        with self._stats_lock:
            self.archived += 1

    def record_skipped(self) -> None:
        with self._stats_lock:
            self.skipped += 1

    def record_failed(self) -> None:
        with self._stats_lock:
            self.failed += 1
