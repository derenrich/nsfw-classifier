"""
SQLite-backed score cache for NSFW classification results.

Caches classification predictions keyed by (url, model) so repeat requests
for the same URL skip image download and inference entirely.  Only URL-based
requests are cached; local file paths are never cached.

The cache is designed to be resilient: if the database path is non-writable or
any other SQLite error occurs during initialization, the cache silently
disables itself (logging a warning) and all operations become no-ops.
"""

import json
import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("nsfw-classifier")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS score_cache (
    url        TEXT NOT NULL,
    model      TEXT NOT NULL,
    predictions TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (url, model)
);
"""


class ScoreCache:
    """Thread-safe SQLite score cache.

    Each thread gets its own ``sqlite3.Connection`` via ``threading.local()``
    which is safe under WAL journal mode.

    If the database cannot be opened or the schema cannot be created the
    instance enters a *disabled* state where :meth:`get` always returns
    ``None`` and :meth:`put` is a silent no-op.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._disabled = False

        # Counters for /health observability
        self.hits = 0
        self.misses = 0
        self._stats_lock = threading.Lock()

        # Eagerly validate that we can open + write the database.
        try:
            conn = self._get_connection()
            conn.execute("SELECT 1")
            logger.info(f"Score cache initialized at {db_path}")
        except Exception as exc:
            logger.warning(
                f"Score cache disabled — failed to open database at "
                f"'{db_path}': {exc}"
            )
            self._disabled = True

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return a per-thread connection, creating one if needed."""
        conn: Optional[sqlite3.Connection] = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=5)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()
            self._local.conn = conn
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, url: str, model: str) -> Optional[List[Dict[str, Any]]]:
        """Look up cached predictions.

        Returns the deserialized prediction list on a cache hit, or ``None``
        on a miss (or if the cache is disabled).
        """
        if self._disabled:
            return None

        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT predictions FROM score_cache WHERE url = ? AND model = ?",
                (url, model),
            ).fetchone()
        except Exception as exc:
            logger.warning(f"Score cache read error: {exc}")
            return None

        with self._stats_lock:
            if row is not None:
                self.hits += 1
            else:
                self.misses += 1

        if row is not None:
            try:
                return json.loads(row[0])
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning(f"Score cache JSON decode error: {exc}")
                return None

        return None

    def put(
        self,
        url: str,
        model: str,
        predictions: List[Dict[str, Any]],
    ) -> None:
        """Insert or replace cached predictions for *(url, model)*."""
        if self._disabled:
            return

        try:
            conn = self._get_connection()
            conn.execute(
                "INSERT OR REPLACE INTO score_cache (url, model, predictions, created_at) "
                "VALUES (?, ?, ?, ?)",
                (url, model, json.dumps(predictions), time.time()),
            )
            conn.commit()
        except Exception as exc:
            logger.warning(f"Score cache write error: {exc}")
