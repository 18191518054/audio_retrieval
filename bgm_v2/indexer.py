from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

from .db import Database
from .embedder import Embedder


Progress = Callable[[dict], None]


def build_embeddings(
    db: Database, embedder: Embedder, limit: int | None = None, progress: Progress | None = None
) -> dict:
    params: list[object] = [embedder.provider, embedder.model_name]
    sql = """
        SELECT a.* FROM asset a
        LEFT JOIN embedding e ON e.asset_id=a.id AND e.provider=? AND e.model=?
        WHERE a.missing=0 AND a.status<>'duplicate'
        AND (e.asset_id IS NULL OR e.source_modified_ns<>a.modified_ns)
        ORDER BY a.id
    """
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    with db.connect() as conn:
        rows = [dict(row) for row in conn.execute(sql, params)]
    result = {"queued": len(rows), "completed": 0, "failed": 0, "current": ""}

    for row in rows:
        result["current"] = row["filename"]
        try:
            vector = np.asarray(embedder.embed_audio(Path(row["path"])), dtype="<f4")
            with db.connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO embedding
                    (asset_id, provider, model, dimension, vector, source_modified_ns, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (row["id"], embedder.provider, embedder.model_name, vector.size,
                     vector.tobytes(), row["modified_ns"], time.time()),
                )
                conn.execute("UPDATE asset SET status='ready', error=NULL WHERE id=?", (row["id"],))
            result["completed"] += 1
        except Exception as exc:  # keep the remaining library usable
            with db.connect() as conn:
                conn.execute(
                    "UPDATE asset SET status='failed', error=? WHERE id=?",
                    (str(exc)[:1000], row["id"]),
                )
            result["failed"] += 1
        if progress:
            progress(result.copy())
    result["current"] = ""
    if progress:
        progress(result.copy())
    return result
