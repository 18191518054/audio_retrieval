from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from .config import SUPPORTED_EXTENSIONS
from .db import Database
from .media import audio_metadata, duration_ms


Progress = Callable[[dict], None]


def scan_directory(db: Database, root: Path, ffprobe: str, progress: Progress | None = None,
                   media_type: str = "music") -> dict:
    if media_type not in {"music", "effect"}:
        raise ValueError("素材类型必须选择音乐或音效")
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"目录不存在：{root}")
    root_id = db.add_root(root, media_type)
    found: set[str] = set()
    counters = {"found": 0, "new": 0, "updated": 0, "unchanged": 0, "failed": 0}

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        normalized = str(path.resolve())
        found.add(normalized.casefold())
        counters["found"] += 1
        try:
            stat = path.stat()
            metadata = audio_metadata(path, ffprobe)
            metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
            with db.connect() as conn:
                old = conn.execute(
                    "SELECT id, size_bytes, modified_ns, media_type, metadata_json FROM asset WHERE path=?", (normalized,)
                ).fetchone()
                changed = old is None or old["size_bytes"] != stat.st_size or old["modified_ns"] != stat.st_mtime_ns
                type_changed = old is not None and old["media_type"] != media_type
                if old is None:
                    conn.execute(
                        """INSERT INTO asset
                        (path, root_id, filename, extension, size_bytes, modified_ns, duration_ms, metadata_json, media_type,
                         status, error, missing, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'discovered', NULL, 0, ?)""",
                        (normalized, root_id, path.name, path.suffix.lower(), stat.st_size,
                         stat.st_mtime_ns, duration_ms(path, ffprobe), metadata_json, media_type, time.time()),
                    )
                    counters["new"] += 1
                elif changed or type_changed:
                    conn.execute(
                        """UPDATE asset SET root_id=?, filename=?, extension=?, size_bytes=?,
                        modified_ns=?, duration_ms=?, metadata_json=?, media_type=?, status='discovered', error=NULL, missing=0,
                        content_sha256=NULL,hash_modified_ns=NULL,duplicate_of=NULL,
                        updated_at=? WHERE id=?""",
                        (root_id, path.name, path.suffix.lower(), stat.st_size, stat.st_mtime_ns,
                        duration_ms(path, ffprobe), metadata_json, media_type, time.time(), old["id"]),
                    )
                    conn.execute("DELETE FROM embedding WHERE asset_id=?", (old["id"],))
                    conn.execute("DELETE FROM waveform WHERE asset_id=?", (old["id"],))
                    conn.execute("DELETE FROM asset_tag WHERE asset_id=?", (old["id"],))
                    conn.execute("DELETE FROM audio_analysis WHERE asset_id=?", (old["id"],))
                    if type_changed:
                        conn.execute("DELETE FROM asset_tag_node WHERE asset_id=?", (old["id"],))
                    counters["updated"] += 1
                elif old["metadata_json"] != metadata_json:
                    conn.execute(
                        "UPDATE asset SET metadata_json=?,missing=0,updated_at=? WHERE id=?",
                        (metadata_json, time.time(), old["id"]),
                    )
                    conn.execute("DELETE FROM asset_text_embedding WHERE asset_id=?", (old["id"],))
                    counters["updated"] += 1
                else:
                    conn.execute("UPDATE asset SET missing=0 WHERE id=?", (old["id"],))
                    counters["unchanged"] += 1
        except OSError:
            counters["failed"] += 1
        if progress and counters["found"] % 10 == 0:
            progress(counters.copy())

    with db.connect() as conn:
        known = conn.execute("SELECT id, path FROM asset WHERE root_id=?", (root_id,)).fetchall()
        for row in known:
            if row["path"].casefold() not in found:
                conn.execute("UPDATE asset SET missing=1, status='missing' WHERE id=?", (row["id"],))
        conn.execute("UPDATE library_root SET last_scan_at=? WHERE id=?", (time.time(), root_id))
    if progress:
        progress(counters.copy())
    return counters
