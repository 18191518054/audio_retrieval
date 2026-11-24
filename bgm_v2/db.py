from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
CREATE TABLE IF NOT EXISTS library_root (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    added_at REAL NOT NULL,
    last_scan_at REAL,
    media_type TEXT NOT NULL DEFAULT 'unknown'
);
CREATE TABLE IF NOT EXISTS asset (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    root_id INTEGER NOT NULL REFERENCES library_root(id),
    filename TEXT NOT NULL,
    extension TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL,
    duration_ms INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    media_type TEXT NOT NULL DEFAULT 'unknown',
    status TEXT NOT NULL DEFAULT 'discovered',
    error TEXT,
    missing INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL
    ,content_sha256 TEXT
    ,hash_modified_ns INTEGER
    ,duplicate_of INTEGER
);
CREATE INDEX IF NOT EXISTS ix_asset_root ON asset(root_id);
CREATE INDEX IF NOT EXISTS ix_asset_missing ON asset(missing);
CREATE TABLE IF NOT EXISTS embedding (
    asset_id INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector BLOB NOT NULL,
    source_modified_ns INTEGER NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(asset_id, provider, model)
);
CREATE TABLE IF NOT EXISTS waveform (
    asset_id INTEGER PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
    source_modified_ns INTEGER NOT NULL,
    points INTEGER NOT NULL,
    peaks BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS asset_tag (
    asset_id INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    namespace TEXT NOT NULL,
    code TEXT NOT NULL,
    label TEXT NOT NULL,
    score REAL NOT NULL,
    rank_no INTEGER NOT NULL,
    PRIMARY KEY(asset_id, namespace, code)
);
CREATE INDEX IF NOT EXISTS ix_asset_tag_filter ON asset_tag(namespace, code, rank_no);
CREATE TABLE IF NOT EXISTS audio_analysis (
    asset_id INTEGER PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
    source_modified_ns INTEGER NOT NULL,
    audio_type TEXT NOT NULL,
    audio_type_label TEXT NOT NULL,
    type_score REAL NOT NULL,
    bpm REAL,
    bpm_confidence REAL,
    model TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS setting (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS effect_action_analysis (
    asset_id INTEGER PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
    source_modified_ns INTEGER NOT NULL,
    model TEXT NOT NULL,
    actions_json TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS effect_caption_analysis (
    asset_id INTEGER PRIMARY KEY REFERENCES asset(id) ON DELETE CASCADE,
    source_modified_ns INTEGER NOT NULL,
    model TEXT NOT NULL,
    caption TEXT NOT NULL,
    labels_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    error TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tag_scheme (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    editable INTEGER NOT NULL DEFAULT 0,
    source_version TEXT
);
CREATE TABLE IF NOT EXISTS tag_node (
    id INTEGER PRIMARY KEY,
    scheme_id INTEGER NOT NULL REFERENCES tag_scheme(id) ON DELETE CASCADE,
    parent_id INTEGER REFERENCES tag_node(id) ON DELETE CASCADE,
    code TEXT,
    label TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    examples_json TEXT NOT NULL DEFAULT '[]',
    sort_order INTEGER NOT NULL DEFAULT 0,
    UNIQUE(scheme_id, parent_id, label)
);
CREATE INDEX IF NOT EXISTS ix_tag_node_parent ON tag_node(scheme_id, parent_id, sort_order);
CREATE TABLE IF NOT EXISTS tag_embedding (
    node_id INTEGER NOT NULL REFERENCES tag_node(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(node_id, provider, model)
);
CREATE INDEX IF NOT EXISTS ix_tag_embedding_model ON tag_embedding(provider, model);
CREATE TABLE IF NOT EXISTS material_context_embedding (
    node_id INTEGER NOT NULL REFERENCES tag_node(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    action_hash TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(node_id, provider, model, action_hash)
);
CREATE INDEX IF NOT EXISTS ix_material_context_model
ON material_context_embedding(provider, model, action_hash);
CREATE TABLE IF NOT EXISTS asset_text_embedding (
    asset_id INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    dimension INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(asset_id, model)
);
CREATE INDEX IF NOT EXISTS ix_asset_text_embedding_model
ON asset_text_embedding(model, asset_id);
CREATE TABLE IF NOT EXISTS asset_tag_node (
    asset_id INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    node_id INTEGER NOT NULL REFERENCES tag_node(id) ON DELETE CASCADE,
    assigned_by TEXT NOT NULL DEFAULT 'user',
    confidence REAL,
    evidence TEXT,
    created_at REAL NOT NULL,
    PRIMARY KEY(asset_id, node_id)
);
CREATE INDEX IF NOT EXISTS ix_asset_tag_node_node ON asset_tag_node(node_id, asset_id);
CREATE TABLE IF NOT EXISTS cloud_tag_cache (
    asset_id INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    scheme_code TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    source_modified_ns INTEGER NOT NULL,
    taxonomy_hash TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY(asset_id, scheme_code, provider, model)
);
CREATE TABLE IF NOT EXISTS asset_tag_analysis_state (
    asset_id INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
    scheme_code TEXT NOT NULL,
    engine TEXT NOT NULL,
    result_count INTEGER NOT NULL DEFAULT 0,
    completed_at REAL NOT NULL,
    PRIMARY KEY(asset_id, scheme_code)
);
"""


class Database:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            root_columns = {row[1] for row in conn.execute("PRAGMA table_info(library_root)")}
            if "media_type" not in root_columns:
                conn.execute("ALTER TABLE library_root ADD COLUMN media_type TEXT NOT NULL DEFAULT 'unknown'")
            asset_columns = {row[1] for row in conn.execute("PRAGMA table_info(asset)")}
            if "media_type" not in asset_columns:
                conn.execute("ALTER TABLE asset ADD COLUMN media_type TEXT NOT NULL DEFAULT 'unknown'")
            if "metadata_json" not in asset_columns:
                conn.execute("ALTER TABLE asset ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'")
            if "content_sha256" not in asset_columns:
                conn.execute("ALTER TABLE asset ADD COLUMN content_sha256 TEXT")
            if "hash_modified_ns" not in asset_columns:
                conn.execute("ALTER TABLE asset ADD COLUMN hash_modified_ns INTEGER")
            if "duplicate_of" not in asset_columns:
                conn.execute("ALTER TABLE asset ADD COLUMN duplicate_of INTEGER")
            tag_node_columns = {row[1] for row in conn.execute("PRAGMA table_info(tag_node)")}
            if "description" not in tag_node_columns:
                conn.execute("ALTER TABLE tag_node ADD COLUMN description TEXT NOT NULL DEFAULT ''")
            if "examples_json" not in tag_node_columns:
                conn.execute("ALTER TABLE tag_node ADD COLUMN examples_json TEXT NOT NULL DEFAULT '[]'")
            assignment_columns = {row[1] for row in conn.execute("PRAGMA table_info(asset_tag_node)")}
            if "confidence" not in assignment_columns:
                conn.execute("ALTER TABLE asset_tag_node ADD COLUMN confidence REAL")
            if "evidence" not in assignment_columns:
                conn.execute("ALTER TABLE asset_tag_node ADD COLUMN evidence TEXT")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(asset_tag)")}
            if "source" in columns or "reliable" in columns:
                conn.executescript("""
                    CREATE TABLE asset_tag_v2 (
                        asset_id INTEGER NOT NULL REFERENCES asset(id) ON DELETE CASCADE,
                        namespace TEXT NOT NULL, code TEXT NOT NULL, label TEXT NOT NULL,
                        score REAL NOT NULL, rank_no INTEGER NOT NULL,
                        PRIMARY KEY(asset_id, namespace, code));
                    INSERT INTO asset_tag_v2
                        SELECT asset_id, namespace, code, label, score, rank_no FROM asset_tag;
                    DROP TABLE asset_tag;
                    ALTER TABLE asset_tag_v2 RENAME TO asset_tag;
                    CREATE INDEX ix_asset_tag_filter ON asset_tag(namespace, code, rank_no);
                """)
            pipeline = conn.execute(
                "SELECT value FROM setting WHERE key='tag_pipeline_version'"
            ).fetchone()
            if pipeline is None or pipeline[0] != "audioset-v1":
                # Old CLAP prompt tags were relative guesses and must not masquerade
                # as results from the conservative local classifier.
                conn.execute("DELETE FROM asset_tag")
                conn.execute(
                    "INSERT OR REPLACE INTO setting(key, value) VALUES ('tag_pipeline_version', 'audioset-v1')"
                )
            # Migrate previously generated local/cloud results into a durable
            # completion marker. Empty cloud results are included so an
            # intentional model abstention is not recalculated on every build.
            conn.execute("""INSERT OR IGNORE INTO asset_tag_analysis_state
                (asset_id,scheme_code,engine,result_count,completed_at)
                SELECT atn.asset_id,s.code,'existing',COUNT(*),MIN(atn.created_at)
                FROM asset_tag_node atn JOIN tag_node n ON n.id=atn.node_id
                JOIN tag_scheme s ON s.id=n.scheme_id
                WHERE atn.assigned_by<>'user'
                GROUP BY atn.asset_id,s.code""")
            conn.execute("""INSERT OR IGNORE INTO asset_tag_analysis_state
                (asset_id,scheme_code,engine,result_count,completed_at)
                SELECT asset_id,scheme_code,provider,0,created_at FROM cloud_tag_cache""")
            # Older local runs did not persist an explicit marker for empty
            # results. A fully completed historical effect-tag job proves that
            # every effect existing at its finish time was attempted.
            previous_job = conn.execute(
                "SELECT value FROM setting WHERE key='background_job'"
            ).fetchone()
            if previous_job:
                try:
                    job = json.loads(previous_job[0])
                    stage = job.get("stages", {}).get("effect_tags", {})
                    finished_at = float(job.get("finished_at") or 0)
                    stage_complete = (
                        job.get("state") == "completed"
                        and int(stage.get("completed", 0)) >= int(stage.get("queued", 0))
                    )
                except (TypeError, ValueError, json.JSONDecodeError):
                    stage_complete, finished_at = False, 0
                if stage_complete and finished_at:
                    for scheme_code in ("ucs", "recommended", "instrument_material"):
                        conn.execute("""INSERT OR IGNORE INTO asset_tag_analysis_state
                            (asset_id,scheme_code,engine,result_count,completed_at)
                            SELECT id,?,'historical-completed-job',0,?
                            FROM asset WHERE missing=0 AND media_type='effect'
                            AND status<>'duplicate' AND updated_at<=?""",
                            (scheme_code, finished_at, finished_at),
                        )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add_root(self, path: Path, media_type: str = "unknown") -> int:
        normalized = str(path.resolve())
        with self._lock, self.connect() as conn:
            conn.execute(
                """INSERT INTO library_root(path, added_at, media_type) VALUES (?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET media_type=excluded.media_type""",
                (normalized, time.time(), media_type),
            )
            row = conn.execute("SELECT id FROM library_root WHERE path=?", (normalized,)).fetchone()
            return int(row["id"])

    def roots(self) -> list[dict]:
        with self.connect() as conn:
            return [dict(row) for row in conn.execute("SELECT * FROM library_root ORDER BY id")]

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM setting WHERE key=?", (key,)).fetchone()
        return row["value"] if row is not None else default

    def set_setting(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO setting(key,value) VALUES (?,?)", (key, value)
            )

    def counts(self) -> dict[str, int]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) n FROM asset WHERE missing=0 GROUP BY status"
            ).fetchall()
            result = {row["status"]: row["n"] for row in rows}
            result["total"] = sum(result.values())
            result["embedded"] = conn.execute(
                "SELECT COUNT(DISTINCT asset_id) FROM embedding"
            ).fetchone()[0]
            result["analyzed"] = conn.execute(
                "SELECT COUNT(*) FROM audio_analysis WHERE status='ready'"
            ).fetchone()[0]
            for media_type in ("music", "effect", "unknown"):
                result[media_type] = conn.execute(
                    "SELECT COUNT(*) FROM asset WHERE missing=0 AND media_type=?", (media_type,)
                ).fetchone()[0]
            return result

    def delete_asset_record(self, asset_id: int) -> bool:
        with self._lock, self.connect() as conn:
            cursor = conn.execute("DELETE FROM asset WHERE id=?", (asset_id,))
            return cursor.rowcount > 0

    def delete_asset_records(self, asset_ids: list[int]) -> int:
        ids = list(dict.fromkeys(int(asset_id) for asset_id in asset_ids if int(asset_id) > 0))
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._lock, self.connect() as conn:
            cursor = conn.execute(f"DELETE FROM asset WHERE id IN ({placeholders})", ids)
            return cursor.rowcount
