from __future__ import annotations

import hashlib
from pathlib import Path

from .db import Database


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_effect_duplicates(db: Database) -> dict:
    """Mark byte-identical effects, retaining the oldest record as canonical."""
    with db.connect() as conn:
        assets = [dict(row) for row in conn.execute(
            """SELECT id,path,size_bytes,modified_ns,content_sha256,hash_modified_ns
            FROM asset WHERE missing=0 AND media_type='effect' ORDER BY id"""
        )]
    by_size: dict[int, list[dict]] = {}
    for asset in assets:
        by_size.setdefault(int(asset["size_bytes"]), []).append(asset)

    candidates = [asset for group in by_size.values() if len(group) > 1 for asset in group]
    hashes: dict[int, str] = {}
    updates = []
    failed = 0
    for asset in candidates:
        cached = asset.get("content_sha256")
        if cached and asset.get("hash_modified_ns") == asset["modified_ns"]:
            hashes[asset["id"]] = cached
            continue
        try:
            value = _sha256(Path(asset["path"]))
            hashes[asset["id"]] = value
            updates.append((value, asset["modified_ns"], asset["id"]))
        except OSError:
            failed += 1

    groups: dict[tuple[int, str], list[int]] = {}
    for size, same_size in by_size.items():
        if len(same_size) < 2:
            continue
        for asset in same_size:
            value = hashes.get(asset["id"])
            if value:
                groups.setdefault((size, value), []).append(asset["id"])
    duplicates: list[tuple[int, int]] = []
    for ids in groups.values():
        ids.sort()
        duplicates.extend((asset_id, ids[0]) for asset_id in ids[1:])

    with db.connect() as conn:
        if updates:
            conn.executemany(
                "UPDATE asset SET content_sha256=?,hash_modified_ns=? WHERE id=?", updates
            )
        # Restore records that were duplicates in a previous run before recomputing.
        conn.execute(
            """UPDATE asset SET duplicate_of=NULL,status=CASE
            WHEN EXISTS(SELECT 1 FROM embedding e WHERE e.asset_id=asset.id)
            THEN 'ready' ELSE 'discovered' END
            WHERE missing=0 AND media_type='effect' AND duplicate_of IS NOT NULL"""
        )
        if duplicates:
            conn.executemany(
                "UPDATE asset SET duplicate_of=?,status='duplicate',error=NULL WHERE id=?",
                [(canonical, duplicate) for duplicate, canonical in duplicates],
            )
            duplicate_ids = [duplicate for duplicate, _ in duplicates]
            placeholders = ",".join("?" for _ in duplicate_ids)
            conn.execute(f"DELETE FROM embedding WHERE asset_id IN ({placeholders})", duplicate_ids)
            conn.execute(f"DELETE FROM waveform WHERE asset_id IN ({placeholders})", duplicate_ids)
            conn.execute(f"DELETE FROM audio_analysis WHERE asset_id IN ({placeholders})", duplicate_ids)
            conn.execute(f"DELETE FROM effect_action_analysis WHERE asset_id IN ({placeholders})", duplicate_ids)
            conn.execute(
                f"DELETE FROM asset_tag_node WHERE asset_id IN ({placeholders}) AND assigned_by<>'user'",
                duplicate_ids,
            )
    duplicate_groups = sum(1 for ids in groups.values() if len(ids) > 1)
    return {
        "effect_files": len(assets),
        "duplicate_files": len(duplicates),
        "indexable_effects": len(assets) - len(duplicates),
        "duplicate_groups": duplicate_groups,
        "hash_failed": failed,
    }
