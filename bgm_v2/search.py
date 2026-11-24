from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .db import Database
from .embedder import Embedder, normalize
from .text_embedding import TextEmbedder


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def lexical_score(query: str, filename: str) -> float:
    tokens = {token.casefold() for token in TOKEN_RE.findall(query) if len(token) > 1}
    if not tokens:
        return 0.0
    target = filename.casefold()
    compact_target = "".join(TOKEN_RE.findall(target))
    return sum(
        token in target or "".join(TOKEN_RE.findall(token)) in compact_target
        for token in tokens
    ) / len(tokens)


def search_assets(
    db: Database,
    embedder: Embedder,
    query: str = "",
    reference_path: Path | None = None,
    reference_vector: np.ndarray | None = None,
    extension: str = "",
    min_duration_ms: int | None = None,
    max_duration_ms: int | None = None,
    media_type: str = "",
    min_bpm: float | None = None,
    max_bpm: float | None = None,
    tag_node_ids: list[int] | None = None,
    tag_filters: dict[str, str] | None = None,
    limit: int = 20,
    offset: int = 0,
    semantic_weight: float = 0.85,
    text_embedder: TextEmbedder | None = None,
) -> list[dict]:
    clauses = ["a.missing=0", "e.provider=?", "e.model=?"]
    params: list[object] = [embedder.provider, embedder.model_name]
    if media_type:
        clauses.append("a.media_type=?")
        params.append(media_type)
    if extension:
        clauses.append("a.extension=?")
        params.append(extension if extension.startswith(".") else f".{extension}")
    if min_duration_ms is not None:
        clauses.append("a.duration_ms>=?")
        params.append(min_duration_ms)
    if max_duration_ms is not None:
        clauses.append("a.duration_ms<=?")
        params.append(max_duration_ms)
    if min_bpm is not None:
        clauses.append("x.bpm>=?")
        params.append(min_bpm)
    if max_bpm is not None:
        clauses.append("x.bpm<=?")
        params.append(max_bpm)
    for tag_node_id in tag_node_ids or []:
        clauses.append("""EXISTS (
            WITH RECURSIVE descendants(id) AS (
                SELECT ? UNION ALL SELECT n.id FROM tag_node n JOIN descendants d ON n.parent_id=d.id
            ) SELECT 1 FROM asset_tag_node atn JOIN descendants d ON d.id=atn.node_id
              WHERE atn.asset_id=a.id)""")
        params.append(tag_node_id)
    for namespace, code in (tag_filters or {}).items():
        if code:
            clauses.append(
                "EXISTS (SELECT 1 FROM asset_tag t WHERE t.asset_id=a.id "
                "AND t.namespace=? AND t.code=? AND t.rank_no<=3)"
            )
            params.extend([namespace, code])
    with db.connect() as conn:
        rows = [dict(row) for row in conn.execute(
            f"""SELECT a.*, e.vector, e.dimension, x.audio_type, x.audio_type_label,
            x.type_score, x.bpm, x.bpm_confidence, x.status AS analysis_status,
            x.error AS analysis_error, c.caption AS audio_caption
            FROM asset a JOIN embedding e ON e.asset_id=a.id
            LEFT JOIN audio_analysis x ON x.asset_id=a.id
            LEFT JOIN effect_caption_analysis c ON c.asset_id=a.id
              AND c.source_modified_ns=a.modified_ns AND c.status='ready'
            WHERE {' AND '.join(clauses)}""", params
        )]
    if not rows:
        return []
    asset_ids = [row["id"] for row in rows]
    placeholders = ",".join("?" for _ in asset_ids)
    with db.connect() as conn:
        tag_rows = conn.execute(
            f"""SELECT asset_id, namespace, label, score FROM asset_tag
            WHERE asset_id IN ({placeholders}) AND rank_no<=3
            ORDER BY asset_id, namespace, rank_no""",
            asset_ids,
        ).fetchall()
    tags_by_asset: dict[int, list[dict]] = {asset_id: [] for asset_id in asset_ids}
    for tag in tag_rows:
        tags_by_asset[tag["asset_id"]].append({
            "namespace": tag["namespace"], "label": tag["label"],
            "score": round(float(tag["score"]), 3),
        })
    with db.connect() as conn:
        effect_rows = conn.execute(
            f"""WITH RECURSIVE node_paths(id,path) AS (
                SELECT id,label FROM tag_node WHERE parent_id IS NULL
                UNION ALL
                SELECT n.id,node_paths.path || char(31) || n.label
                FROM tag_node n JOIN node_paths ON n.parent_id=node_paths.id
            )
            SELECT atn.asset_id,n.label,s.label scheme_label,s.code scheme_code,
                   node_paths.path,
                   NOT EXISTS(SELECT 1 FROM tag_node child WHERE child.parent_id=n.id) AS is_leaf
            FROM asset_tag_node atn JOIN tag_node n ON n.id=atn.node_id
            JOIN tag_scheme s ON s.id=n.scheme_id
            JOIN node_paths ON node_paths.id=n.id
            WHERE atn.asset_id IN ({placeholders}) ORDER BY atn.asset_id,s.id,n.label""",
            asset_ids,
        ).fetchall()
    for tag in effect_rows:
        tags_by_asset[tag["asset_id"]].append({
            "namespace": tag["scheme_code"], "label": tag["label"],
            "scheme_label": tag["scheme_label"], "score": 1.0,
            "path": tag["path"].split("\x1f"),
            "is_leaf": bool(tag["is_leaf"]),
        })
    text_vectors: dict[int, np.ndarray] = {}
    if query.strip() and text_embedder is not None:
        with db.connect() as conn:
            text_rows = conn.execute(
                f"""SELECT asset_id,dimension,vector FROM asset_text_embedding
                WHERE model=? AND asset_id IN ({placeholders})""",
                [text_embedder.model_name, *asset_ids],
            ).fetchall()
        text_vectors = {
            row["asset_id"]: np.frombuffer(
                row["vector"], dtype="<f4", count=row["dimension"]
            ).copy() for row in text_rows
        }
    vectors = np.stack([np.frombuffer(row["vector"], dtype="<f4", count=row["dimension"]) for row in rows])
    vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    query_vectors = []
    if query.strip():
        query_vectors.append(embedder.embed_text(query.strip()))
    if reference_vector is not None:
        query_vectors.append(reference_vector)
    elif reference_path is not None:
        query_vectors.append(embedder.embed_audio(reference_path))
    if not query_vectors:
        semantic = np.zeros(len(rows), dtype=np.float32)
    else:
        semantic = np.max(vectors @ np.stack([normalize(v) for v in query_vectors]).T, axis=1)
        semantic = np.clip((semantic + 1.0) / 2.0, 0.0, 1.0)
    text_semantic = np.zeros(len(rows), dtype=np.float32)
    if query.strip() and text_embedder is not None and text_vectors:
        query_text_vector = normalize(text_embedder.embed_query(query.strip()))
        for index, row in enumerate(rows):
            vector = text_vectors.get(row["id"])
            if vector is not None:
                text_semantic[index] = float(np.clip(
                    (normalize(vector) @ query_text_vector + 1.0) / 2.0, 0.0, 1.0
                ))
    for index, row in enumerate(rows):
        lexical = lexical_score(query, row["filename"])
        # Search-box keyword matching uses only the actually assigned node.
        # Ancestors remain available for tree filtering and detail display,
        # but a path such as 转场 > 呼啸 must not make 转场 an exact hit when
        # only the leaf 呼啸 is assigned to the asset.
        tag_text = " ".join(
            tag["label"] for tag in tags_by_asset.get(row["id"], [])
            if tag.get("is_leaf", True)
        )
        tag_lexical = lexical_score(query, tag_text)
        # Text search admission intentionally uses only filename and tags.
        # Metadata often contains artist/encoder fields that create noise.
        metadata_lexical = 0.0
        exact = max(lexical, tag_lexical)
        if query.strip() and text_embedder is not None:
            # Once admitted by exact text or BGE, CLAP helps rank but never
            # decides whether a text query result is allowed into the list.
            score = 0.70 * float(text_semantic[index]) + 0.30 * float(semantic[index])
        else:
            score = semantic_weight * float(semantic[index]) + (1 - semantic_weight) * exact
        row.pop("vector", None)
        try:
            row["metadata"] = json.loads(row.pop("metadata_json", "{}") or "{}")
        except json.JSONDecodeError:
            row["metadata"] = {}
        row["semantic_score"] = round(float(semantic[index]), 4)
        row["text_semantic_score"] = round(float(text_semantic[index]), 4)
        # Keep the original cosine scale for relevance gating. Mapping cosine
        # to 0..1 is convenient for display, but makes an unrelated zero
        # cosine look like a misleading 0.50 match.
        audio_cosine = 2.0 * float(semantic[index]) - 1.0
        text_cosine = 2.0 * float(text_semantic[index]) - 1.0
        row["audio_cosine"] = round(audio_cosine, 4)
        row["text_cosine"] = round(text_cosine, 4)
        row["lexical_score"] = round(lexical, 4)
        row["tag_lexical_score"] = round(tag_lexical, 4)
        row["metadata_lexical_score"] = round(metadata_lexical, 4)
        row["text_match_score"] = round(exact, 4)
        row["score"] = round(score, 4)
        reasons = [f"语义相似度 {row['semantic_score']:.2f}"]
        if lexical > 0:
            reasons.append("文件名关键词命中")
        if tag_lexical > 0:
            reasons.append("标签名称命中")
        if metadata_lexical > 0:
            reasons.append("元数据关键词命中")
        if text_semantic[index] > 0:
            reasons.append(f"文本语义相似度 {text_semantic[index]:.2f}")
        if reference_path or reference_vector is not None:
            reasons.append("参考歌相似")
        row["reason"] = "；".join(reasons)
        row["tags"] = tags_by_asset.get(row["id"], [])
        if query.strip():
            has_text_vector = row["id"] in text_vectors
            row["_high_relevance"] = bool(exact > 0 or (
                has_text_vector and text_cosine >= 0.55
            ))
        elif reference_path is not None or reference_vector is not None:
            row["_high_relevance"] = audio_cosine >= 0.30
        else:
            # Pure browsing and structured tag filters must not be hidden by
            # a semantic threshold when there is no search query.
            row["_high_relevance"] = True
    if query.strip() or reference_path is not None or reference_vector is not None:
        rows = [row for row in rows if row["_high_relevance"]]
    for row in rows:
        row.pop("_high_relevance", None)
    # 用户明确输入了文件名或标签时，文字命中应优先于宽泛的向量相似度。
    rows.sort(key=lambda item: (
        -item["text_match_score"], -item["score"], item["filename"].casefold()
    ))
    safe_limit = max(1, min(limit, 101))
    safe_offset = max(0, offset)
    return rows[safe_offset:safe_offset + safe_limit]
