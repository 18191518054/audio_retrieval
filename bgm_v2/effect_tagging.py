from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from .db import Database
from .embedder import Embedder, normalize
from .text_embedding import TextEmbedder
from .taxonomy import UCS_VERSION, ensure_ucs


Progress = Callable[[dict], None]
AUTO_SOURCE = "semantic-v2"
SEMANTIC_POLICY = {
    "ucs": (0.34, 0.045),
    "recommended": (0.30, 0.025),
    "instrument_material": (0.34, 0.05),
}
ATTRIBUTE_DIMENSIONS = {"速度", "频段", "音色", "方向", "强度", "空间感", "距离", "质感", "节奏", "长度"}
EVENT_TRANSLATIONS = {
    "whoosh, swoosh, swish": "呼啸 转场 掠过 穿梭",
    "drip": "水滴 滴水",
    "boom": "爆炸 轰鸣 重击",
    "eruption": "喷发 爆发",
    "walk, footsteps": "行走 脚步",
    "finger snapping": "打响指 手指弹响",
    "tearing": "撕裂 撕扯",
    "scissors": "剪刀 剪切",
    "crumpling, crinkling": "揉皱 褶皱 摩擦",
    "crunch": "咔嚓 碎裂 咀嚼",
    "chink, clink": "金属 玻璃 轻碰",
    "glass": "玻璃",
    "vehicle": "车辆 交通",
    "speech": "人声 说话",
    "animal": "动物",
    "squawk": "鸟叫 尖叫",
}
GENERIC_EVENTS = {"music", "sound effect", "silence", "noise"}


def _localized_event_text(text: str) -> str:
    useful = []
    remaining = text
    lowered = text.casefold()
    for key, translated in EVENT_TRANSLATIONS.items():
        if key in lowered:
            useful.extend([key, translated])
            remaining = re.sub(re.escape(key), " ", remaining, flags=re.IGNORECASE)
    parts = [part.strip() for part in re.split(r"[,;]", remaining) if part.strip()]
    for part in parts:
        key = part.casefold()
        if key in GENERIC_EVENTS:
            continue
        useful.append(part)
    return "；".join(dict.fromkeys(useful))


def _ucs_metadata_match(asset: dict, rows: list[dict], nodes_by_code: dict[str, int]) -> int | None:
    try:
        metadata = json.loads(asset.get("metadata_json") or "{}")
    except json.JSONDecodeError:
        return None
    cat_id = str(metadata.get("UCS CatID", "")).strip()
    if cat_id in nodes_by_code:
        return nodes_by_code[cat_id]
    category = str(metadata.get("UCS Category", "")).strip().casefold()
    subcategory = str(metadata.get("UCS SubCategory", "")).strip().casefold()
    if not category or not subcategory:
        return None
    matches = [row["CatID"].strip() for row in rows
               if row["Category"].strip().casefold() == category
               and row["SubCategory"].strip().casefold() == subcategory]
    return nodes_by_code.get(matches[0]) if len(matches) == 1 else None


def _ucs_filename_match(asset: dict, nodes_by_code: dict[str, int]) -> int | None:
    stem = Path(asset["filename"]).stem
    match = re.match(r"^([^_]+)_", stem)
    if not match:
        return None
    return nodes_by_code.get(match.group(1))


def _ucs_descriptive_match(asset: dict, rows: list[dict],
                           nodes_by_code: dict[str, int]) -> int | None:
    """Match an unambiguous official UCS name/synonym in filename metadata."""
    try:
        metadata = json.loads(asset.get("metadata_json") or "{}")
    except json.JSONDecodeError:
        metadata = {}
    useful_keys = ("title", "description", "comment", "keyword", "subject",
                   "标题", "描述", "说明", "关键词", "注释")
    values = [Path(asset["filename"]).stem]
    values.extend(str(value) for key, value in metadata.items()
                  if value and any(token in str(key).casefold() for token in useful_keys))
    haystack = re.sub(r"[^a-z0-9]+", "", " ".join(values).casefold())
    if not haystack:
        return None
    matched: set[str] = set()
    for row in rows:
        terms = [row.get("SubCategory", "")]
        terms.extend(row.get("Synonyms", "").split(","))
        for term in terms:
            normalized = re.sub(r"[^a-z0-9]+", "", term.casefold())
            if len(normalized) >= 5 and normalized in haystack:
                matched.add(row["CatID"].strip())
                break
    if len(matched) != 1:
        return None
    return nodes_by_code.get(next(iter(matched)))


def _semantic_candidates(db: Database, scheme_code: str) -> list[dict]:
    with db.connect() as conn:
        rows = [dict(row) for row in conn.execute(
            """WITH RECURSIVE paths(id,path,depth,root_order) AS (
                SELECT id,label,0,sort_order FROM tag_node WHERE parent_id IS NULL
                UNION ALL
                SELECT n.id,paths.path || ' > ' || n.label,paths.depth+1,paths.root_order
                FROM tag_node n JOIN paths ON n.parent_id=paths.id
            )
            SELECT n.id,n.parent_id,n.code,n.label,n.description,n.examples_json,
                   paths.path,paths.depth,paths.root_order
            FROM tag_node n JOIN tag_scheme s ON s.id=n.scheme_id
            JOIN paths ON paths.id=n.id
            WHERE s.code=? AND (? IN ('recommended','instrument_material') OR NOT EXISTS
                (SELECT 1 FROM tag_node child WHERE child.parent_id=n.id))
            ORDER BY n.sort_order,n.label""", (scheme_code, scheme_code)
        )]
        child_rows = conn.execute(
            """SELECT n.parent_id,n.label FROM tag_node n
            JOIN tag_scheme s ON s.id=n.scheme_id
            WHERE s.code=? AND n.parent_id IS NOT NULL
            ORDER BY n.sort_order,n.label""", (scheme_code,)
        ).fetchall()
    children_by_parent: dict[int, list[str]] = {}
    child_ids_by_parent: dict[int, list[int]] = {}
    for child in child_rows:
        children_by_parent.setdefault(child["parent_id"], []).append(child["label"])
    for row in rows:
        if row["parent_id"] is not None:
            child_ids_by_parent.setdefault(row["parent_id"], []).append(row["id"])
    labels_by_id = {row["id"]: row["label"] for row in rows}

    def descendant_labels(node_id: int) -> list[str]:
        found: list[str] = []
        pending = list(child_ids_by_parent.get(node_id, []))
        while pending:
            child_id = pending.pop()
            found.append(labels_by_id[child_id])
            pending.extend(child_ids_by_parent.get(child_id, []))
        return found

    for row in rows:
        try:
            examples = json.loads(row["examples_json"] or "[]")
        except json.JSONDecodeError:
            examples = []
        parts = [f"A sound effect in the category: {row['path']}."]
        if row["description"]:
            parts.append(row["description"])
        if examples:
            parts.append("Typical examples: " + "; ".join(map(str, examples[:6])))
        child_labels = children_by_parent.get(row["id"], [])
        if child_labels:
            # Parent labels such as "主观音效" are too abstract in isolation.
            # Their direct children provide taxonomy-owned semantics without a
            # hard-coded filename-to-label mapping and automatically track edits.
            parts.append("Includes subcategories: " + "; ".join(child_labels[:48]))
        row["child_labels"] = child_labels
        row["descendant_labels"] = descendant_labels(row["id"])
        row["prompt"] = " ".join(parts)
    return rows


def _prepare_semantic_models(
    db: Database, embedder: Embedder, progress: Progress | None = None
) -> dict[str, tuple[list[dict], np.ndarray]]:
    models = {}
    all_candidates: list[dict] = []
    for scheme_code in SEMANTIC_POLICY:
        candidates = _semantic_candidates(db, scheme_code)
        for item in candidates:
            item["scheme_code"] = scheme_code
            item["content_hash"] = hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest()
        all_candidates.extend(candidates)
    if not all_candidates:
        return models

    candidate_ids = [item["id"] for item in all_candidates]
    placeholders = ",".join("?" for _ in candidate_ids)
    with db.connect() as conn:
        cached_rows = conn.execute(
            f"""SELECT node_id,content_hash,dimension,vector FROM tag_embedding
            WHERE provider=? AND model=? AND node_id IN ({placeholders})""",
            [embedder.provider, embedder.model_name, *candidate_ids],
        ).fetchall()
    cached = {row["node_id"]: row for row in cached_rows}
    vectors_by_id: dict[int, np.ndarray] = {}
    missing: list[dict] = []
    for item in all_candidates:
        row = cached.get(item["id"])
        if row is not None and row["content_hash"] == item["content_hash"]:
            vectors_by_id[item["id"]] = np.frombuffer(
                row["vector"], dtype="<f4", count=row["dimension"]
            ).copy()
        else:
            missing.append(item)
    total = len(all_candidates)
    completed = total - len(missing)
    if progress:
        progress({"stage": "tag_vectors", "total": total, "cached": completed,
                  "generated": 0, "message": f"标签向量缓存命中 {completed}/{total}"})
    for start in range(0, len(missing), 16):
        batch = missing[start:start + 16]
        batch_vectors = embedder.embed_texts([item["prompt"] for item in batch], batch_size=16)
        now = time.time()
        records = []
        for item, vector in zip(batch, batch_vectors):
            value = np.asarray(normalize(vector), dtype="<f4")
            vectors_by_id[item["id"]] = value
            records.append((item["id"], embedder.provider, embedder.model_name,
                            item["content_hash"], value.size, value.tobytes(), now))
        with db.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO tag_embedding
                (node_id,provider,model,content_hash,dimension,vector,created_at)
                VALUES (?,?,?,?,?,?,?)""", records,
            )
        completed += len(batch)
        if progress:
            progress({"stage": "tag_vectors", "total": total,
                      "cached": total - len(missing),
                      "generated": completed - (total - len(missing)),
                      "message": f"正在生成标签向量 {completed}/{total}"})
    for scheme_code in SEMANTIC_POLICY:
        candidates = [item for item in all_candidates if item["scheme_code"] == scheme_code]
        if candidates:
            models[scheme_code] = (
                candidates, np.stack([vectors_by_id[item["id"]] for item in candidates])
            )
    return models


def prepare_effect_tag_vectors(
    db: Database,
    cache_dir: Path,
    embedder: Embedder,
    text_embedder: TextEmbedder,
    progress: Progress | None = None,
) -> dict[str, int]:
    """Initialize all persistent effect-tag vectors without classifying assets."""
    ensure_ucs(db, cache_dir)
    clap_models = _prepare_semantic_models(db, embedder, progress)
    bge_model = prepare_recommended_text_vectors(db, text_embedder, progress)
    material_bge_model = prepare_scheme_text_vectors(
        db, text_embedder, "instrument_material", progress
    )
    clap_total = sum(len(model[0]) for model in clap_models.values())
    bge_total = ((len(bge_model[0]) if bge_model else 0)
                 + (len(material_bge_model[0]) if material_bge_model else 0))
    return {"clap_total": clap_total, "bge_total": bge_total}


def _semantic_match(audio_vector: np.ndarray, model: tuple[list[dict], np.ndarray],
                    policy: tuple[float, float]) -> tuple[int, float] | None:
    candidates, vectors = model
    scores = vectors @ normalize(audio_vector)
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    best_score = float(scores[best])
    second_score = float(scores[int(order[1])]) if len(order) > 1 else -1.0
    threshold, minimum_margin = policy
    if best_score < threshold or best_score - second_score < minimum_margin:
        return None
    return int(candidates[best]["id"]), best_score


def prepare_recommended_text_vectors(
    db: Database, text_embedder: TextEmbedder, progress: Progress | None = None
) -> tuple[list[dict], np.ndarray] | None:
    return prepare_scheme_text_vectors(db, text_embedder, "recommended", progress)


def prepare_scheme_text_vectors(
    db: Database,
    text_embedder: TextEmbedder,
    scheme_code: str,
    progress: Progress | None = None,
) -> tuple[list[dict], np.ndarray] | None:
    candidates = _semantic_candidates(db, scheme_code)
    if not candidates:
        return None
    title = "推荐分类路径" if scheme_code == "recommended" else "乐器材质分类路径"
    for item in candidates:
        item["bge_prompt"] = "；".join(filter(None, [
            f"{title}：{item['path']}", item.get("description", ""),
            "；".join(json.loads(item.get("examples_json") or "[]")[:6]),
            ("包含下级分类：" + "；".join(item.get("child_labels", [])[:48]))
            if item.get("child_labels") else "",
        ]))
        item["bge_hash"] = hashlib.sha256(item["bge_prompt"].encode("utf-8")).hexdigest()
    ids = [item["id"] for item in candidates]
    placeholders = ",".join("?" for _ in ids)
    with db.connect() as conn:
        rows = conn.execute(
            f"""SELECT node_id,content_hash,dimension,vector FROM tag_embedding
            WHERE provider='bge' AND model=? AND node_id IN ({placeholders})""",
            [text_embedder.model_name, *ids],
        ).fetchall()
    expected = {item["id"]: item["bge_hash"] for item in candidates}
    vectors = {}
    for row in rows:
        if row["content_hash"] == expected[row["node_id"]]:
            vectors[row["node_id"]] = np.frombuffer(
                row["vector"], dtype="<f4", count=row["dimension"]
            ).copy()
    missing = [item for item in candidates if item["id"] not in vectors]
    if progress:
        progress({"stage": f"{scheme_code}_text_vectors", "total": len(candidates),
                  "cached": len(candidates) - len(missing), "generated": 0})
    for start in range(0, len(missing), 16):
        batch = missing[start:start + 16]
        generated = text_embedder.embed_documents(
            [item["bge_prompt"] for item in batch], batch_size=16
        )
        records = []
        for item, vector in zip(batch, generated):
            value = np.asarray(normalize(vector), dtype="<f4")
            vectors[item["id"]] = value
            records.append((item["id"], "bge", text_embedder.model_name,
                            item["bge_hash"], value.size, value.tobytes(), time.time()))
        with db.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO tag_embedding
                (node_id,provider,model,content_hash,dimension,vector,created_at)
                VALUES (?,?,?,?,?,?,?)""", records,
            )
        if progress:
            progress({"stage": f"{scheme_code}_text_vectors", "total": len(candidates),
                      "cached": len(candidates) - len(missing),
                      "generated": min(start + len(batch), len(missing))})
    return candidates, np.stack([vectors[item["id"]] for item in candidates])


def _subset_model(model: tuple[list[dict], np.ndarray], root_order: int):
    candidates, vectors = model
    indices = [index for index, item in enumerate(candidates)
               if item.get("root_order") == root_order]
    if not indices:
        return None
    return [candidates[index] for index in indices], vectors[indices]


def _instrument_material_matches(
    audio_vector: np.ndarray,
    model: tuple[list[dict], np.ndarray],
    action_context: list[dict],
    embedder: Embedder,
    conditioned_cache: dict[str, np.ndarray],
    db: Database | None = None,
) -> list[tuple[int, float, str]]:
    """Classify instruments directly; classify materials only after AST action evidence.

    A material is accepted only when generic CLAP and action-conditioned CLAP choose
    the same leaf. AST and CLAP therefore cross-check different parts of the decision.
    """
    matches: list[tuple[int, float, str]] = []
    instrument_model = _subset_model(model, 0)
    if instrument_model:
        instrument = _semantic_match(audio_vector, instrument_model, (0.36, 0.06))
        if instrument:
            matches.append((instrument[0], instrument[1], "CLAP 乐器高置信度判断"))

    # Accuracy-first abstention: without a reliable action from the independent AST
    # model, material inference is too unconstrained and is deliberately skipped.
    reliable_actions = [item for item in action_context if float(item.get("score", 0)) >= 0.25]
    material_model = _subset_model(model, 1)
    if not reliable_actions or not material_model:
        return matches
    candidates, generic_vectors = material_model
    audio = normalize(audio_vector)
    generic_scores = generic_vectors @ audio
    generic_order = np.argsort(generic_scores)[::-1]
    if not len(generic_order):
        return matches
    action_text = ", ".join(str(item.get("label", "")) for item in reliable_actions[:3])
    cache_key = action_text.casefold()
    conditioned_vectors = conditioned_cache.get(cache_key)
    if conditioned_vectors is None:
        prompts = [f"A sound of {action_text}, involving the material {item['path']}."
                   for item in candidates]
        action_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        content_hashes = [hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                          for prompt in prompts]
        persisted: dict[int, np.ndarray] = {}
        if db is not None:
            node_ids = [int(item["id"]) for item in candidates]
            placeholders = ",".join("?" for _ in node_ids)
            with db.connect() as conn:
                rows = conn.execute(
                    f"""SELECT node_id,content_hash,dimension,vector
                    FROM material_context_embedding
                    WHERE provider=? AND model=? AND action_hash=?
                    AND node_id IN ({placeholders})""",
                    [embedder.provider, embedder.model_name, action_hash, *node_ids],
                ).fetchall()
            expected = {int(item["id"]): content_hashes[index]
                        for index, item in enumerate(candidates)}
            for row in rows:
                if row["content_hash"] == expected[row["node_id"]]:
                    persisted[row["node_id"]] = np.frombuffer(
                        row["vector"], dtype="<f4", count=row["dimension"]
                    ).copy()
        missing_indices = [index for index, item in enumerate(candidates)
                           if int(item["id"]) not in persisted]
        for start in range(0, len(missing_indices), 16):
            indices = missing_indices[start:start + 16]
            generated = embedder.embed_texts([prompts[index] for index in indices], batch_size=16)
            records = []
            for index, vector in zip(indices, generated):
                value = np.asarray(normalize(vector), dtype="<f4")
                node_id = int(candidates[index]["id"])
                persisted[node_id] = value
                records.append((node_id, embedder.provider, embedder.model_name,
                                action_hash, content_hashes[index], value.size,
                                value.tobytes(), time.time()))
            if db is not None:
                with db.connect() as conn:
                    conn.executemany(
                        """INSERT OR REPLACE INTO material_context_embedding
                        (node_id,provider,model,action_hash,content_hash,dimension,vector,created_at)
                        VALUES (?,?,?,?,?,?,?,?)""", records,
                    )
        conditioned_vectors = np.stack(
            [persisted[int(item["id"])] for item in candidates]
        )
        conditioned_cache[cache_key] = conditioned_vectors
    conditioned_scores = conditioned_vectors @ audio
    conditioned_order = np.argsort(conditioned_scores)[::-1]
    generic_best, conditioned_best = int(generic_order[0]), int(conditioned_order[0])
    if generic_best != conditioned_best:
        return matches
    generic_second = float(generic_scores[int(generic_order[1])]) if len(generic_order) > 1 else -1.0
    conditioned_second = (float(conditioned_scores[int(conditioned_order[1])])
                          if len(conditioned_order) > 1 else -1.0)
    generic_score = float(generic_scores[generic_best])
    conditioned_score = float(conditioned_scores[conditioned_best])
    if (generic_score < 0.30 or generic_score - generic_second < 0.04
            or conditioned_score < 0.32 or conditioned_score - conditioned_second < 0.04):
        return matches
    confidence = min(generic_score, conditioned_score)
    matches.append((candidates[generic_best]["id"], confidence,
                    f"AST 动作（{action_text}）与 CLAP 材质双路一致"))
    return matches


def _hierarchical_instrument_material_matches(
    audio_vector: np.ndarray,
    model: tuple[list[dict], np.ndarray],
    text_model: tuple[list[dict], np.ndarray] | None,
    filename: str,
    metadata_text: str,
    action_context: list[dict],
    audio_caption: str,
    text_embedder: TextEmbedder | None,
) -> list[tuple[int, float, str]]:
    """Classify instrument and material independently along their existing trees."""
    candidates, vectors = model
    audio = normalize(audio_vector)
    by_parent: dict[int | None, list[int]] = {}
    for index, item in enumerate(candidates):
        by_parent.setdefault(item["parent_id"], []).append(index)
    bge_by_id: dict[int, np.ndarray] = {}
    if text_model:
        text_candidates, text_vectors = text_model
        bge_by_id = {item["id"]: text_vectors[index]
                     for index, item in enumerate(text_candidates)}

    stem = Path(filename).stem.strip()
    meaningful = re.sub(r"[\W_]+", "", stem, flags=re.UNICODE)
    generic_name = bool(re.fullmatch(
        r"(?i)(?:audio|sound|track|take|rec|recording|final|copy|新建|未命名|最终版)?\d*",
        meaningful,
    ))
    name_quality = 0 if len(meaningful) < 3 or generic_name else (2 if len(meaningful) >= 6 else 1)
    filename_vector = caption_vector = action_vector = None
    if text_embedder:
        if name_quality:
            source = f"文件名：{stem}" + (f"；元数据：{metadata_text}" if metadata_text else "")
            filename_vector = normalize(text_embedder.embed_query(source))
        caption_text = _localized_event_text(
            re.sub(r"(?i)^this audio contains\s+|\.$", "", audio_caption.strip())
        )
        if caption_text:
            caption_vector = normalize(text_embedder.embed_query(f"音频内容描述：{caption_text}"))
        reliable = [item for item in action_context if float(item.get("score", 0)) >= 0.22]
        action_text = "；".join(filter(None, (
            _localized_event_text(str(item.get("label", ""))) for item in reliable[:4]
        )))
        if action_text:
            action_vector = normalize(text_embedder.embed_query(f"声音动作：{action_text}"))
    has_reliable_action = any(
        float(item.get("score", 0)) >= 0.22 for item in action_context
    )

    def choose(indices: list[int], kind: str, level: int) -> tuple[int, float, str] | None:
        terms_by_index: dict[int, set[str]] = {}
        term_counts: dict[str, int] = {}
        for index in indices:
            item = candidates[index]
            terms: set[str] = set()
            for label in [item["label"], *item.get("descendant_labels", [])]:
                terms.update(term.casefold() for term in re.findall(
                    r"[a-zA-Z]{3,}|[\u4e00-\u9fff]{2,}", label
                ))
                # Compound material labels such as "塑料与树脂" need partial
                # matching. Keep instrument names whole; fragments from long
                # instrument categories produced unrelated false positives.
                if kind == "material":
                    for chinese in re.findall(r"[\u4e00-\u9fff]{3,}", label):
                        terms.update(chinese[pos:pos + 2]
                                     for pos in range(len(chinese) - 1))
            terms_by_index[index] = terms
            for term in terms:
                term_counts[term] = term_counts.get(term, 0) + 1
        rows = []
        for index in indices:
            item = candidates[index]
            audio_score = float(vectors[index] @ audio)
            text_vector = bge_by_id.get(item["id"])
            filename_score = (float(text_vector @ filename_vector)
                              if text_vector is not None and filename_vector is not None else None)
            caption_score = (float(text_vector @ caption_vector)
                             if text_vector is not None and caption_vector is not None else None)
            action_score = (float(text_vector @ action_vector)
                            if text_vector is not None and action_vector is not None else None)
            lexical = False
            if name_quality and filename_score is not None:
                unique_terms = [term for term in terms_by_index[index]
                                if term_counts.get(term) == 1]
                lexical = any(term in stem.casefold() for term in unique_terms)
                if lexical:
                    filename_score += 0.35
            rows.append((index, audio_score, filename_score,
                         caption_score, action_score, lexical))
        if not rows:
            return None
        audio_rows = sorted(rows, key=lambda row: row[1], reverse=True)
        audio_best = audio_rows[0]
        audio_second = audio_rows[1][1] if len(audio_rows) > 1 else -1.0
        audio_margin = audio_best[1] - audio_second
        audio_minimum = 0.30 if level == 0 else 0.29
        audio_high = audio_best[1] >= audio_minimum and audio_margin >= 0.025
        if kind == "material":
            audio_high = audio_high and has_reliable_action

        # Exact taxonomy terms are explicit fallback evidence. A conflict with
        # high-confidence audio causes abstention instead of silently trusting
        # either source.
        lexical_rows = [row for row in rows if row[5]]
        lexical_best = (max(lexical_rows, key=lambda row: row[2] or -1.0)
                        if lexical_rows else None)
        if audio_high:
            if lexical_best is not None and lexical_best[0] != audio_best[0]:
                return None
            if kind == "material" and lexical_best is None:
                filename_rows = [row for row in rows if row[2] is not None]
                caption_rows = [row for row in rows if row[3] is not None]
                filename_best = (max(filename_rows, key=lambda row: row[2])
                                 if filename_rows else None)
                caption_best = (max(caption_rows, key=lambda row: row[3])
                                if caption_rows else None)
                filename_confirms = (
                    filename_best is not None
                    and filename_best[0] == audio_best[0]
                    and (filename_best[2] or -1.0) >= 0.55
                )
                caption_confirms = (
                    caption_best is not None
                    and caption_best[0] == audio_best[0]
                    and (caption_best[3] or -1.0) >= 0.56
                )
                if not (filename_confirms or caption_confirms):
                    return None
            confidence = float(np.clip((audio_best[1] + 1.0) / 2.0, 0.0, 1.0))
            return audio_best[0], confidence, "audio"

        # Only after audio abstains may filename, metadata and generated audio
        # description provide the answer.
        fallback = lexical_best
        fallback_source = "filename"
        if fallback is None:
            filename_rows = [row for row in rows if row[2] is not None]
            caption_rows = [row for row in rows if row[3] is not None]
            filename_best = (max(filename_rows, key=lambda row: row[2])
                             if filename_rows else None)
            caption_best = (max(caption_rows, key=lambda row: row[3])
                            if caption_rows else None)
            filename_order = sorted((row[2] for row in filename_rows), reverse=True)
            filename_margin = (filename_order[0] - filename_order[1]
                               if len(filename_order) > 1 else 1.0)
            if (filename_best is not None and caption_best is not None
                    and filename_best[0] == caption_best[0]
                    and (filename_best[2] or -1.0) >= 0.52
                    and (caption_best[3] or -1.0) >= 0.54
                    and filename_margin >= 0.020):
                fallback = filename_best
                fallback_source = "description"
        if fallback is None:
            return None
        if (audio_best[0] != fallback[0] and audio_best[1] >= 0.27
                and audio_margin >= 0.015):
            return None
        if kind == "material" and not (has_reliable_action or fallback[5]):
            return None
        auxiliary = max(value for value in (fallback[2], fallback[3], fallback[4])
                        if value is not None)
        confidence = float(np.clip((auxiliary + 1.0) / 2.0, 0.0, 1.0))
        return fallback[0], confidence, fallback_source

    matches: list[tuple[int, float, str]] = []
    roots = [index for index in by_parent.get(None, [])
             if candidates[index].get("root_order") in (0, 1)]
    for root_index in roots:
        kind = "instrument" if candidates[root_index].get("root_order") == 0 else "material"
        children = by_parent.get(candidates[root_index]["id"], [])
        chosen = choose(children, kind, 0)
        if chosen is None:
            continue
        current_index, confidence, source = chosen
        level = 1
        while by_parent.get(candidates[current_index]["id"]):
            deeper = choose(by_parent[candidates[current_index]["id"]], kind, level)
            if deeper is None:
                break
            current_index, confidence, source = deeper
            level += 1
        evidence = {
            "audio": "CLAP音频内容高置信度分层判断",
            "filename": "音频模型未确认，按明确文件名或元数据分层判断",
            "description": "音频模型未确认，按文件名与音频描述一致性分层判断",
        }[source]
        matches.append((candidates[current_index]["id"], confidence, evidence))
    return matches


def _recommended_matches(
    audio_vector: np.ndarray,
    model: tuple[list[dict], np.ndarray],
    prototypes: dict[int, np.ndarray] | None = None,
    text_model: tuple[list[dict], np.ndarray] | None = None,
    filename: str = "",
    metadata_text: str = "",
    action_context: list[dict] | None = None,
    text_embedder: TextEmbedder | None = None,
    audio_caption: str = "",
) -> list[tuple[int, float]]:
    candidates, vectors = model
    audio = normalize(audio_vector)
    by_parent: dict[int | None, list[int]] = {}
    for index, item in enumerate(candidates):
        by_parent.setdefault(item["parent_id"], []).append(index)

    stem = Path(filename).stem.strip()
    meaningful = re.sub(r"[\W_]+", "", stem, flags=re.UNICODE)
    generic_name = bool(re.fullmatch(
        r"(?i)(?:audio|sound|track|take|rec|recording|final|copy|新建|未命名|最终版)?\d*",
        meaningful,
    ))
    digit_ratio = sum(char.isdigit() for char in meaningful) / max(1, len(meaningful))
    if len(meaningful) < 3 or generic_name or digit_ratio > 0.6:
        name_quality = 0
    elif len(meaningful) >= 6 or re.search(r"[\u4e00-\u9fff]{3,}", meaningful):
        name_quality = 2
    else:
        name_quality = 1

    bge_by_id: dict[int, np.ndarray] = {}
    filename_vector = action_vector = caption_vector = None
    if text_model and text_embedder:
        text_candidates, text_vectors = text_model
        bge_by_id = {item["id"]: text_vectors[index]
                     for index, item in enumerate(text_candidates)}
        if name_quality:
            source = f"文件名：{stem}"
            if metadata_text:
                source += f"；元数据：{metadata_text}"
            filename_vector = normalize(text_embedder.embed_query(source))
        reliable_actions = [item for item in (action_context or [])
                            if float(item.get("score", 0)) >= 0.25]
        if reliable_actions:
            action_text = "；".join(filter(None, (
                _localized_event_text(str(item.get("label", "")))
                for item in reliable_actions[:3]
            )))
            if action_text:
                action_vector = normalize(text_embedder.embed_query(f"检测到的声音动作：{action_text}"))
        if audio_caption.strip():
            caption_text = _localized_event_text(
                re.sub(r"(?i)^this audio contains\s+|\.$", "", audio_caption.strip())
            )
            if caption_text:
                caption_vector = normalize(text_embedder.embed_query(
                    f"音频内容描述：{caption_text}"
                ))

    def choose(indices: list[int]) -> tuple[int, float] | None:
        scored = []
        signal_rows = []
        for index in indices:
            item = candidates[index]
            audio_score = float(vectors[index] @ audio)
            prototype = (prototypes or {}).get(item["id"])
            if prototype is not None:
                audio_score = 0.65 * audio_score + 0.35 * float(prototype @ audio)
            node_text_vector = bge_by_id.get(item["id"])
            filename_score = (float(node_text_vector @ filename_vector)
                              if node_text_vector is not None and filename_vector is not None else None)
            action_score = (float(node_text_vector @ action_vector)
                             if node_text_vector is not None and action_vector is not None else None)
            caption_score = (float(node_text_vector @ caption_vector)
                             if node_text_vector is not None and caption_vector is not None else None)
            if filename_score is not None and name_quality:
                tree_terms: list[str] = []
                for label in [item["label"], *item.get("descendant_labels", [])]:
                    tree_terms.extend(re.findall(
                        r"[a-zA-Z]{3,}|[\u4e00-\u9fff]{2,}", label
                    ))
                if any(term.casefold() in stem.casefold() for term in tree_terms):
                    filename_score += 0.10
            if int(item.get("depth", 0)) == 0:
                child_indices = by_parent.get(item["id"], [])
                if child_indices:
                    audio_score = max(
                        audio_score,
                        max(float(vectors[child_index] @ audio)
                            for child_index in child_indices),
                    )
                    child_text_vectors = [bge_by_id.get(candidates[child_index]["id"])
                                          for child_index in child_indices]
                    child_text_vectors = [value for value in child_text_vectors
                                          if value is not None]
                    if filename_vector is not None and child_text_vectors:
                        filename_score = max(
                            filename_score if filename_score is not None else -1.0,
                            max(float(value @ filename_vector) for value in child_text_vectors),
                        )
                    if action_vector is not None and child_text_vectors:
                        action_score = max(
                            action_score if action_score is not None else -1.0,
                            max(float(value @ action_vector) for value in child_text_vectors),
                        )
                    if caption_vector is not None and child_text_vectors:
                        caption_score = max(
                            caption_score if caption_score is not None else -1.0,
                            max(float(value @ caption_vector) for value in child_text_vectors),
                        )
            if name_quality == 2:
                weights = [0.25, 0.45, 0.10, 0.20]
            elif name_quality == 1:
                weights = [0.40, 0.15, 0.15, 0.30]
            else:
                weights = [0.48, 0.0, 0.17, 0.35]
            signals = [audio_score, filename_score, action_score, caption_score]
            active = [(value, weight) for value, weight in zip(signals, weights)
                      if value is not None and weight > 0]
            total_weight = sum(weight for _, weight in active)
            score = sum(value * weight for value, weight in active) / max(total_weight, 1e-9)
            if name_quality:
                label_tokens = re.findall(r"[a-zA-Z]{3,}|[\u4e00-\u9fff]{2,}", item["label"])
                if any(token.casefold() in stem.casefold() for token in label_tokens):
                    score += 0.10
            scored.append((score, index))
            signal_rows.append((index, audio_score, filename_score, action_score, caption_score))
        scored.sort(reverse=True)
        if not scored:
            return None
        best_score, best_index = scored[0]
        second = scored[1][0] if len(scored) > 1 else -1.0
        by_index = {row[0]: row[1:] for row in signal_rows}
        best_audio, best_filename, best_action, best_caption = by_index[best_index]
        audio_top = max(signal_rows, key=lambda row: row[1])[0]
        filename_rows = [row for row in signal_rows if row[2] is not None]
        filename_top = max(filename_rows, key=lambda row: row[2])[0] if filename_rows else None
        filename_order = sorted((row[2] for row in filename_rows), reverse=True)
        filename_margin = (filename_order[0] - filename_order[1]
                           if len(filename_order) > 1 else 1.0)
        action_rows = [row for row in signal_rows if row[3] is not None]
        action_top = max(action_rows, key=lambda row: row[3])[0] if action_rows else None
        caption_rows = [row for row in signal_rows if row[4] is not None]
        caption_top = max(caption_rows, key=lambda row: row[4])[0] if caption_rows else None

        depth = int(candidates[best_index].get("depth", 0))
        # Target roughly 80% precision: two agreeing signals are enough, while a
        # single source must be clearly strong. Deeper levels remain stricter.
        if (filename_top is not None and audio_top != filename_top
                and max(row[2] for row in filename_rows) >= 0.60
                and max(row[1] for row in signal_rows) >= 0.30):
            return None
        tops = [audio_top, filename_top, action_top, caption_top]
        agreement = sum(value == best_index for value in tops if value is not None)
        margin = best_score - second
        if depth == 0:
            accepted = (
                best_score >= 0.29 and margin >= 0.012 and
                (agreement >= 2 or best_audio >= 0.28
                 or (filename_top == best_index and (best_filename or -1) >= 0.36
                     and filename_margin >= 0.015)
                 or (best_caption or -1) >= 0.56)
            )
        else:
            accepted = (
                best_score >= 0.32 and margin >= 0.020 and
                (agreement >= 2 or best_audio >= 0.34
                 or (filename_top == best_index and (best_filename or -1) >= 0.43
                     and filename_margin >= 0.025)
                 or (best_caption or -1) >= 0.62)
            )
        if not accepted:
            return None
        confidence = float(np.clip((best_score + 1.0) / 2.0, 0.0, 1.0))
        return best_index, confidence

    chosen = choose(by_parent.get(None, []))
    if chosen is None:
        return []
    path: list[tuple[int, float]] = []
    current_index, current_score = chosen
    while True:
        current = candidates[current_index]
        path.append((current["id"], current_score))
        children = by_parent.get(current["id"], [])
        if not children:
            break
        attribute_children = [index for index in children
                              if candidates[index]["label"] in ATTRIBUTE_DIMENSIONS]
        if attribute_children:
            results = [path[-1]]
            for attribute_index in attribute_children:
                branch = choose(by_parent.get(candidates[attribute_index]["id"], []))
                if branch is None:
                    continue
                child_index, child_score = branch
                while by_parent.get(candidates[child_index]["id"]):
                    deeper = choose(by_parent[candidates[child_index]["id"]])
                    if deeper is None:
                        break
                    child_index, child_score = deeper
                results.append((candidates[child_index]["id"], child_score))
            return results
        next_choice = choose(children)
        if next_choice is None:
            break
        current_index, current_score = next_choice
    return [path[-1]]


def _recommended_prototypes(db: Database, embedder: Embedder) -> dict[int, np.ndarray]:
    with db.connect() as conn:
        rows = conn.execute(
            """WITH RECURSIVE ancestors(leaf_id,id,parent_id) AS (
                SELECT id,id,parent_id FROM tag_node
                UNION ALL
                SELECT ancestors.leaf_id,parent.id,parent.parent_id
                FROM ancestors JOIN tag_node parent ON parent.id=ancestors.parent_id
            )
            SELECT ancestors.id node_id,e.dimension,e.vector
            FROM asset_tag_node atn JOIN tag_node n ON n.id=atn.node_id
            JOIN tag_scheme s ON s.id=n.scheme_id
            JOIN ancestors ON ancestors.leaf_id=n.id
            JOIN embedding e ON e.asset_id=atn.asset_id AND e.provider=? AND e.model=?
            WHERE s.code='recommended' AND atn.assigned_by='user'""",
            (embedder.provider, embedder.model_name),
        ).fetchall()
    grouped: dict[int, list[np.ndarray]] = {}
    for row in rows:
        grouped.setdefault(row["node_id"], []).append(
            np.frombuffer(row["vector"], dtype="<f4", count=row["dimension"])
        )
    return {node_id: normalize(np.mean(vectors, axis=0))
            for node_id, vectors in grouped.items()}


def auto_tag_effects(db: Database, cache_dir: Path, embedder: Embedder,
                     limit: int | None = None,
                     progress: Progress | None = None,
                     text_embedder: TextEmbedder | None = None,
                     recommended_classifier=None,
                     cloud_fallback_local: bool = True) -> dict:
    ensure_ucs(db, cache_dir)
    with db.connect() as conn:
        sql = """SELECT a.*,r.path root_path,e.vector,e.dimension,
                 action.actions_json,caption.caption AS audio_caption,
                 EXISTS(SELECT 1 FROM asset_tag_node manual
                   JOIN tag_node mn ON mn.id=manual.node_id
                   JOIN tag_scheme ms ON ms.id=mn.scheme_id
                   WHERE manual.asset_id=a.id AND manual.assigned_by='user'
                   AND ms.code='ucs') AS manual_ucs
                 ,EXISTS(SELECT 1 FROM asset_tag_node manual
                   JOIN tag_node mn ON mn.id=manual.node_id
                   JOIN tag_scheme ms ON ms.id=mn.scheme_id
                   WHERE manual.asset_id=a.id AND manual.assigned_by='user'
                   AND ms.code='recommended') AS manual_recommended
                 FROM asset a JOIN library_root r ON r.id=a.root_id
                 LEFT JOIN embedding e ON e.asset_id=a.id AND e.provider=? AND e.model=?
                 LEFT JOIN effect_action_analysis action ON action.asset_id=a.id
                    AND action.source_modified_ns=a.modified_ns AND action.status='ready'
                 LEFT JOIN effect_caption_analysis caption ON caption.asset_id=a.id
                   AND caption.source_modified_ns=a.modified_ns AND caption.status='ready'
                 WHERE a.missing=0 AND a.media_type='effect'
                 AND a.status<>'duplicate' ORDER BY a.id"""
        params: list[object] = [embedder.provider, embedder.model_name]
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        assets = [dict(row) for row in conn.execute(sql, params)]
        ucs_rows = conn.execute(
            """SELECT n.id,n.code FROM tag_node n JOIN tag_scheme s ON s.id=n.scheme_id
            WHERE s.code='ucs' AND n.code IS NOT NULL"""
        ).fetchall()
        state_rows = conn.execute(
            "SELECT asset_id,scheme_code FROM asset_tag_analysis_state"
        ).fetchall()
    completed_by_asset: dict[int, set[str]] = {}
    for row in state_rows:
        completed_by_asset.setdefault(int(row["asset_id"]), set()).add(row["scheme_code"])
    nodes_by_code = {row["code"]: row["id"] for row in ucs_rows}
    csv_path = cache_dir / f"ucs_v{UCS_VERSION}.csv"
    ucs_rows = list(csv.DictReader(csv_path.read_text("utf-8-sig").splitlines()))
    required_schemes = {"ucs", "recommended", "instrument_material"}
    semantic_models = _prepare_semantic_models(db, embedder, progress)
    recommended_text_model = (
        prepare_recommended_text_vectors(db, text_embedder, progress)
        if text_embedder is not None else None
    )
    instrument_material_text_model = (
        prepare_scheme_text_vectors(db, text_embedder, "instrument_material", progress)
        if text_embedder is not None else None
    )
    recommended_prototypes = _recommended_prototypes(db, embedder)
    result = {"queued": len(assets), "completed": 0, "skipped": 0, "tagged": 0,
              "cloud_failed": 0, "current": ""}
    recommended_futures: dict[int, Future] = {}
    ucs_futures: dict[int, Future] = {}
    cloud_executor = None
    recommended_model = semantic_models.get("recommended")
    ucs_model = semantic_models.get("ucs")
    resolved_ucs: dict[int, tuple[int, str] | None] = {}
    for asset in assets:
        resolved = None
        if (not asset["manual_ucs"]
                and "ucs" not in completed_by_asset.get(asset["id"], set())):
            node_id = _ucs_metadata_match(asset, ucs_rows, nodes_by_code)
            if node_id:
                resolved = (node_id, "正式 UCS 元数据")
            else:
                node_id = _ucs_filename_match(asset, nodes_by_code)
                if node_id:
                    resolved = (node_id, "标准 UCS 文件名")
                else:
                    node_id = _ucs_descriptive_match(asset, ucs_rows, nodes_by_code)
                    if node_id:
                        resolved = (node_id, "UCS 官方名称或同义词")
        resolved_ucs[asset["id"]] = resolved
    if recommended_classifier is not None and (recommended_model or ucs_model):
        workers = recommended_classifier.max_concurrency
        print(f"[云端标签] 推荐标签与UCS共享并发上限：{workers}", flush=True)
        cloud_executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="aliyun-tags"
        )
        for asset in assets:
            completed = completed_by_asset.get(asset["id"], set())
            if (recommended_model and not asset["manual_recommended"]
                    and "recommended" not in completed
                    and asset.get("vector") is not None):
                recommended_futures[asset["id"]] = cloud_executor.submit(
                    recommended_classifier.classify, asset, recommended_model[0]
                )
            if (ucs_model and not asset["manual_ucs"] and "ucs" not in completed
                    and resolved_ucs[asset["id"]] is None):
                ucs_futures[asset["id"]] = cloud_executor.submit(
                    recommended_classifier.classify_ucs, asset, ucs_model[0]
                )
    for asset in assets:
        result["current"] = asset["filename"]
        matches: list[tuple[int, float, str]] = []
        completed_schemes = completed_by_asset.get(asset["id"], set())
        processed_schemes: set[str] = set()
        if required_schemes.issubset(completed_schemes):
            result["skipped"] += 1
        ucs_id = None
        ucs_cloud_handled = False
        deterministic_ucs = resolved_ucs[asset["id"]]
        if deterministic_ucs:
            ucs_id, evidence = deterministic_ucs
            matches.append((ucs_id, 1.0, evidence))
            processed_schemes.add("ucs")
        elif asset["id"] in ucs_futures:
            try:
                cloud_ucs = ucs_futures[asset["id"]].result()
            except Exception as exc:
                result["cloud_failed"] += 1
                print(
                    f"[UCS][云端失败] {asset['filename']}：{exc}；"
                    f"{'回退本地' if cloud_fallback_local else '不回退'}",
                    flush=True,
                )
                ucs_cloud_handled = not cloud_fallback_local
            else:
                ucs_cloud_handled = True
                processed_schemes.add("ucs")
                if cloud_ucs:
                    ucs_id = int(cloud_ucs["node_id"])
                    matches.append((
                        ucs_id, float(cloud_ucs["confidence"]),
                        f"阿里云 {recommended_classifier.model} 分层UCS："
                        f"{cloud_ucs.get('evidence') or '音频、文件名与元数据综合判断'}",
                    ))
        audio_vector = None
        if asset.get("vector") is not None:
            audio_vector = np.frombuffer(asset["vector"], dtype="<f4", count=asset["dimension"])
        if audio_vector is not None:
            try:
                action_context = json.loads(asset.get("actions_json") or "[]")
            except json.JSONDecodeError:
                action_context = []
            try:
                metadata = json.loads(asset.get("metadata_json") or "{}")
            except json.JSONDecodeError:
                metadata = {}
            metadata_text = "；".join(
                f"{key}：{value}" for key, value in sorted(metadata.items())
            )
            for scheme_code, policy in SEMANTIC_POLICY.items():
                if scheme_code in completed_schemes:
                    continue
                if scheme_code == "ucs" and (
                    ucs_id or asset["manual_ucs"] or ucs_cloud_handled
                ):
                    continue
                if scheme_code == "recommended":
                    if asset["manual_recommended"]:
                        continue
                    model = semantic_models.get(scheme_code)
                    if recommended_classifier is not None and model:
                        try:
                            future = recommended_futures.get(asset["id"])
                            cloud_matches = (future.result() if future is not None
                                             else recommended_classifier.classify(asset, model[0]))
                        except Exception as exc:
                            result["cloud_failed"] += 1
                            print(
                                f"[推荐标签][云端失败] {asset['filename']}：{exc}；"
                                f"{'回退本地' if cloud_fallback_local else '不回退'}",
                                flush=True,
                            )
                            if progress:
                                progress({
                                    **result,
                                    "message": f"云端推荐标签失败，已回退本地：{exc}",
                                })
                            if not cloud_fallback_local:
                                continue
                        else:
                            for item in cloud_matches:
                                evidence = item.get("evidence") or "音频内容受限分类"
                                matches.append((
                                    int(item["node_id"]), float(item["confidence"]),
                                    f"阿里云 {recommended_classifier.model}：{evidence}",
                                ))
                            # A successful cloud abstention is intentional and
                            # must not be replaced by a less reliable local tag.
                            processed_schemes.add("recommended")
                            continue
                    for node_id, confidence in (
                        _recommended_matches(
                            audio_vector, model, recommended_prototypes,
                            recommended_text_model, asset["filename"], metadata_text,
                            action_context, text_embedder,
                            asset.get("audio_caption") or "",
                        )
                        if model else []
                    ):
                        matches.append((node_id, confidence,
                                        "音频描述、BGE、CLAP 与 AST 分层交叉验证"))
                    if model:
                        processed_schemes.add("recommended")
                    continue
                model = semantic_models.get(scheme_code)
                if scheme_code == "instrument_material" and model:
                    matches.extend(_hierarchical_instrument_material_matches(
                        audio_vector, model, instrument_material_text_model,
                        asset["filename"], metadata_text, action_context,
                        asset.get("audio_caption") or "", text_embedder,
                    ))
                    processed_schemes.add("instrument_material")
                    continue
                match = _semantic_match(audio_vector, model, policy) if model else None
                if match:
                    matches.append((match[0], match[1], "CLAP 动态候选分类"))
                if model:
                    processed_schemes.add(scheme_code)
        with db.connect() as conn:
            for scheme_code in processed_schemes:
                conn.execute(
                    """DELETE FROM asset_tag_node WHERE asset_id=? AND assigned_by<>'user'
                    AND node_id IN (SELECT n.id FROM tag_node n JOIN tag_scheme s
                    ON s.id=n.scheme_id WHERE s.code=?)""",
                    (asset["id"], scheme_code),
                )
            conn.executemany(
                """INSERT OR IGNORE INTO asset_tag_node
                (asset_id,node_id,assigned_by,confidence,evidence,created_at)
                VALUES (?,?,?,?,?,?)""",
                [(asset["id"], node_id, AUTO_SOURCE, confidence, evidence, time.time())
                 for node_id, confidence, evidence in matches],
            )
            now = time.time()
            for scheme_code in processed_schemes:
                count = conn.execute(
                    """SELECT COUNT(*) FROM asset_tag_node atn JOIN tag_node n ON n.id=atn.node_id
                    JOIN tag_scheme s ON s.id=n.scheme_id
                    WHERE atn.asset_id=? AND s.code=? AND atn.assigned_by<>'user'""",
                    (asset["id"], scheme_code),
                ).fetchone()[0]
                conn.execute(
                    """INSERT OR IGNORE INTO asset_tag_analysis_state
                    (asset_id,scheme_code,engine,result_count,completed_at)
                    VALUES (?,?,?,?,?)""",
                    (asset["id"], scheme_code,
                     "aliyun" if recommended_classifier is not None else "local",
                     count, now),
                )
        result["completed"] += 1
        result["tagged"] += len(matches)
        if progress:
            progress(result.copy())
    if cloud_executor is not None:
        cloud_executor.shutdown(wait=True)
    result["current"] = ""
    if result["skipped"]:
        print(
            f"[音效标签] 已计算并跳过 {result['skipped']}/{result['queued']} 条素材",
            flush=True,
        )
    if progress:
        progress(result.copy())
    return result
