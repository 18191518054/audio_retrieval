from __future__ import annotations

import numpy as np

from .embedder import Embedder


TAXONOMY: dict[str, list[tuple[str, str, str]]] = {
    "instrument": [
        ("piano", "钢琴", "piano music"), ("strings", "弦乐", "orchestral strings"),
        ("guitar", "吉他", "guitar music"), ("woodwind", "木管", "woodwind instruments"),
        ("synth", "合成器", "synthesizer music"), ("percussion", "打击乐", "percussion and drums"),
        ("orchestra", "管弦乐", "full orchestra"),
    ],
    "mood": [
        ("sad", "悲伤", "sad melancholic music"), ("warm", "温暖", "warm comforting music"),
        ("tense", "紧张", "tense suspenseful music"), ("mysterious", "神秘", "mysterious music"),
        ("solemn", "庄严", "solemn majestic music"), ("relaxed", "轻松", "relaxed lighthearted music"),
        ("romantic", "浪漫", "romantic emotional music"), ("epic", "史诗", "epic cinematic music"),
    ],
    "rhythm": [
        ("slow", "慢速", "slow tempo music"), ("medium", "中速", "medium tempo music"),
        ("fast", "快速", "fast tempo energetic music"),
    ],
    "scene": [
        ("monologue", "独白铺底", "subtle background music for a monologue"),
        ("memory", "回忆", "music for a memory or flashback scene"),
        ("farewell", "离别", "music for a farewell scene"),
        ("chase", "追逐", "music for a chase scene"),
        ("investigation", "调查", "music for an investigation scene"),
        ("transition", "转场", "short transition music"),
        ("climax", "高潮", "music for a dramatic climax"),
        ("ending", "片尾", "ending credits music"),
    ],
}


def build_label_vectors(embedder: Embedder) -> dict[str, list[tuple[str, str, np.ndarray]]]:
    return {
        namespace: [(code, label, embedder.embed_text(prompt)) for code, label, prompt in entries]
        for namespace, entries in TAXONOMY.items()
    }


def classify(vector: np.ndarray, label_vectors, keep: int = 2) -> list[dict]:
    results = []
    for namespace, entries in label_vectors.items():
        raw = np.array([float(vector @ label_vector) for _, _, label_vector in entries])
        shifted = raw - raw.max()
        probabilities = np.exp(shifted * 10)
        probabilities /= probabilities.sum()
        order = np.argsort(-raw)[:keep]
        for rank, index in enumerate(order, 1):
            code, label, _ = entries[int(index)]
            results.append({
                "namespace": namespace, "code": code, "label": label,
                "score": float(probabilities[index]), "rank_no": rank,
            })
    return results

