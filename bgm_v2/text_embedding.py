from __future__ import annotations

import hashlib
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

from .db import Database
from .embedder import normalize


class TextEmbedder(ABC):
    model_name: str

    @abstractmethod
    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


class BgeTextEmbedder(TextEmbedder):
    def __init__(self, model_name: str):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("缺少 BGE 文本向量依赖，请重新运行 Conda 环境安装脚本") from exc
        self.torch = torch
        self.model_name = model_name
        self.device = "cpu"  # Keep GPU memory available for CLAP on resource-limited PCs.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, use_safetensors=True, local_files_only=True
            ).to(self.device)
        except OSError as exc:
            raise RuntimeError(
                "未找到本地 BGE 中文模型，请先运行：下载本地模型.ps1"
            ) from exc
        self.model.eval()
        self._lock = threading.Lock()

    def _embed(self, texts: list[str], batch_size: int) -> np.ndarray:
        values = []
        with self._lock, self.torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                inputs = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=512,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                output = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(output.dtype)
                pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                values.extend(normalize(value) for value in pooled.cpu().numpy())
        return np.stack(values) if values else np.empty((0, 0), dtype=np.float32)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return self._embed(texts, batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        instruction = "为这个句子生成表示以用于检索相关文章："
        return self._embed([instruction + text.strip()], 1)[0]


def asset_documents(db: Database, asset_ids: list[int] | None = None) -> list[dict]:
    params: list[object] = []
    asset_filter = ""
    if asset_ids:
        placeholders = ",".join("?" for _ in asset_ids)
        asset_filter = f" AND a.id IN ({placeholders})"
        params.extend(asset_ids)
    with db.connect() as conn:
        assets = [dict(row) for row in conn.execute(
            f"""SELECT a.id,a.filename
            FROM asset a WHERE a.missing=0 AND a.status<>'duplicate'{asset_filter}
            ORDER BY a.id""", params,
        )]
    if not assets:
        return []
    ids = [asset["id"] for asset in assets]
    placeholders = ",".join("?" for _ in ids)
    with db.connect() as conn:
        music_tags = conn.execute(
            f"""SELECT asset_id,namespace,label FROM asset_tag
            WHERE asset_id IN ({placeholders}) ORDER BY asset_id,namespace,rank_no""", ids,
        ).fetchall()
        effect_tags = conn.execute(
            f"""SELECT atn.asset_id,s.label scheme_label,n.label
            FROM asset_tag_node atn JOIN tag_node n ON n.id=atn.node_id
            JOIN tag_scheme s ON s.id=n.scheme_id
            WHERE atn.asset_id IN ({placeholders})
              AND NOT EXISTS(SELECT 1 FROM tag_node child WHERE child.parent_id=n.id)
            ORDER BY atn.asset_id,s.id,n.label""", ids,
        ).fetchall()
    tags: dict[int, list[str]] = {asset_id: [] for asset_id in ids}
    for row in music_tags:
        tags[row["asset_id"]].append(f"{row['namespace']}：{row['label']}")
    for row in effect_tags:
        tags[row["asset_id"]].append(f"{row['scheme_label']}：{row['label']}")
    documents = []
    for asset in assets:
        # Search admission is deliberately based only on business-visible
        # filename and tags. Generic metadata (artist, encoder, album, etc.)
        # makes BGE retrieve many acoustically and semantically unrelated files.
        parts = [f"文件名：{asset['filename']}"]
        if tags[asset["id"]]:
            parts.append("标签：" + "；".join(tags[asset["id"]]))
        text = "\n".join(parts)
        documents.append({
            "id": asset["id"], "text": text,
            "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    return documents


def build_text_embeddings(db: Database, embedder: TextEmbedder,
                          asset_ids: list[int] | None = None,
                          progress=None) -> dict:
    documents = asset_documents(db, asset_ids)
    with db.connect() as conn:
        cached = {row["asset_id"]: row["content_hash"] for row in conn.execute(
            "SELECT asset_id,content_hash FROM asset_text_embedding WHERE model=?",
            (embedder.model_name,),
        )}
        conn.execute(
            """DELETE FROM asset_text_embedding WHERE asset_id IN
            (SELECT id FROM asset WHERE missing<>0 OR status='duplicate')"""
        )
    missing = [item for item in documents if cached.get(item["id"]) != item["content_hash"]]
    result = {"total": len(documents), "cached": len(documents) - len(missing),
              "generated": 0, "current": ""}
    if progress:
        progress(result.copy())
    for start in range(0, len(missing), 16):
        batch = missing[start:start + 16]
        vectors = embedder.embed_documents([item["text"] for item in batch], batch_size=16)
        now = time.time()
        with db.connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO asset_text_embedding
                (asset_id,model,content_hash,dimension,vector,created_at)
                VALUES (?,?,?,?,?,?)""",
                [(item["id"], embedder.model_name, item["content_hash"], vector.size,
                  np.asarray(vector, dtype="<f4").tobytes(), now)
                 for item, vector in zip(batch, vectors)],
            )
        result["generated"] += len(batch)
        result["current"] = batch[-1]["id"]
        if progress:
            progress(result.copy())
    result["current"] = ""
    if progress:
        progress(result.copy())
    return result
