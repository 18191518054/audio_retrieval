from __future__ import annotations

import base64
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

from .db import Database
from .media import decode_excerpt, excerpt_starts


class CloudRecommendedError(RuntimeError):
    pass


class AliyunRecommendedClassifier:
    provider = "aliyun"

    def __init__(self, db: Database, api_key: str, base_url: str, model: str,
                 ffmpeg: str, timeout_seconds: int = 120,
                 min_confidence: float = 0.78, max_concurrency: int = 2,
                 ucs_category_confidence: float = 0.85,
                 ucs_subcategory_confidence: float = 0.82):
        if not api_key:
            raise ValueError("云端推荐标签已启用，但未配置 DASHSCOPE_API_KEY")
        if not base_url:
            raise ValueError("云端推荐标签已启用，但未配置 DASHSCOPE_BASE_URL")
        self.db = db
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.ffmpeg = ffmpeg
        self.timeout_seconds = timeout_seconds
        self.min_confidence = min(max(float(min_confidence), 0.0), 1.0)
        self.max_concurrency = min(8, max(1, int(max_concurrency)))
        self.ucs_category_confidence = min(max(float(ucs_category_confidence), 0.0), 1.0)
        self.ucs_subcategory_confidence = min(max(float(ucs_subcategory_confidence), 0.0), 1.0)

    @staticmethod
    def _http_error_detail(exc: urllib.error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""
        request_id = ""
        if exc.headers:
            request_id = (exc.headers.get("x-request-id")
                          or exc.headers.get("x-dashscope-request-id") or "")
        parts = [f"HTTP {exc.code}"]
        if exc.reason:
            parts.append(str(exc.reason))
        if request_id:
            parts.append(f"Request ID={request_id}")
        if body:
            parts.append(f"响应体={body[:4000]}")
        return "；".join(parts)

    @staticmethod
    def _taxonomy_hash(candidates: list[dict]) -> str:
        content = json.dumps([
            {
                "id": int(item["id"]),
                "code": item.get("code"),
                "parent_id": item.get("parent_id"),
                "label": item.get("label"),
                "path": item["path"],
                "description": item.get("description", ""),
                "examples": item.get("examples_json", "[]"),
            }
            for item in candidates
        ], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _cached(self, asset: dict, taxonomy_hash: str,
                scheme_code: str = "recommended") -> object | None:
        with self.db.connect() as conn:
            row = conn.execute(
                """SELECT result_json,source_modified_ns,taxonomy_hash
                FROM cloud_tag_cache WHERE asset_id=? AND scheme_code=?
                AND provider=? AND model=?""",
                (asset["id"], scheme_code, self.provider, self.model),
            ).fetchone()
        if (not row or row["source_modified_ns"] != asset["modified_ns"]
                or row["taxonomy_hash"] != taxonomy_hash):
            return None
        try:
            return json.loads(row["result_json"])
        except json.JSONDecodeError:
            return None

    def _store(self, asset: dict, taxonomy_hash: str, result: object,
               scheme_code: str = "recommended") -> None:
        with self.db.connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO cloud_tag_cache
                (asset_id,scheme_code,provider,model,source_modified_ns,
                 taxonomy_hash,result_json,created_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (asset["id"], scheme_code, self.provider, self.model, asset["modified_ns"],
                 taxonomy_hash, json.dumps(result, ensure_ascii=False), time.time()),
            )

    @staticmethod
    def _asset_context(asset: dict) -> str:
        try:
            metadata = json.loads(asset.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        useful = []
        accepted = ("title", "description", "comment", "keyword", "subject",
                    "标题", "描述", "说明", "关键词", "注释")
        for key, value in metadata.items():
            if any(token in str(key).casefold() for token in accepted) and value:
                useful.append(f"{key}={value}")
        return f"文件名={asset['filename']}\n元数据=" + ("；".join(useful) or "无")

    def _audio_data_uri(self, asset: dict) -> str:
        path = Path(asset["path"])
        clips = [decode_excerpt(path, self.ffmpeg, start, 10, 16000)
                 for start in excerpt_starts(asset.get("duration_ms"), 10)]
        silence = np.zeros(4000, dtype=np.float32)
        joined: list[np.ndarray] = []
        for index, clip in enumerate(clips):
            if index:
                joined.append(silence)
            joined.append(np.asarray(clip, dtype=np.float32))
        audio = np.concatenate(joined) if joined else np.empty(0, dtype=np.float32)
        if not audio.size:
            raise CloudRecommendedError("云端分类音频采样为空")
        pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(pcm.tobytes())
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:audio/wav;base64,{encoded}"

    @staticmethod
    def _prompt(candidates: list[dict]) -> str:
        lines = []
        for item in candidates:
            examples = ""
            try:
                values = json.loads(item.get("examples_json") or "[]")
                if values:
                    examples = "；示例=" + "、".join(map(str, values[:4]))
            except json.JSONDecodeError:
                pass
            description = ("；说明=" + item["description"]
                           if item.get("description") else "")
            lines.append(f"{int(item['id'])} | {item['path']}{description}{examples}")
        taxonomy = "\n".join(lines)
        return (
            "你是影视音效素材分类器。只根据实际听到的音频内容判断，不参考文件名。"
            "只能从下面给出的推荐标签节点中选择，禁止创造标签。沿树选择证据可靠的最深节点；"
            "可选择最多3个互不重复的有效分支。若无法可靠判断，必须abstain=true且labels为空。"
            "不要因为声音相似就臆测具体物体、材质或场景。输出严格JSON："
            '{"abstain":false,"labels":[{"node_id":123,"confidence":0.86,'
            '"evidence":"听到的声学事实"}]}。confidence范围0到1。\n\n标签节点：\n'
            + taxonomy
        )

    def classify(self, asset: dict, candidates: list[dict]) -> list[dict]:
        taxonomy_hash = self._taxonomy_hash(candidates)
        cached = self._cached(asset, taxonomy_hash)
        if cached is not None:
            print(
                f"[推荐标签][云端缓存] {asset['filename']}："
                f"复用 {self.model} 结果（{len(cached)} 个标签）",
                flush=True,
            )
            return cached
        print(
            f"[推荐标签][云端请求] {asset['filename']} -> aliyun/{self.model}",
            flush=True,
        )
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {
                        "data": self._audio_data_uri(asset), "format": "wav",
                    }},
                    {"type": "text", "text": self._prompt(candidates)},
                ],
            }],
            "modalities": ["text"],
            "response_format": {"type": "json_object"},
            "reasoning_effort": "none",
            "stream": False,
            "temperature": 0,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            print(
                f"[推荐标签][云端原始返回] {asset['filename']}：{content}",
                flush=True,
            )
            parsed = json.loads(content)
        except urllib.error.HTTPError as exc:
            detail = self._http_error_detail(exc)
            print(f"[推荐标签][云端HTTP错误] {asset['filename']}：{detail}", flush=True)
            raise CloudRecommendedError(f"阿里云推荐标签调用失败：{detail}") from exc
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError,
                UnicodeDecodeError, json.JSONDecodeError) as exc:
            detail = str(getattr(exc, "reason", exc)) or type(exc).__name__
            print(f"[推荐标签][云端错误] {asset['filename']}：{detail}", flush=True)
            raise CloudRecommendedError(f"阿里云推荐标签调用失败：{detail}") from exc
        valid_ids = {int(item["id"]) for item in candidates}
        result = []
        if not parsed.get("abstain"):
            for item in parsed.get("labels", [])[:3]:
                try:
                    node_id = int(item["node_id"])
                    confidence = float(item["confidence"])
                except (KeyError, TypeError, ValueError):
                    continue
                if node_id not in valid_ids or confidence < self.min_confidence:
                    continue
                result.append({
                    "node_id": node_id,
                    "confidence": min(confidence, 1.0),
                    "evidence": str(item.get("evidence", ""))[:500],
                })
        self._store(asset, taxonomy_hash, result)
        print(
            f"[推荐标签][云端完成] {asset['filename']}：{len(result)} 个可靠标签",
            flush=True,
        )
        return result

    def _request_json(self, asset: dict, prompt: str, log_stage: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {
                        "data": self._audio_data_uri(asset), "format": "wav",
                    }},
                    {"type": "text", "text": prompt},
                ],
            }],
            "modalities": ["text"],
            "response_format": {"type": "json_object"},
            "reasoning_effort": "none",
            "stream": False,
            "temperature": 0,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            print(f"[UCS][云端原始返回][{log_stage}] {asset['filename']}：{content}",
                  flush=True)
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise TypeError("返回值不是JSON对象")
            return parsed
        except urllib.error.HTTPError as exc:
            detail = self._http_error_detail(exc)
            print(f"[UCS][云端HTTP错误][{log_stage}] {asset['filename']}：{detail}",
                  flush=True)
            raise CloudRecommendedError(
                f"阿里云UCS {log_stage}调用失败：{detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError,
                TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            detail = str(getattr(exc, "reason", exc)) or type(exc).__name__
            print(f"[UCS][云端错误][{log_stage}] {asset['filename']}：{detail}",
                  flush=True)
            raise CloudRecommendedError(
                f"阿里云UCS {log_stage}调用失败：{detail}"
            ) from exc

    def classify_ucs(self, asset: dict, candidates: list[dict]) -> dict | None:
        """Classify UCS hierarchically and return one validated leaf node."""
        taxonomy_hash = self._taxonomy_hash(candidates)
        cached = self._cached(asset, taxonomy_hash, "ucs")
        if cached is not None:
            print(f"[UCS][云端缓存] {asset['filename']}：复用 {self.model} 结果",
                  flush=True)
            return cached if isinstance(cached, dict) and cached else None
        groups: dict[str, list[dict]] = {}
        for item in candidates:
            category = str(item.get("path", "")).split(" > ", 1)[0].strip()
            if category:
                groups.setdefault(category, []).append(item)
        if not groups:
            return None
        category_lines = [
            f"{name} | 子类示例=" + "、".join(child["label"] for child in children[:12])
            for name, children in groups.items()
        ]
        context = self._asset_context(asset)
        print(f"[UCS][云端请求][一级] {asset['filename']} -> aliyun/{self.model}",
              flush=True)
        category_result = self._request_json(asset, (
            "你是UCS音效分类专家。综合声音内容、文件名和业务元数据，从给定的UCS一级Category中"
            "选择一个；不得创造类别。无法可靠判断时abstain=true。输出严格JSON："
            '{"abstain":false,"category":"ANIMALS","confidence":0.90,'
            '"evidence":"文件名、元数据及听觉证据"}。\n素材信息：\n' + context
            + "\n候选Category：\n" + "\n".join(category_lines)
        ), "一级Category")
        category = str(category_result.get("category", "")).strip()
        try:
            category_confidence = float(category_result.get("confidence", 0))
        except (TypeError, ValueError):
            category_confidence = 0.0
        if (category_result.get("abstain") or category not in groups
                or category_confidence < self.ucs_category_confidence):
            self._store(asset, taxonomy_hash, {}, "ucs")
            print(f"[UCS][云端完成] {asset['filename']}：一级类别不可靠，保持为空",
                  flush=True)
            return None
        children = groups[category]
        child_lines = []
        for item in children:
            examples = ""
            try:
                values = json.loads(item.get("examples_json") or "[]")
                if values:
                    examples = "；同义词=" + "、".join(map(str, values[:8]))
            except json.JSONDecodeError:
                pass
            child_lines.append(
                f"{int(item['id'])} | CatID={item.get('code') or ''} | {item['label']}"
                f"；说明={item.get('description') or ''}{examples}"
            )
        print(f"[UCS][云端请求][二级] {asset['filename']}：{category}", flush=True)
        child_result = self._request_json(asset, (
            f"一级UCS Category已经可靠确定为 {category}。只从下面直属SubCategory中选择一个节点，"
            "不得跨类别或创造节点。综合声音、文件名和元数据；无法可靠判断时abstain=true。"
            "输出严格JSON："
            '{"abstain":false,"node_id":123,"confidence":0.86,'
            '"evidence":"文件名、元数据及听觉证据"}。\n素材信息：\n' + context
            + "\n候选SubCategory：\n" + "\n".join(child_lines)
        ), "二级SubCategory")
        try:
            node_id = int(child_result.get("node_id"))
            confidence = float(child_result.get("confidence", 0))
        except (TypeError, ValueError):
            node_id, confidence = -1, 0.0
        valid = {int(item["id"]): item for item in children}
        if (child_result.get("abstain") or node_id not in valid
                or confidence < self.ucs_subcategory_confidence):
            self._store(asset, taxonomy_hash, {}, "ucs")
            print(f"[UCS][云端完成] {asset['filename']}：二级类别不可靠，保持为空",
                  flush=True)
            return None
        result = {
            "node_id": node_id,
            "confidence": min(confidence, 1.0),
            "evidence": str(child_result.get("evidence", ""))[:500],
            "category": category,
        }
        self._store(asset, taxonomy_hash, result, "ucs")
        print(f"[UCS][云端完成] {asset['filename']}：{valid[node_id]['path']}", flush=True)
        return result
