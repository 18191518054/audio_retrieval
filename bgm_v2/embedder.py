from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .media import decode_excerpt, duration_ms, excerpt_starts


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


class Embedder(ABC):
    provider: str
    model_name: str

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def embed_audio(self, path: Path) -> np.ndarray: ...

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.stack([self.embed_text(text) for text in texts])


class DiagnosticEmbedder(Embedder):
    """Deterministic low-cost backend for tests only; it is not semantic search."""

    provider = "diagnostic"
    model_name = "hash-v1"
    dimension = 64

    def _hash(self, payload: bytes) -> np.ndarray:
        chunks = bytearray()
        seed = payload
        while len(chunks) < self.dimension * 4:
            seed = hashlib.sha256(seed).digest()
            chunks.extend(seed)
        raw = np.frombuffer(bytes(chunks[: self.dimension * 4]), dtype="<u4").astype(np.float32)
        return normalize(raw / np.float32(2**32) - 0.5)

    def embed_text(self, text: str) -> np.ndarray:
        return self._hash(text.strip().encode("utf-8"))

    def embed_audio(self, path: Path) -> np.ndarray:
        with path.open("rb") as stream:
            sample = stream.read(1024 * 1024)
        return self._hash(sample)


class ClapEmbedder(Embedder):
    provider = "clap"

    def __init__(self, model_name: str, ffmpeg: str, ffprobe: str, device: str = "auto"):
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError(
                "CLAP 依赖未安装，请执行：python -m pip install -r requirements-clap.txt"
            ) from exc
        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name
        self.ffmpeg = ffmpeg
        self.ffprobe = ffprobe
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model = ClapModel.from_pretrained(model_name)
        try:
            self.model = self.model.to(device)
        except RuntimeError as exc:
            if device != "cuda" or "out of memory" not in str(exc).casefold():
                raise
            self.device = "cpu"
            self.model = self.model.to("cpu")
            torch.cuda.empty_cache()
        self.model.eval()

    def _run_model(self, method_name: str, inputs: dict):
        try:
            with self._torch.inference_mode():
                return getattr(self.model, method_name)(**inputs)
        except RuntimeError as exc:
            if self.device != "cuda" or "out of memory" not in str(exc).casefold():
                raise
            self.model = self.model.to("cpu")
            self.device = "cpu"
            self._torch.cuda.empty_cache()
            cpu_inputs = {key: value.to("cpu") for key, value in inputs.items()}
            with self._torch.inference_mode():
                return getattr(self.model, method_name)(**cpu_inputs)

    def embed_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[expand_chinese_query(text)], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self._run_model("get_text_features", inputs)
        features = output.pooler_output if hasattr(output, "pooler_output") else output
        value = features[0].detach().cpu().numpy()
        return normalize(value)

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        values = []
        for start in range(0, len(texts), batch_size):
            batch = [expand_chinese_query(text) for text in texts[start:start + batch_size]]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self._run_model("get_text_features", inputs)
            features = output.pooler_output if hasattr(output, "pooler_output") else output
            current = features.detach().cpu().numpy()
            values.extend(normalize(value) for value in current)
        return np.stack(values)

    def embed_audio(self, path: Path) -> np.ndarray:
        values = []
        total_ms = duration_ms(path, self.ffprobe)
        for start in excerpt_starts(total_ms):
            excerpt = decode_excerpt(path, self.ffmpeg, start_seconds=start)
            # Transformers 5 renamed `audios` to `audio`. Keep a fallback for 4.x.
            try:
                inputs = self.processor(
                    audio=excerpt, sampling_rate=48000, return_tensors="pt"
                )
            except (TypeError, ValueError) as current_api_error:
                try:
                    inputs = self.processor(
                        audios=excerpt, sampling_rate=48000, return_tensors="pt"
                    )
                except (TypeError, ValueError):
                    raise current_api_error
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            output = self._run_model("get_audio_features", inputs)
            features = output.pooler_output if hasattr(output, "pooler_output") else output
            value = features[0].detach().cpu().numpy()
            values.append(normalize(value))
        return normalize(np.mean(values, axis=0))


def create_embedder(provider: str, model_name: str, ffmpeg: str, ffprobe: str = "ffprobe") -> Embedder:
    if provider == "clap":
        return ClapEmbedder(model_name, ffmpeg, ffprobe)
    if provider == "diagnostic":
        return DiagnosticEmbedder()
    raise ValueError(f"未知向量提供器：{provider}")


QUERY_GLOSSARY = {
    "悲伤": "sad melancholic", "伤感": "sad emotional", "克制": "restrained subtle",
    "温暖": "warm comforting", "紧张": "tense suspenseful", "悬疑": "mysterious suspense",
    "神秘": "mysterious", "庄严": "solemn majestic", "轻松": "lighthearted relaxed",
    "浪漫": "romantic", "激昂": "uplifting powerful", "史诗": "epic cinematic",
    "回忆": "memory flashback", "离别": "farewell", "追逐": "chase action",
    "调查": "investigation", "战争": "war", "独白": "background music for monologue",
    "转场": "transition", "高潮": "climax", "片尾": "ending credits",
    "钢琴": "piano", "弦乐": "strings", "木管": "woodwind", "合成器": "synthesizer",
    "打击乐": "percussion", "管弦": "orchestral", "电子": "electronic",
    "爵士": "jazz", "中国风": "traditional Chinese style", "无人声": "instrumental no vocals",
    "人声": "vocals", "低能量": "low energy", "慢节奏": "slow tempo",
}


def expand_chinese_query(text: str) -> str:
    additions = [english for chinese, english in QUERY_GLOSSARY.items() if chinese in text]
    return f"{text}; {'; '.join(additions)}" if additions else text
