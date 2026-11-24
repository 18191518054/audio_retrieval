from __future__ import annotations

import os
import time
from pathlib import Path
import json
from typing import Callable

import numpy as np

from .db import Database
from .media import decode_excerpt, excerpt_starts


Progress = Callable[[dict], None]

INSTRUMENTS = {
    "piano": ("钢琴", ("piano",)),
    "acoustic_guitar": ("木吉他", ("acoustic guitar",)),
    "electric_guitar": ("电吉他", ("electric guitar",)),
    "guitar": ("吉他", ("guitar",)),
    "violin": ("小提琴", ("violin", "fiddle")),
    "cello": ("大提琴", ("cello",)),
    "strings": ("弦乐组", ("string section",)),
    "woodwind": ("木管", ("flute", "saxophone", "clarinet", "oboe", "bassoon")),
    "brass": ("铜管", ("trumpet", "trombone", "french horn", "brass instrument")),
    "drums": ("鼓组", ("drum kit", "drum", "snare drum")),
    "percussion": ("打击乐", ("percussion", "cymbal", "tambourine")),
    "synth": ("合成器", ("synthesizer",)),
    "organ": ("风琴", ("organ",)),
}

MOODS = {
    "happy": ("欢快", ("happy music",)),
    "sad": ("悲伤", ("sad music",)),
    "tense": ("紧张", ("scary music", "suspense music")),
    "relaxed": ("舒缓", ("tender music",)),
    "energetic": ("活力", ("exciting music",)),
    "romantic": ("浪漫", ("romantic music",)),
    "solemn": ("庄严", ("solemn music", "hymn")),
    "dark": ("阴暗", ("scary music",)),
}

STYLES = {
    "classical": ("古典", ("classical music", "opera")),
    "orchestral": ("管弦/电影配乐", ("orchestra", "soundtrack music")),
    "electronic": ("电子", ("electronic music", "electronica", "techno", "house music")),
    "ambient": ("氛围", ("ambient music", "new-age music")),
    "pop": ("流行", ("pop music",)),
    "rock": ("摇滚", ("rock music",)),
    "metal": ("金属", ("heavy metal",)),
    "jazz": ("爵士", ("jazz",)),
    "blues": ("布鲁斯", ("blues",)),
    "folk": ("民谣/民族", ("folk music", "traditional music")),
    "country": ("乡村", ("country",)),
    "hiphop": ("嘻哈", ("hip hop music", "rap")),
    "dance": ("舞曲", ("dance music", "disco")),
    "latin": ("拉丁", ("latin music", "salsa music")),
    "reggae": ("雷鬼", ("reggae",)),
}

EVENTS = {
    "footsteps": ("脚步", ("walk, footsteps", "run")),
    "whoosh": ("呼啸/转场声", ("whoosh, swoosh, swish",)),
    "impact": ("撞击/敲击", ("thump", "whack", "impact", "slam")),
    "plop": ("啵/弹出声", ("plop", "boing")),
    "crunch": ("咔嚓/摩擦", ("crunch", "crack", "crackle", "scrape")),
    "chime": ("风铃/铃声", ("wind chime", "chime", "jingle, tinkle")),
    "speech": ("人声/交谈", ("speech", "conversation", "hubbub")),
    "traffic": ("车辆/街道", ("vehicle", "car", "traffic noise", "outside, urban")),
    "weather": ("天气环境", ("rain", "thunder", "wind")),
    "alarm": ("提示/警报声", ("alarm", "beep", "ding", "ping")),
    "sonar": ("电子脉冲/声呐", ("sonar", "chirp tone")),
    "resonance": ("锣/共鸣声", ("singing bowl", "gong")),
}

NON_MUSIC_HINTS = (
    "speech", "conversation", "footsteps", "vehicle", "engine", "door", "wind",
    "rain", "thunder", "animal", "crowd", "alarm", "typing", "silence", "sound effect",
)

GENERIC_EFFECT_LABELS = {
    "music", "sound effect", "silence", "noise", "inside, small room",
    "inside, large room or hall", "outside, urban or manmade",
    "outside, rural or natural",
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -30, 30)
    return 1.0 / (1.0 + np.exp(-values))


def estimate_tempo(audio: np.ndarray, sample_rate: int = 16000) -> tuple[float | None, float]:
    """Small dependency-free onset autocorrelation estimator for local validation."""
    frame, hop = 1024, 256
    if audio.size < frame * 4:
        return None, 0.0
    usable = 1 + (audio.size - frame) // hop
    shape = (usable, frame)
    strides = (audio.strides[0] * hop, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    energy = np.mean(frames * frames, axis=1)
    onset = np.maximum(0.0, np.diff(np.log1p(1000 * energy)))
    onset -= onset.mean()
    corr = np.correlate(onset, onset, mode="full")[onset.size - 1:]
    min_lag = max(1, int(round(60 * sample_rate / (200 * hop))))
    max_lag = min(corr.size - 1, int(round(60 * sample_rate / (55 * hop))))
    if max_lag <= min_lag or corr[0] <= 1e-9:
        return None, 0.0
    window = corr[min_lag:max_lag + 1]
    lag = min_lag + int(np.argmax(window))
    confidence = float(np.clip(corr[lag] / corr[0], 0, 1))
    if confidence < 0.08:
        return None, confidence
    return round(60 * sample_rate / (lag * hop), 1), round(confidence, 3)


class LocalAudioAnalyzer:
    def __init__(self, model_name: str, ffmpeg: str):
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError as exc:
            raise RuntimeError("缺少本地分析依赖，请重新运行安装Conda环境.ps1") from exc
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        default_batch = 2 if self.device == "cuda" else 1
        self.batch_size = max(1, int(os.getenv("BGM_AST_BATCH_SIZE", default_batch)))
        self.extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_name, use_safetensors=True
        ).to(self.device)
        self.model.eval()
        self.model_name = f"{model_name}@music-v2"
        self.ffmpeg = ffmpeg
        self.labels = {int(k): str(v) for k, v in self.model.config.id2label.items()}

    def _predict_scores(self, clips: list[np.ndarray]) -> np.ndarray:
        scores = []
        try:
            with self.torch.inference_mode():
                for start in range(0, len(clips), self.batch_size):
                    batch = clips[start:start + self.batch_size]
                    inputs = self.extractor(
                        batch, sampling_rate=16000, return_tensors="pt", padding=True
                    )
                    inputs = {key: value.to(self.device) for key, value in inputs.items()}
                    logits = self.model(**inputs).logits.detach().cpu().numpy()
                    scores.extend(_sigmoid(row) for row in logits)
        except RuntimeError as exc:
            if self.device != "cuda" or "out of memory" not in str(exc).casefold():
                raise
            self.torch.cuda.empty_cache()
            if self.batch_size > 1:
                # Retry the same asset with a smaller batch before giving up the
                # accelerator. This is the normal fallback for 4 GB cards.
                self.batch_size = max(1, self.batch_size // 2)
                return self._predict_scores(clips)
            self.model = self.model.to("cpu")
            self.device = "cpu"
            self.batch_size = 1
            self.torch.cuda.empty_cache()
            return self._predict_scores(clips)
        return np.median(np.stack(scores), axis=0)

    def analyze(self, path: Path, duration_ms: int | None) -> dict:
        clips = [decode_excerpt(path, self.ffmpeg, start, 10, 16000)
                 for start in excerpt_starts(duration_ms, 10)]
        # A tag must describe the music consistently, not only one accidental excerpt.
        probabilities = self._predict_scores(clips)
        label_scores = {self.labels[i].casefold(): float(score) for i, score in enumerate(probabilities)}
        music_score = label_scores.get("music", 0.0)
        audio_type, type_label, type_score = "music", "音乐", music_score
        tags = []
        tags.extend(self._mapped_tags(
            label_scores, "instrument", INSTRUMENTS, 0.35, 1, min_margin=0.08
        ))
        tags.extend(self._mapped_tags(label_scores, "mood", MOODS, 0.35, 2))
        tags.extend(self._mapped_tags(label_scores, "style", STYLES, 0.35, 2))
        joined = np.concatenate(clips)
        bpm, bpm_confidence = estimate_tempo(joined)
        if bpm is not None and bpm_confidence >= 0.20:
            if bpm < 70:
                code, label = "very_slow", "很慢（<70 BPM）"
            elif bpm < 90:
                code, label = "slow", "慢速（70–89 BPM）"
            elif bpm < 120:
                code, label = "medium", "中速（90–119 BPM）"
            elif bpm < 150:
                code, label = "fast", "快速（120–149 BPM）"
            else:
                code, label = "very_fast", "很快（≥150 BPM）"
            tags.append({"namespace": "rhythm", "code": code, "label": label,
                         "score": bpm_confidence, "rank_no": 1})
        return {"audio_type": audio_type, "audio_type_label": type_label,
                "type_score": type_score, "bpm": bpm, "bpm_confidence": bpm_confidence,
                "tags": tags}

    def analyze_effect_description(self, path: Path, duration_ms: int | None) -> dict:
        """Create a cached, model-grounded description from several audio excerpts.

        The description deliberately contains only labels supported by the audio
        classifier.  It is less fluent than a generative caption, but avoids
        hallucinating an event that is not present in the recording.
        """
        clips = [decode_excerpt(path, self.ffmpeg, start, 10, 16000)
                 for start in excerpt_starts(duration_ms, 10)]
        probabilities = self._predict_scores(clips)
        broad = []
        result = []
        non_actions = {
            "music", "musical instrument", "sound effect", "noise", "silence",
            "inside, small room", "inside, large room or hall", "outside, urban or manmade",
            "outside, rural or natural", "echo", "speech", "glass", "wood", "metal",
        }
        for index in np.argsort(probabilities)[::-1][:12]:
            score = float(probabilities[int(index)])
            if score < 0.18:
                continue
            label = self.labels[int(index)]
            broad.append({"label": label, "score": round(score, 4)})
            if score < 0.25:
                continue
            if label.casefold() in non_actions:
                continue
            result.append({"label": label, "score": round(score, 4)})
        # Generic AudioSet classes add almost no discriminative information and
        # used to pull many effects towards unrelated recommendation branches.
        labels = [item["label"] for item in broad
                  if item["label"].casefold() not in GENERIC_EFFECT_LABELS][:8]
        caption = "This audio contains " + ", ".join(labels) + "." if labels else ""
        return {"caption": caption, "labels": broad, "actions": result}

    def analyze_effect_actions(self, path: Path, duration_ms: int | None) -> list[dict]:
        return self.analyze_effect_description(path, duration_ms)["actions"]

    @staticmethod
    def _mapped_tags(scores, namespace, mapping, threshold, keep, min_margin=0.0):
        found = []
        for code, (label, hints) in mapping.items():
            # AudioSet labels are matched exactly to avoid overlaps such as
            # "guitar" also claiming "acoustic guitar" and "electric guitar".
            score = max((value for name, value in scores.items()
                         if name in hints), default=0.0)
            if score >= threshold:
                found.append((score, code, label))
        found.sort(reverse=True)
        if min_margin and len(found) > 1 and found[0][0] - found[1][0] < min_margin:
            return []
        return [{"namespace": namespace, "code": code, "label": label,
                 "score": score, "rank_no": rank}
                for rank, (score, code, label) in enumerate(found[:keep], 1)]


def analyze_library(db: Database, analyzer: LocalAudioAnalyzer, limit: int | None = None,
                    progress: Progress | None = None) -> dict:
    sql = """SELECT a.* FROM asset a LEFT JOIN audio_analysis x ON x.asset_id=a.id
             WHERE a.missing=0 AND a.media_type='music'
             AND (x.asset_id IS NULL OR x.status<>'ready')
             ORDER BY a.id"""
    params: list[object] = []
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with db.connect() as conn:
        rows = [dict(row) for row in conn.execute(sql, params)]
    result = {"queued": len(rows), "completed": 0, "failed": 0, "current": ""}
    for row in rows:
        result["current"] = row["filename"]
        try:
            output = analyzer.analyze(Path(row["path"]), row["duration_ms"])
            with db.connect() as conn:
                conn.execute("DELETE FROM asset_tag WHERE asset_id=?", (row["id"],))
                conn.executemany(
                    """INSERT INTO asset_tag
                    (asset_id, namespace, code, label, score, rank_no)
                    VALUES (?, ?, ?, ?, ?, ?)""",
                    [(row["id"], t["namespace"], t["code"], t["label"], t["score"],
                      t["rank_no"]) for t in output["tags"]],
                )
                conn.execute("""INSERT OR REPLACE INTO audio_analysis
                    (asset_id, source_modified_ns, audio_type, audio_type_label, type_score,
                     bpm, bpm_confidence, model, status, error, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', NULL, ?)""",
                    (row["id"], row["modified_ns"], output["audio_type"],
                     output["audio_type_label"], output["type_score"], output["bpm"],
                     output["bpm_confidence"], analyzer.model_name, time.time()))
            result["completed"] += 1
        except Exception as exc:
            with db.connect() as conn:
                conn.execute("""INSERT OR REPLACE INTO audio_analysis
                    (asset_id, source_modified_ns, audio_type, audio_type_label, type_score,
                     bpm, bpm_confidence, model, status, error, created_at)
                    VALUES (?, ?, 'unknown', '分析失败', 0, NULL, NULL, ?, 'failed', ?, ?)""",
                    (row["id"], row["modified_ns"], analyzer.model_name, str(exc)[:1000], time.time()))
            result["failed"] += 1
        if progress:
            progress(result.copy())
    result["current"] = ""
    if progress:
        progress(result.copy())
    return result


def analyze_effect_actions(db: Database, analyzer: LocalAudioAnalyzer,
                           limit: int | None = None,
                           progress: Progress | None = None) -> dict:
    sql = """SELECT a.* FROM asset a LEFT JOIN effect_action_analysis x ON x.asset_id=a.id
             WHERE a.missing=0 AND a.media_type='effect' AND a.status<>'duplicate'
             AND (x.asset_id IS NULL OR x.status<>'ready') ORDER BY a.id"""
    action_model = f"{analyzer.model_name}@effect-action-v2"
    params: list[object] = []
    if limit:
        sql += " LIMIT ?"
        params.append(limit)
    with db.connect() as conn:
        rows = [dict(row) for row in conn.execute(sql, params)]
    result = {"queued": len(rows), "completed": 0, "failed": 0, "current": ""}
    for row in rows:
        result["current"] = row["filename"]
        try:
            description = analyzer.analyze_effect_description(
                Path(row["path"]), row["duration_ms"]
            )
            actions = description["actions"]
            with db.connect() as conn:
                conn.execute("""INSERT OR REPLACE INTO effect_action_analysis
                    (asset_id,source_modified_ns,model,actions_json,status,error,created_at)
                    VALUES (?,?,?,?, 'ready',NULL,?)""",
                    (row["id"], row["modified_ns"], action_model,
                     json.dumps(actions, ensure_ascii=False), time.time()))
                conn.execute("""INSERT OR REPLACE INTO effect_caption_analysis
                    (asset_id,source_modified_ns,model,caption,labels_json,status,error,created_at)
                    VALUES (?,?,?,?,?,'ready',NULL,?)""",
                    (row["id"], row["modified_ns"], action_model,
                     description["caption"],
                     json.dumps(description["labels"], ensure_ascii=False), time.time()))
            result["completed"] += 1
        except Exception as exc:
            with db.connect() as conn:
                conn.execute("""INSERT OR REPLACE INTO effect_action_analysis
                    (asset_id,source_modified_ns,model,actions_json,status,error,created_at)
                    VALUES (?,?,?,'[]','failed',?,?)""",
                    (row["id"], row["modified_ns"], action_model,
                     str(exc)[:1000], time.time()))
                conn.execute("""INSERT OR REPLACE INTO effect_caption_analysis
                    (asset_id,source_modified_ns,model,caption,labels_json,status,error,created_at)
                    VALUES (?,?,?,'','[]','failed',?,?)""",
                    (row["id"], row["modified_ns"], action_model,
                     str(exc)[:1000], time.time()))
            result["failed"] += 1
        if progress:
            progress(result.copy())
    result["current"] = ""
    if progress:
        progress(result.copy())
    return result
