from __future__ import annotations

import json
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np


WAV_INFO_FIELDS = {
    b"INAM": "标题",
    b"IART": "艺术家",
    b"IPRD": "专辑",
    b"IGNR": "类型",
    b"ICMT": "描述",
    b"ISBJ": "主题",
}


def _decode_metadata_text(raw: bytes) -> str:
    value = raw.rstrip(b"\x00 \r\n")
    decoded: dict[str, str] = {}
    for encoding in ("utf-8", "gb18030"):
        try:
            decoded[encoding] = value.decode(encoding).strip()
        except UnicodeDecodeError:
            pass
    if "utf-8" in decoded and "gb18030" in decoded:
        utf_cjk = sum("\u4e00" <= char <= "\u9fff" for char in decoded["utf-8"])
        gb_cjk = sum("\u4e00" <= char <= "\u9fff" for char in decoded["gb18030"])
        if gb_cjk >= 2 and utf_cjk == 0:
            return decoded["gb18030"]
        return decoded["utf-8"]
    if decoded:
        return next(iter(decoded.values()))
    return value.decode("utf-8", errors="replace").strip()


def _wav_info_metadata(path: Path) -> dict[str, str]:
    """Read legacy RIFF INFO tags without forcing their bytes through UTF-8."""
    result: dict[str, str] = {}
    try:
        with path.open("rb") as source:
            header = source.read(12)
            if len(header) != 12 or header[:4] not in {b"RIFF", b"RF64"} or header[8:] != b"WAVE":
                return result
            while True:
                chunk = source.read(8)
                if len(chunk) != 8:
                    break
                chunk_id, size = struct.unpack("<4sI", chunk)
                if chunk_id != b"LIST" or size < 4 or size > 4 * 1024 * 1024:
                    source.seek(size + (size & 1), 1)
                    continue
                payload = source.read(size)
                if size & 1:
                    source.seek(1, 1)
                if payload[:4] != b"INFO":
                    continue
                position = 4
                while position + 8 <= len(payload):
                    field_id, field_size = struct.unpack_from("<4sI", payload, position)
                    position += 8
                    field = payload[position:position + field_size]
                    position += field_size + (field_size & 1)
                    label = WAV_INFO_FIELDS.get(field_id)
                    text = _decode_metadata_text(field)
                    if label and text:
                        result[label] = text[:1000]
    except (OSError, struct.error):
        return {}
    return result


def executable_available(command: str) -> bool:
    return Path(command).is_file() or shutil.which(command) is not None


def duration_ms(path: Path, ffprobe: str) -> int | None:
    if executable_available(ffprobe):
        try:
            proc = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=True,
            )
            seconds = float(json.loads(proc.stdout)["format"]["duration"])
            return max(0, round(seconds * 1000))
        except (subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
            return None
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                return round(wav.getnframes() / wav.getframerate() * 1000)
        except (wave.Error, OSError, ZeroDivisionError):
            return None
    return None


def audio_metadata(path: Path, ffprobe: str) -> dict[str, str]:
    """Read useful embedded metadata without exposing technical stream noise."""
    result = _wav_info_metadata(path) if path.suffix.casefold() == ".wav" else {}
    if not executable_available(ffprobe):
        return result
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format_tags", "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=True,
        )
        tags = json.loads(proc.stdout).get("format", {}).get("tags", {})
    except (subprocess.SubprocessError, json.JSONDecodeError, AttributeError):
        return result
    aliases = {
        "title": "标题", "artist": "艺术家", "album": "专辑", "genre": "类型",
        "comment": "描述", "description": "描述", "keywords": "关键词",
        "subject": "主题", "composer": "作曲", "track": "音轨",
        "catid": "UCS CatID", "category": "UCS Category",
        "subcategory": "UCS SubCategory",
    }
    for key, value in tags.items():
        clean = str(value).strip()
        label = aliases.get(str(key).casefold())
        if label and clean and label not in result and "�" not in clean:
            result[label] = clean[:1000]
    return result


def decode_excerpt(
    path: Path, ffmpeg: str, start_seconds: float = 0, seconds: int = 10, sample_rate: int = 48000
) -> np.ndarray:
    """Decode only one short mono excerpt to keep peak memory low."""
    if path.suffix.casefold() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav:
                channels = wav.getnchannels()
                width = wav.getsampwidth()
                source_rate = wav.getframerate()
                wav.setpos(min(wav.getnframes(), round(max(0, start_seconds) * source_rate)))
                raw = wav.readframes(min(wav.getnframes() - wav.tell(), seconds * source_rate))
            if width == 1:
                audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
            elif width == 2:
                audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768
            elif width == 3:
                packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
                values = (packed[:, 0].astype(np.int32)
                          | (packed[:, 1].astype(np.int32) << 8)
                          | (packed[:, 2].astype(np.int32) << 16))
                values = np.where(values & 0x800000, values - 0x1000000, values)
                audio = values.astype(np.float32) / 8388608
            elif width == 4:
                audio = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648
            else:
                raise ValueError(f"unsupported PCM width: {width}")
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            if source_rate != sample_rate and audio.size:
                target_size = max(1, round(audio.size * sample_rate / source_rate))
                audio = np.interp(
                    np.linspace(0, audio.size - 1, target_size),
                    np.arange(audio.size), audio,
                ).astype(np.float32)
            if audio.size:
                return np.asarray(audio, dtype=np.float32)
        except (OSError, ValueError, wave.Error):
            # Compressed/float WAV variants unsupported by the stdlib continue
            # through the normal FFmpeg decoder below.
            pass
    if not executable_available(ffmpeg):
        raise RuntimeError("未找到 FFmpeg；CLAP 分析需要先安装 FFmpeg 或设置 BGM_FFMPEG")
    proc = subprocess.run(
        [
            ffmpeg, "-nostdin", "-v", "error", "-ss", str(max(0, start_seconds)), "-i", str(path), "-t", str(seconds),
            "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "pipe:1",
        ],
        capture_output=True,
        timeout=30,
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"FFmpeg 解码失败（退出码 {proc.returncode}）：{message or '没有错误输出'}")
    audio = np.frombuffer(proc.stdout, dtype="<f4").copy()
    if audio.size == 0:
        raise RuntimeError("解码结果为空")
    return audio


def excerpt_starts(total_ms: int | None, seconds: int = 10) -> list[float]:
    if not total_ms or total_ms <= seconds * 1000:
        return [0]
    total = total_ms / 1000
    max_start = max(0, total - seconds)
    return sorted(set([min(5.0, max_start), max_start / 2, max_start]))
