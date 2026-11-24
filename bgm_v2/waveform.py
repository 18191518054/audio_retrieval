from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .media import executable_available


def generate_peaks(
    path: Path,
    ffmpeg: str,
    points: int = 800,
    sample_rate: int = 4000,
    max_seconds: int = 1800,
) -> np.ndarray:
    """Generate normalized mono waveform peaks with a bounded memory footprint."""
    if not executable_available(ffmpeg):
        raise RuntimeError("未找到 FFmpeg，无法生成波形")
    proc = subprocess.run(
        [
            ffmpeg, "-v", "error", "-i", str(path), "-t", str(max_seconds),
            "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
        ],
        capture_output=True,
        timeout=max(120, max_seconds * 2),
    )
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"FFmpeg 波形解码失败（退出码 {proc.returncode}）：{message or '没有错误输出'}")
    samples = np.abs(np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32))
    if samples.size == 0:
        raise RuntimeError("音频解码结果为空")
    edges = np.linspace(0, samples.size, points + 1, dtype=np.int64)
    peaks = np.empty(points, dtype=np.float32)
    for index in range(points):
        section = samples[edges[index] : edges[index + 1]]
        peaks[index] = float(section.max()) if section.size else 0.0
    maximum = float(peaks.max())
    if maximum > 0:
        peaks /= maximum
    return peaks

