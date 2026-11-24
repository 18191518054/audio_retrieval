from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    host: str
    port: int
    ffmpeg: str
    ffprobe: str
    clap_model: str
    max_upload_mb: int
    analysis_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    text_model: str = "BAAI/bge-small-zh-v1.5"
    recommended_provider: str = "local"
    aliyun_api_key: str = ""
    aliyun_base_url: str = ""
    aliyun_model: str = "qwen3.8-omni-flash"
    cloud_fallback_local: bool = True
    cloud_min_confidence: float = 0.78
    cloud_timeout_seconds: int = 120
    cloud_max_concurrency: int = 2
    cloud_ucs_category_confidence: float = 0.85
    cloud_ucs_subcategory_confidence: float = 0.82

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("BGM_DATA_DIR", "data")).resolve()
        project_root = Path(__file__).resolve().parent.parent
        model_root = Path(os.getenv("BGM_MODEL_DIR", project_root / "models")).resolve()
        def local_or_remote(folder: str, remote: str) -> str:
            local = model_root / folder
            return str(local) if local.is_dir() else remote
        recommended_provider = os.getenv("BGM_RECOMMENDED_PROVIDER", "local").strip().casefold()
        if recommended_provider not in {"local", "aliyun"}:
            raise ValueError("BGM_RECOMMENDED_PROVIDER 必须是 local 或 aliyun")
        return cls(
            data_dir=data_dir,
            db_path=data_dir / "library.db",
            host=os.getenv("BGM_HOST", "127.0.0.1"),
            port=int(os.getenv("BGM_PORT", "8765")),
            ffmpeg=os.getenv("BGM_FFMPEG", "ffmpeg"),
            ffprobe=os.getenv("BGM_FFPROBE", "ffprobe"),
            clap_model=os.getenv("BGM_CLAP_MODEL", local_or_remote("clap", "laion/clap-htsat-unfused")),
            analysis_model=os.getenv("BGM_ANALYSIS_MODEL", local_or_remote("ast", "MIT/ast-finetuned-audioset-10-10-0.4593")),
            text_model=os.getenv("BGM_TEXT_MODEL", local_or_remote("bge", "BAAI/bge-small-zh-v1.5")),
            max_upload_mb=int(os.getenv("BGM_MAX_UPLOAD_MB", "200")),
            recommended_provider=recommended_provider,
            aliyun_api_key=os.getenv("DASHSCOPE_API_KEY", "").strip(),
            aliyun_base_url=os.getenv("DASHSCOPE_BASE_URL", "").strip().rstrip("/"),
            aliyun_model=os.getenv("BGM_ALIYUN_MODEL", "qwen3.8-omni-flash").strip(),
            cloud_fallback_local=os.getenv("BGM_CLOUD_FALLBACK_LOCAL", "1").strip().casefold()
            not in {"0", "false", "no", "off"},
            cloud_min_confidence=float(os.getenv("BGM_CLOUD_MIN_CONFIDENCE", "0.78")),
            cloud_timeout_seconds=max(10, int(os.getenv("BGM_CLOUD_TIMEOUT_SECONDS", "120"))),
            cloud_max_concurrency=min(
                8, max(1, int(os.getenv("BGM_CLOUD_MAX_CONCURRENCY", "2")))
            ),
            cloud_ucs_category_confidence=float(
                os.getenv("BGM_CLOUD_UCS_CATEGORY_CONFIDENCE", "0.85")
            ),
            cloud_ucs_subcategory_confidence=float(
                os.getenv("BGM_CLOUD_UCS_SUBCATEGORY_CONFIDENCE", "0.82")
            ),
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
