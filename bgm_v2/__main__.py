from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from .config import Settings
from .db import Database
from .embedder import create_embedder
from .indexer import build_embeddings
from .media import executable_available
from .scanner import scan_directory
from .search import search_assets
from .web import serve


def doctor(settings: Settings) -> int:
    checks = {
        "python": sys.version.split()[0],
        "supported_python": sys.version_info[:2] in {(3, 11), (3, 12)},
        "platform": platform.platform(),
        "cpu_threads": os.cpu_count(),
        "recommended_provider": settings.recommended_provider,
        "aliyun_configured": bool(settings.aliyun_api_key and settings.aliyun_base_url),
        "ffmpeg": shutil.which(settings.ffmpeg) or (settings.ffmpeg if Path(settings.ffmpeg).is_file() else None),
        "ffprobe": shutil.which(settings.ffprobe) or (settings.ffprobe if Path(settings.ffprobe).is_file() else None),
    }
    try:
        import torch
        import transformers
        checks["torch"] = torch.__version__
        checks["transformers"] = transformers.__version__
        checks["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        checks["clap_dependencies"] = False
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not checks["supported_python"]:
        print("建议使用 Python 3.11 或 3.12。")
    if not executable_available(settings.ffmpeg):
        print("未找到 FFmpeg：可以扫描 WAV，但不能生成 CLAP 音频向量。")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="本地 BGM V2 检索验证版")
    sub = root.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="检查运行环境")
    sub.add_parser("serve", help="启动本地网页")
    scan = sub.add_parser("scan", help="扫描素材目录")
    scan.add_argument("path")
    scan.add_argument("--type", choices=["music", "effect"], default="music")
    index = sub.add_parser("index", help="生成音频向量")
    index.add_argument("--provider", choices=["clap", "diagnostic"], default="clap")
    index.add_argument("--limit", type=int)
    search = sub.add_parser("search", help="搜索")
    search.add_argument("query", nargs="?", default="")
    search.add_argument("--reference")
    search.add_argument("--provider", choices=["clap", "diagnostic"], default="clap")
    search.add_argument("--limit", type=int, default=10)
    return root


def main() -> int:
    args = parser().parse_args()
    settings = Settings.from_env()
    settings.ensure_dirs()
    if args.command == "doctor":
        return doctor(settings)
    if args.command == "serve":
        serve(settings)
        return 0
    db = Database(settings.db_path)
    if args.command == "scan":
        print(json.dumps(scan_directory(db, Path(args.path), settings.ffprobe,
                                        media_type=args.type), ensure_ascii=False, indent=2))
        return 0
    embedder = create_embedder(
        args.provider, settings.clap_model, settings.ffmpeg, settings.ffprobe
    )
    if args.command == "index":
        print(json.dumps(build_embeddings(db, embedder, args.limit), ensure_ascii=False, indent=2))
    elif args.command == "search":
        items = search_assets(db, embedder, args.query, Path(args.reference) if args.reference else None, limit=args.limit)
        for item in items:
            print(f"{item['score']:.3f}  {item['filename']}  {item['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
