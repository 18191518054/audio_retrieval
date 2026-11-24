from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


SAFE_PATTERNS = [
    "*.json", "*.txt", "*.model", "*.safetensors", "merges.txt", "vocab.json"
]


def download(repo_id: str, target: Path, allow_bin: bool = False) -> str:
    # Local paths are already complete and must not be sent to the Hub.
    from pathlib import Path
    if Path(repo_id).is_dir():
        print(f"已是本地模型，跳过下载：{repo_id}")
        return str(Path(repo_id).resolve())
    target.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        allow_patterns=SAFE_PATTERNS + (["pytorch_model.bin"] if allow_bin else []),
        ignore_patterns=["*.h5", "*.msgpack"] + ([] if allow_bin else ["*.bin"]),
    )
    print(f"下载完成：{repo_id}\n缓存位置：{path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="预下载 BGM 本地检索所需模型")
    parser.add_argument("--clap", default="laion/clap-htsat-unfused")
    parser.add_argument("--analysis", default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--text", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--model-root", default="models")
    args = parser.parse_args()
    root = Path(args.model_root).resolve()
    download(args.clap, root / "clap", allow_bin=True)
    download(args.analysis, root / "ast")
    download(args.text, root / "bge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
