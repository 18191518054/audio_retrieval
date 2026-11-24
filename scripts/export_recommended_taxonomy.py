from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path


def export(source: Path, destination: Path) -> int:
    with zipfile.ZipFile(source) as archive:
        sheets = json.loads(archive.read("content.json"))
    root = sheets[0]["rootTopic"]
    count = 0

    def convert(topic: dict, parents: list[str]) -> dict:
        nonlocal count
        label = str(topic.get("title", "")).strip()
        path = [*parents, label]
        code = "rec-" + hashlib.sha1("\x1f".join(path).encode("utf-8")).hexdigest()[:16]
        children = [convert(child, path) for child in topic.get("children", {}).get("attached", [])]
        count += 1
        return {
            "code": code,
            "label": label,
            "description": " > ".join(path),
            "examples": [],
            "children": children,
        }

    nodes = [convert(topic, []) for topic in root.get("children", {}).get("attached", [])]
    payload = {
        "schema_version": 1,
        "taxonomy_code": "recommended",
        "label": "推荐标签",
        "nodes": nodes,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return count


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: export_recommended_taxonomy.py SOURCE.xmind DESTINATION.json")
    total = export(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"exported {total} nodes")
