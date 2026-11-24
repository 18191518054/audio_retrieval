from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

from .db import Database
from .instrument_material import INSTRUMENT_MATERIAL_TREE


UCS_VERSION = "8.2.1"
INSTRUMENT_MATERIAL_VERSION = "business-v2-semantic"
UCS_URL = (
    "https://raw.githubusercontent.com/jmrsound/ucs-tools/main/"
    "src/ucs_tools/data/ucs_v8.2.1.csv"
)
SCHEMES = (
    ("ucs", "UCS 行业标准", 0),
    ("recommended", "推荐标签", 0),
    ("instrument_material", "乐器/材质", 0),
    ("custom", "用户自定义标签", 1),
)


def initialize_schemes(db: Database, project_root: Path) -> None:
    with db.connect() as conn:
        for code, label, editable in SCHEMES:
            conn.execute(
                """INSERT INTO tag_scheme(code,label,editable) VALUES (?,?,?)
                ON CONFLICT(code) DO UPDATE SET label=excluded.label, editable=excluded.editable""",
                (code, label, editable),
            )
    import_recommended_resource(db)
    import_instrument_material(db)


def _scheme(db: Database, code: str):
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM tag_scheme WHERE code=?", (code,)).fetchone()
    if row is None:
        raise ValueError(f"未知标签体系：{code}")
    return dict(row)


def import_recommended_resource(db: Database) -> None:
    path = Path(__file__).parent / "resources" / "recommended_taxonomy.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    scheme = _scheme(db, "recommended")
    # Remove nodes imported by the retired XMind pipeline. They had no stable
    # code and would otherwise coexist with the bundled JSON tree.
    with db.connect() as conn:
        conn.execute(
            "DELETE FROM tag_node WHERE scheme_id=? AND code IS NULL", (scheme["id"],)
        )
    if scheme["source_version"] == digest:
        return

    with db.connect() as conn:
        existing = {row["code"]: dict(row) for row in conn.execute(
            "SELECT id,code FROM tag_node WHERE scheme_id=? AND code IS NOT NULL",
            (scheme["id"],),
        )}
        seen_codes: set[str] = set()

        def insert_topics(topics, parent_id=None):
            for order, topic in enumerate(topics):
                label = str(topic.get("label", "")).strip()
                code = str(topic.get("code", "")).strip()
                if not label or not code:
                    continue
                seen_codes.add(code)
                values = (parent_id, label, str(topic.get("description", "")),
                          json.dumps(topic.get("examples", []), ensure_ascii=False), order)
                if code in existing:
                    node_id = existing[code]["id"]
                    conn.execute(
                        """UPDATE tag_node SET parent_id=?,label=?,description=?,
                        examples_json=?,sort_order=? WHERE id=?""", (*values, node_id),
                    )
                else:
                    cursor = conn.execute(
                        """INSERT INTO tag_node
                        (scheme_id,parent_id,code,label,description,examples_json,sort_order)
                        VALUES (?,?,?,?,?,?,?)""",
                        (scheme["id"], parent_id, code, label, values[2], values[3], order),
                    )
                    node_id = int(cursor.lastrowid)
                insert_topics(topic.get("children", []), node_id)

        insert_topics(payload.get("nodes", []))
        stale_ids = [row["id"] for code, row in existing.items() if code not in seen_codes]
        if stale_ids:
            conn.executemany("DELETE FROM tag_node WHERE id=?", [(node_id,) for node_id in stale_ids])
        conn.execute(
            "UPDATE tag_scheme SET source_version=? WHERE id=?", (digest, scheme["id"])
        )


def import_instrument_material(db: Database) -> None:
    scheme = _scheme(db, "instrument_material")
    if scheme["source_version"] == INSTRUMENT_MATERIAL_VERSION:
        return
    with db.connect() as conn:
        conn.execute("DELETE FROM tag_node WHERE scheme_id=?", (scheme["id"],))

        def insert_nodes(nodes, parent_id=None, parents=()):
            for order, (label, children) in enumerate(nodes):
                path = (*parents, label)
                code = "im-" + hashlib.sha1("\x1f".join(path).encode("utf-8")).hexdigest()[:16]
                cursor = conn.execute(
                    """INSERT INTO tag_node
                    (scheme_id,parent_id,code,label,description,sort_order)
                    VALUES (?,?,?,?,?,?)""",
                    (scheme["id"], parent_id, code, label, " > ".join(path), order),
                )
                insert_nodes(children, cursor.lastrowid, path)

        insert_nodes(INSTRUMENT_MATERIAL_TREE)
        conn.execute(
            "UPDATE tag_scheme SET source_version=? WHERE id=?",
            (INSTRUMENT_MATERIAL_VERSION, scheme["id"]),
        )


def ensure_ucs(db: Database, cache_dir: Path) -> None:
    scheme = _scheme(db, "ucs")
    if scheme["source_version"] == UCS_VERSION:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cache_dir / f"ucs_v{UCS_VERSION}.csv"
    if not csv_path.is_file():
        try:
            with urllib.request.urlopen(UCS_URL, timeout=60) as response:
                payload = response.read()
            csv_path.write_bytes(payload)
        except Exception as exc:
            raise RuntimeError(
                "首次使用 UCS 需要下载官方 v8.2.1 分类表，请检查网络后重试"
            ) from exc
    rows = list(csv.DictReader(io.StringIO(csv_path.read_text("utf-8-sig"))))
    if not rows or not {"Category", "SubCategory", "CatID"}.issubset(rows[0]):
        raise RuntimeError("UCS 分类表格式无效")
    with db.connect() as conn:
        conn.execute("DELETE FROM tag_node WHERE scheme_id=?", (scheme["id"],))
        categories: dict[str, int] = {}
        for order, row in enumerate(rows):
            category = row["Category"].strip()
            if category not in categories:
                cursor = conn.execute(
                    """INSERT INTO tag_node
                    (scheme_id,parent_id,code,label,description,sort_order)
                    VALUES (?,NULL,NULL,?,?,?)""",
                    (scheme["id"], category, category, len(categories)),
                )
                categories[category] = int(cursor.lastrowid)
            conn.execute(
                """INSERT OR IGNORE INTO tag_node
                (scheme_id,parent_id,code,label,description,examples_json,sort_order)
                VALUES (?,?,?,?,?,?,?)""",
                (scheme["id"], categories[category], row["CatID"].strip(),
                 row["SubCategory"].strip(), row.get("Explanations", "").strip(),
                 json.dumps([item.strip() for item in row.get("Synonyms", "").split(",")
                             if item.strip()], ensure_ascii=False), order),
            )
        conn.execute(
            "UPDATE tag_scheme SET source_version=? WHERE id=?",
            (UCS_VERSION, scheme["id"]),
        )


def list_children(db: Database, scheme_code: str, parent_id: int | None = None) -> dict:
    scheme = _scheme(db, scheme_code)
    if parent_id is not None:
        with db.connect() as conn:
            parent = conn.execute(
                "SELECT id,label,parent_id FROM tag_node WHERE id=? AND scheme_id=?",
                (parent_id, scheme["id"]),
            ).fetchone()
        if parent is None:
            raise ValueError("标签节点不存在")
        parent_data = dict(parent)
    else:
        parent_data = None
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT n.id,n.code,n.label,n.parent_id,n.sort_order,
            (SELECT COUNT(*) FROM tag_node c WHERE c.parent_id=n.id) child_count,
            (SELECT COUNT(*) FROM asset_tag_node a WHERE a.node_id=n.id) asset_count
            FROM tag_node n WHERE n.scheme_id=? AND n.parent_id IS ?
            ORDER BY n.sort_order,n.label""",
            (scheme["id"], parent_id),
        ).fetchall()
    return {"scheme": scheme, "parent": parent_data, "nodes": [dict(row) for row in rows]}


def create_custom_node(db: Database, label: str, parent_id: int | None = None) -> dict:
    label = label.strip()
    if not label or len(label) > 80:
        raise ValueError("标签名称长度应为 1–80 个字符")
    scheme = _scheme(db, "custom")
    with db.connect() as conn:
        if parent_id is not None:
            parent = conn.execute(
                "SELECT id FROM tag_node WHERE id=? AND scheme_id=?",
                (parent_id, scheme["id"]),
            ).fetchone()
            if parent is None:
                raise ValueError("父标签不存在")
        order = conn.execute(
            "SELECT COUNT(*) FROM tag_node WHERE scheme_id=? AND parent_id IS ?",
            (scheme["id"], parent_id),
        ).fetchone()[0]
        cursor = conn.execute(
            "INSERT INTO tag_node(scheme_id,parent_id,label,sort_order) VALUES (?,?,?,?)",
            (scheme["id"], parent_id, label, order),
        )
        return {"id": cursor.lastrowid, "label": label, "parent_id": parent_id}


def rename_custom_node(db: Database, node_id: int, label: str) -> None:
    scheme = _scheme(db, "custom")
    label = label.strip()
    if not label or len(label) > 80:
        raise ValueError("标签名称长度应为 1–80 个字符")
    with db.connect() as conn:
        cursor = conn.execute(
            "UPDATE tag_node SET label=? WHERE id=? AND scheme_id=?",
            (label, node_id, scheme["id"]),
        )
        if cursor.rowcount == 0:
            raise ValueError("自定义标签不存在")


def delete_custom_node(db: Database, node_id: int) -> None:
    scheme = _scheme(db, "custom")
    with db.connect() as conn:
        cursor = conn.execute(
            "DELETE FROM tag_node WHERE id=? AND scheme_id=?", (node_id, scheme["id"])
        )
        if cursor.rowcount == 0:
            raise ValueError("自定义标签不存在")


def set_asset_node(db: Database, asset_id: int, node_id: int, assigned: bool) -> None:
    with db.connect() as conn:
        asset = conn.execute(
            "SELECT id FROM asset WHERE id=? AND missing=0 AND media_type='effect'", (asset_id,)
        ).fetchone()
        node = conn.execute(
            """SELECT n.id,s.code scheme_code FROM tag_node n
            JOIN tag_scheme s ON s.id=n.scheme_id WHERE n.id=?""", (node_id,)
        ).fetchone()
        if asset is None:
            raise ValueError("只能给音效素材设置音效标签")
        if node is None:
            raise ValueError("标签不存在")
        if node["scheme_code"] != "custom":
            raise ValueError("列表中只能人工添加或移除用户自定义标签")
        if assigned:
            conn.execute(
                "INSERT OR IGNORE INTO asset_tag_node(asset_id,node_id,assigned_by,created_at) VALUES (?,?,?,?)",
                (asset_id, node_id, "user", time.time()),
            )
        else:
            conn.execute(
                "DELETE FROM asset_tag_node WHERE asset_id=? AND node_id=?", (asset_id, node_id)
            )


def set_asset_ucs(db: Database, asset_id: int, node_id: int | None) -> None:
    with db.connect() as conn:
        asset = conn.execute(
            "SELECT id FROM asset WHERE id=? AND missing=0 AND media_type='effect'", (asset_id,)
        ).fetchone()
        if asset is None:
            raise ValueError("只能给音效素材设置 UCS 标签")
        if node_id is not None:
            node = conn.execute(
                """SELECT n.id FROM tag_node n JOIN tag_scheme s ON s.id=n.scheme_id
                WHERE n.id=? AND s.code='ucs' AND NOT EXISTS
                (SELECT 1 FROM tag_node child WHERE child.parent_id=n.id)""", (node_id,)
            ).fetchone()
            if node is None:
                raise ValueError("请选择 UCS 末级标签")
        conn.execute(
            """DELETE FROM asset_tag_node WHERE asset_id=? AND node_id IN
            (SELECT n.id FROM tag_node n JOIN tag_scheme s ON s.id=n.scheme_id
             WHERE s.code='ucs')""", (asset_id,),
        )
        if node_id is not None:
            conn.execute(
                """INSERT INTO asset_tag_node
                (asset_id,node_id,assigned_by,confidence,evidence,created_at)
                VALUES (?,?, 'user',1.0,'用户人工设置',?)""",
                (asset_id, node_id, time.time()),
            )


def set_asset_recommended(db: Database, asset_id: int, node_id: int, assigned: bool) -> None:
    with db.connect() as conn:
        asset = conn.execute(
            "SELECT id FROM asset WHERE id=? AND missing=0 AND media_type='effect'", (asset_id,)
        ).fetchone()
        node = conn.execute(
            """SELECT n.id FROM tag_node n JOIN tag_scheme s ON s.id=n.scheme_id
            WHERE n.id=? AND s.code='recommended'""", (node_id,)
        ).fetchone()
        if asset is None:
            raise ValueError("只能给音效素材设置推荐标签")
        if node is None:
            raise ValueError("推荐标签不存在")
        if assigned:
            conn.execute(
                """DELETE FROM asset_tag_node WHERE asset_id=? AND assigned_by<>'user'
                AND node_id IN (SELECT n.id FROM tag_node n JOIN tag_scheme s
                ON s.id=n.scheme_id WHERE s.code='recommended')""", (asset_id,),
            )
            conn.execute(
                """INSERT OR REPLACE INTO asset_tag_node
                (asset_id,node_id,assigned_by,confidence,evidence,created_at)
                VALUES (?,?,'user',1.0,'用户人工确认',?)""",
                (asset_id, node_id, time.time()),
            )
        else:
            conn.execute(
                "DELETE FROM asset_tag_node WHERE asset_id=? AND node_id=? AND assigned_by='user'",
                (asset_id, node_id),
            )


def asset_nodes(db: Database, asset_id: int) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute(
            """SELECT n.id,n.code,n.label,s.code scheme_code,s.label scheme_label,
            a.assigned_by,a.confidence,a.evidence
            FROM asset_tag_node a JOIN tag_node n ON n.id=a.node_id
            JOIN tag_scheme s ON s.id=n.scheme_id WHERE a.asset_id=?
            ORDER BY s.id,n.label""",
            (asset_id,),
        ).fetchall()
    return [dict(row) for row in rows]
