from __future__ import annotations

import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from bgm_v2.db import Database
from bgm_v2.embedder import DiagnosticEmbedder
from bgm_v2.effect_tagging import (
    auto_tag_effects, _hierarchical_instrument_material_matches,
    _instrument_material_matches, _recommended_matches,
)
from bgm_v2.indexer import build_embeddings
from bgm_v2.scanner import scan_directory
from bgm_v2.search import search_assets
from bgm_v2.taxonomy import (
    asset_nodes, create_custom_node, delete_custom_node, initialize_schemes,
    list_children, rename_custom_node, set_asset_node, set_asset_ucs,
    set_asset_recommended,
)
from bgm_v2.text_embedding import TextEmbedder

def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        out.writeframes(np.zeros(8000, dtype="<i2").tobytes())


class CountingDiagnosticEmbedder(DiagnosticEmbedder):
    def __init__(self):
        self.texts_embedded = 0

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.texts_embedded += len(texts)
        return super().embed_texts(texts, batch_size)


class FixedConditionedEmbedder(DiagnosticEmbedder):
    def __init__(self):
        self.calls = 0

    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.calls += 1
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)[:len(texts)]


class FixedBgeEmbedder(TextEmbedder):
    model_name = "fixed-bge"

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


class TaxonomyTests(unittest.TestCase):
    def test_instrument_material_priority_uses_clear_audio_first(self):
        candidates = [
            {"id": 1, "parent_id": None, "root_order": 0,
             "label": "乐器", "descendant_labels": []},
            {"id": 2, "parent_id": 1, "root_order": 0,
             "label": "二胡", "descendant_labels": []},
            {"id": 3, "parent_id": 1, "root_order": 0,
             "label": "琵琶", "descendant_labels": []},
        ]
        vectors = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        matches = _hierarchical_instrument_material_matches(
            np.asarray([1.0, 0.0], dtype=np.float32), (candidates, vectors),
            None, "unknown.wav", "", [], "", None,
        )
        self.assertEqual(matches[0][0], 2)
        self.assertIn("CLAP", matches[0][2])

    def test_material_audio_requires_detected_action(self):
        candidates = [
            {"id": 1, "parent_id": None, "root_order": 1,
             "label": "材质", "descendant_labels": []},
            {"id": 2, "parent_id": 1, "root_order": 1,
             "label": "金属", "descendant_labels": []},
            {"id": 3, "parent_id": 1, "root_order": 1,
             "label": "玻璃", "descendant_labels": []},
        ]
        vectors = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        matches = _hierarchical_instrument_material_matches(
            np.asarray([1.0, 0.0], dtype=np.float32), (candidates, vectors),
            None, "unknown.wav", "", [], "", None,
        )
        self.assertEqual(matches, [])

    def test_whoosh_cannot_authorize_material_classification(self):
        candidates = [
            {"id": 1, "parent_id": None, "root_order": 1,
             "label": "材质", "descendant_labels": []},
            {"id": 2, "parent_id": 1, "root_order": 1,
             "label": "橡胶", "descendant_labels": []},
            {"id": 3, "parent_id": 1, "root_order": 1,
             "label": "金属", "descendant_labels": []},
        ]
        vectors = np.asarray([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        matches = _hierarchical_instrument_material_matches(
            np.asarray([1.0, 0.0], dtype=np.float32), (candidates, vectors),
            None, "Big Whoosh.wav", "",
            [{"label": "Whoosh, swoosh, swish", "score": 0.92}], "", None,
        )
        self.assertEqual(matches, [])

    def test_recommended_hierarchy_accepts_consensus_and_hides_conflict(self):
        candidates = [
            {"id": 1, "parent_id": None, "label": "脚步", "path": "脚步"},
            {"id": 2, "parent_id": None, "label": "爆炸", "path": "爆炸"},
        ]
        vectors = np.asarray([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        model = (candidates, vectors)
        text_model = (candidates, vectors.copy())
        text_embedder = FixedBgeEmbedder()
        accepted = _recommended_matches(
            np.asarray([1.0, 0.0]), model, text_model=text_model,
            filename="木地板脚步行走.wav", action_context=[{"label": "Walk", "score": 0.9}],
            text_embedder=text_embedder,
        )
        self.assertEqual(accepted[0][0], 1)
        conflicted = _recommended_matches(
            np.asarray([-1.0, 0.0]), model, text_model=text_model,
            filename="木地板脚步行走.wav", action_context=[{"label": "Walk", "score": 0.9}],
            text_embedder=text_embedder,
        )
        self.assertEqual(conflicted, [])

    def test_material_requires_action_and_cross_model_agreement(self):
        candidates = [
            {"id": 10, "root_order": 1, "path": "材质 > 金属"},
            {"id": 11, "root_order": 1, "path": "材质 > 木材"},
        ]
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        model = (candidates, vectors)
        embedder = FixedConditionedEmbedder()
        self.assertEqual(_instrument_material_matches(
            np.asarray([1.0, 0.0]), model, [], embedder, {}
        ), [])
        matches = _instrument_material_matches(
            np.asarray([1.0, 0.0]), model,
            [{"label": "Hammer", "score": 0.8}], embedder, {},
        )
        self.assertEqual(matches[0][0], 10)
        self.assertIn("AST", matches[0][2])

    def test_conditioned_material_vectors_survive_restart(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Database(Path(temp) / "library.db")
            initialize_schemes(db, Path(temp))
            with db.connect() as conn:
                rows = conn.execute(
                    """WITH RECURSIVE descendants(id,path) AS (
                        SELECT id,label FROM tag_node WHERE parent_id IS NULL AND sort_order=1
                        UNION ALL SELECT n.id,descendants.path || ' > ' || n.label
                        FROM tag_node n JOIN descendants ON n.parent_id=descendants.id)
                    SELECT d.id,d.path FROM descendants d
                    WHERE NOT EXISTS (SELECT 1 FROM tag_node c WHERE c.parent_id=d.id)
                    LIMIT 2"""
                ).fetchall()
            candidates = [{"id": row["id"], "root_order": 1, "path": row["path"]}
                          for row in rows]
            model = (candidates, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
            embedder = FixedConditionedEmbedder()
            actions = [{"label": "Hammer", "score": 0.8}]
            _instrument_material_matches(
                np.asarray([1.0, 0.0]), model, actions, embedder, {}, db
            )
            self.assertEqual(embedder.calls, 1)
            _instrument_material_matches(
                np.asarray([1.0, 0.0]), model, actions, embedder, {}, db
            )
            self.assertEqual(embedder.calls, 1)

    def test_dynamic_effect_tags_preserve_custom_and_use_ucs_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = Database(root / "library.db")
            initialize_schemes(db, root)
            audio_dir = root / "effects"
            audio_dir.mkdir()
            write_wav(audio_dir / "AIRBlow_metal.wav")
            write_wav(audio_dir / "WHSH_whoosh.wav")
            scan_directory(db, audio_dir, "missing", media_type="effect")
            custom = create_custom_node(db, "精选")
            with db.connect() as conn:
                asset_id = conn.execute("SELECT id FROM asset").fetchone()[0]
            set_asset_node(db, asset_id, custom["id"], True)
            cache = root / "taxonomy"
            cache.mkdir()
            (cache / "ucs_v8.2.1.csv").write_text(
                "Category,SubCategory,CatID,CatShort,Explanations,Synonyms\n"
                "AIR,BLOW,AIRBlow,AIR,Steady air blows,Airhose; Balloon\n"
                "SWOOSHES,WHOOSH,WHSH,WHSH,Large object whoosh,Whoosh; Swoosh\n",
                encoding="utf-8",
            )
            embedder = CountingDiagnosticEmbedder()
            build_embeddings(db, embedder)
            result = auto_tag_effects(db, cache, embedder)
            self.assertEqual(result["completed"], 2)
            nodes = asset_nodes(db, asset_id)
            labels = {(node["scheme_code"], node["label"]) for node in nodes}
            self.assertIn(("ucs", "BLOW"), labels)
            self.assertIn(("custom", "精选"), labels)
            with db.connect() as conn:
                whoosh_id = conn.execute(
                    "SELECT id FROM asset WHERE filename='WHSH_whoosh.wav'"
                ).fetchone()[0]
            whoosh_labels = {(node["scheme_code"], node["label"])
                              for node in asset_nodes(db, whoosh_id)}
            self.assertIn(("ucs", "WHOOSH"), whoosh_labels)
            with db.connect() as conn:
                whoosh_node_id = conn.execute(
                    """SELECT n.id FROM tag_node n JOIN tag_scheme s ON s.id=n.scheme_id
                    WHERE s.code='ucs' AND n.code='WHSH'"""
                ).fetchone()[0]
            set_asset_ucs(db, whoosh_id, whoosh_node_id)
            auto_tag_effects(db, cache, embedder)
            with db.connect() as conn:
                manual = conn.execute(
                    """SELECT assigned_by,evidence FROM asset_tag_node
                    WHERE asset_id=? AND node_id=?""", (whoosh_id, whoosh_node_id)
                ).fetchone()
            self.assertEqual(manual["assigned_by"], "user")
            self.assertEqual(manual["evidence"], "用户人工设置")
            with db.connect() as conn:
                recommended_leaf = conn.execute(
                    """SELECT n.id FROM tag_node n JOIN tag_scheme s ON s.id=n.scheme_id
                    WHERE s.code='recommended' AND NOT EXISTS
                    (SELECT 1 FROM tag_node c WHERE c.parent_id=n.id) LIMIT 1"""
                ).fetchone()[0]
            set_asset_recommended(db, whoosh_id, recommended_leaf, True)
            auto_tag_effects(db, cache, embedder)
            with db.connect() as conn:
                confirmed = conn.execute(
                    """SELECT assigned_by,evidence FROM asset_tag_node
                    WHERE asset_id=? AND node_id=?""", (whoosh_id, recommended_leaf)
                ).fetchone()
            self.assertEqual(confirmed["assigned_by"], "user")
            self.assertEqual(confirmed["evidence"], "用户人工确认")
            first_generated = embedder.texts_embedded
            self.assertGreater(first_generated, 0)
            auto_tag_effects(db, cache, embedder)
            self.assertEqual(embedder.texts_embedded, first_generated)
            with db.connect() as conn:
                leaf = conn.execute(
                    """SELECT n.id FROM tag_node n JOIN tag_scheme s ON s.id=n.scheme_id
                    WHERE s.code='recommended' AND NOT EXISTS
                    (SELECT 1 FROM tag_node c WHERE c.parent_id=n.id) LIMIT 1"""
                ).fetchone()[0]
                conn.execute("UPDATE tag_node SET description=description || ' updated' WHERE id=?", (leaf,))
            auto_tag_effects(db, cache, embedder)
            self.assertEqual(embedder.texts_embedded, first_generated + 1)

    def test_recommended_taxonomy_is_bundled_without_xmind(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = Database(root / "library.db")
            initialize_schemes(db, root)
            tree = list_children(db, "recommended")
            self.assertGreater(len(tree["nodes"]), 3)
            self.assertTrue(all(node["code"] for node in tree["nodes"]))
            self.assertEqual(
                [node["label"] for node in tree["nodes"][:4]],
                ["环境音", "人物动态", "主观音效", "过场音效（Whoosh）"],
            )
            with db.connect() as conn:
                scheme_id = conn.execute(
                    "SELECT id FROM tag_scheme WHERE code='recommended'"
                ).fetchone()[0]
                conn.execute(
                    "INSERT INTO tag_node(scheme_id,label,sort_order) VALUES (?,?,0)",
                    (scheme_id, tree["nodes"][0]["label"]),
                )
            initialize_schemes(db, root)
            with db.connect() as conn:
                legacy_count = conn.execute(
                    "SELECT COUNT(*) FROM tag_node WHERE scheme_id=? AND code IS NULL",
                    (scheme_id,),
                ).fetchone()[0]
            self.assertEqual(legacy_count, 0)

    def test_fixed_instrument_material_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = Database(root / "library.db")
            initialize_schemes(db, root)
            tree = list_children(db, "instrument_material")
            self.assertEqual([node["label"] for node in tree["nodes"]], ["乐器", "材质"])
            self.assertGreater(tree["nodes"][0]["child_count"], 5)
            self.assertGreater(tree["nodes"][1]["child_count"], 10)

    def test_custom_tree_and_effect_assignment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = Database(root / "library.db")
            initialize_schemes(db, root)
            parent = create_custom_node(db, "常用")
            child = create_custom_node(db, "门声", parent["id"])
            second_tag = create_custom_node(db, "室内", parent["id"])
            tree = list_children(db, "custom")
            self.assertEqual(tree["nodes"][0]["child_count"], 2)
            rename_custom_node(db, child["id"], "开关门")

            audio_dir = root / "effects"
            audio_dir.mkdir()
            write_wav(audio_dir / "door.wav")
            write_wav(audio_dir / "other.wav")
            scan_directory(db, audio_dir, "missing", media_type="effect")
            with db.connect() as conn:
                asset_id = conn.execute("SELECT id FROM asset WHERE filename='door.wav'").fetchone()[0]
                other_id = conn.execute("SELECT id FROM asset WHERE filename='other.wav'").fetchone()[0]
            set_asset_node(db, asset_id, child["id"], True)
            set_asset_node(db, asset_id, second_tag["id"], True)
            set_asset_node(db, other_id, child["id"], True)
            self.assertIn("开关门", {node["label"] for node in asset_nodes(db, asset_id)})
            embedder = DiagnosticEmbedder()
            build_embeddings(db, embedder)
            self.assertEqual(len(search_assets(
                db, embedder, media_type="effect", tag_node_ids=[parent["id"]]
            )), 2)
            self.assertEqual(len(search_assets(
                db, embedder, media_type="effect", tag_node_ids=[child["id"], second_tag["id"]]
            )), 1)
            label_search = search_assets(db, embedder, query="室内", media_type="effect")
            self.assertEqual(label_search[0]["id"], asset_id)
            self.assertIn("标签名称命中", label_search[0]["reason"])
            custom_paths = [tag.get("path") for tag in label_search[0]["tags"]
                            if tag["namespace"] == "custom"]
            self.assertIn(["常用", "室内"], custom_paths)
            filename_search = search_assets(db, embedder, query="door", media_type="effect")
            self.assertEqual(filename_search[0]["filename"], "door.wav")
            self.assertIn("文件名关键词命中", filename_search[0]["reason"])
            delete_custom_node(db, parent["id"])
            self.assertEqual(asset_nodes(db, asset_id), [])


if __name__ == "__main__":
    unittest.main()
