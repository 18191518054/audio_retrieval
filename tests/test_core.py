from __future__ import annotations

import tempfile
import time
import unittest
import wave
import struct
from pathlib import Path

import numpy as np

from bgm_v2.db import Database
from bgm_v2.dedupe import prepare_effect_duplicates
from bgm_v2.embedder import DiagnosticEmbedder, expand_chinese_query
from bgm_v2.indexer import build_embeddings
from bgm_v2.media import audio_metadata, decode_excerpt
from bgm_v2.scanner import scan_directory
from bgm_v2.search import lexical_score, search_assets
from bgm_v2.waveform import generate_peaks
from bgm_v2.tagging import classify
from bgm_v2.analysis import INSTRUMENTS, LocalAudioAnalyzer, estimate_tempo
from bgm_v2.text_embedding import TextEmbedder, asset_documents, build_text_embeddings


class FakeTextEmbedder(TextEmbedder):
    model_name = "fake-text-v1"

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.asarray([
            [1.0, 0.0] if "first.wav" in text else [-1.0, 0.0] for text in texts
        ], dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)


def write_wav(path: Path, value: int = 100, frames: int = 8000) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(8000)
        samples = (np.ones(frames, dtype="<i2") * value).tobytes()
        out.writeframes(samples)


class CoreFlowTests(unittest.TestCase):
    def test_pcm24_wav_uses_native_decoder_without_ffmpeg(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pcm24.wav"
            values = np.asarray([0, 0x200000, -0x200000], dtype=np.int32)
            packed = np.column_stack((
                values & 0xFF, (values >> 8) & 0xFF, (values >> 16) & 0xFF,
            )).astype(np.uint8).tobytes()
            with wave.open(str(path), "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(3)
                out.setframerate(48000)
                out.writeframes(packed)
            audio = decode_excerpt(path, "missing-ffmpeg")
            np.testing.assert_allclose(audio, [0.0, 0.25, -0.25], atol=1e-6)

    def test_wav_info_metadata_decodes_gb18030_without_ffprobe(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metadata.wav"
            write_wav(path)
            fields = b""
            for code, text in ((b"INAM", "乐器小萝卜"), (b"IART", "十一@ET十一LX")):
                value = text.encode("gb18030") + b"\0"
                fields += code + struct.pack("<I", len(value)) + value
                fields += b"\0" * (len(value) & 1)
            payload = b"INFO" + fields
            chunk = b"LIST" + struct.pack("<I", len(payload)) + payload
            if len(payload) & 1:
                chunk += b"\0"
            with path.open("ab") as out:
                out.write(chunk)
            with path.open("r+b") as out:
                out.seek(4)
                out.write(struct.pack("<I", path.stat().st_size - 8))
            self.assertEqual(
                audio_metadata(path, "missing-ffprobe"),
                {"标题": "乐器小萝卜", "艺术家": "十一@ET十一LX"},
            )

    def test_hybrid_search_uses_persisted_text_semantics(self):
        first = self.music / "first.wav"
        second = self.music / "second.wav"
        write_wav(first, 100)
        write_wav(second, 200)
        scan_directory(self.db, self.music, "missing-ffprobe", media_type="effect")
        audio_embedder = DiagnosticEmbedder()
        build_embeddings(self.db, audio_embedder)
        text_embedder = FakeTextEmbedder()
        built = build_text_embeddings(self.db, text_embedder)
        self.assertEqual(built["generated"], 2)
        matches = search_assets(
            self.db, audio_embedder, query="行走的声音",
            text_embedder=text_embedder,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["filename"], "first.wav")
        self.assertGreater(matches[0]["text_cosine"], 0.5)

    def test_exact_tag_match_is_returned_before_semantic_match(self):
        first = self.music / "first.wav"
        second = self.music / "second.wav"
        write_wav(first, 100)
        write_wav(second, 200)
        scan_directory(self.db, self.music, "missing-ffprobe", media_type="effect")
        audio_embedder = DiagnosticEmbedder()
        build_embeddings(self.db, audio_embedder)
        with self.db.connect() as conn:
            scheme_id = conn.execute(
                "INSERT INTO tag_scheme(code,label,editable) VALUES ('custom','自定义',1)"
            ).lastrowid
            node_id = conn.execute(
                "INSERT INTO tag_node(scheme_id,label) VALUES (?, '必须出现')",
                (scheme_id,),
            ).lastrowid
            second_id = conn.execute(
                "SELECT id FROM asset WHERE filename='second.wav'"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO asset_tag_node(asset_id,node_id,created_at) VALUES (?,?,?)",
                (second_id, node_id, time.time()),
            )
        text_embedder = FakeTextEmbedder()
        build_text_embeddings(self.db, text_embedder)
        matches = search_assets(
            self.db, audio_embedder, query="必须出现", text_embedder=text_embedder,
        )
        self.assertEqual(matches[0]["filename"], "second.wav")
        self.assertEqual(matches[0]["tag_lexical_score"], 1.0)

    def test_clap_alone_cannot_admit_text_search_result(self):
        audio = self.music / "unrelated.wav"
        write_wav(audio)
        scan_directory(self.db, self.music, "missing-ffprobe", media_type="effect")
        embedder = DiagnosticEmbedder()
        build_embeddings(self.db, embedder)
        self.assertEqual(search_assets(
            self.db, embedder, query="玻璃破碎", text_embedder=None,
        ), [])

    def test_parent_tag_path_does_not_match_search_box(self):
        audio = self.music / "whoosh.wav"
        write_wav(audio)
        scan_directory(self.db, self.music, "missing-ffprobe", media_type="effect")
        embedder = DiagnosticEmbedder()
        build_embeddings(self.db, embedder)
        with self.db.connect() as conn:
            scheme_id = conn.execute(
                "INSERT INTO tag_scheme(code,label) VALUES ('recommended','推荐标签')"
            ).lastrowid
            parent_id = conn.execute(
                "INSERT INTO tag_node(scheme_id,label) VALUES (?, '转场')",
                (scheme_id,),
            ).lastrowid
            child_id = conn.execute(
                "INSERT INTO tag_node(scheme_id,parent_id,label) VALUES (?, ?, '基础呼啸')",
                (scheme_id, parent_id),
            ).lastrowid
            asset_id = conn.execute("SELECT id FROM asset").fetchone()[0]
            conn.execute(
                "INSERT INTO asset_tag_node(asset_id,node_id,created_at) VALUES (?,?,?)",
                (asset_id, child_id, time.time()),
            )
            conn.execute(
                "INSERT INTO asset_tag_node(asset_id,node_id,created_at) VALUES (?,?,?)",
                (asset_id, parent_id, time.time()),
            )
        document = asset_documents(self.db)[0]["text"]
        self.assertIn("基础呼啸", document)
        self.assertNotIn("转场", document)
        self.assertEqual(search_assets(
            self.db, embedder, query="转场", text_embedder=None,
        ), [])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.music = self.root / "music"
        self.music.mkdir()
        self.db = Database(self.root / "library.db")

    def tearDown(self):
        self.temp.cleanup()

    def test_scan_is_incremental_and_reads_wav_duration(self):
        song = self.music / "温暖钢琴.wav"
        write_wav(song)
        first = scan_directory(self.db, self.music, "missing-ffprobe")
        second = scan_directory(self.db, self.music, "missing-ffprobe")
        self.assertEqual(first["new"], 1)
        self.assertEqual(second["unchanged"], 1)
        with self.db.connect() as conn:
            row = conn.execute("SELECT * FROM asset").fetchone()
        self.assertEqual(row["duration_ms"], 1000)
        self.assertEqual(row["missing"], 0)

    def test_scan_assigns_and_updates_media_type(self):
        song = self.music / "typed.wav"
        write_wav(song)
        scan_directory(self.db, self.music, "missing-ffprobe", media_type="effect")
        with self.db.connect() as conn:
            self.assertEqual(conn.execute("SELECT media_type FROM asset").fetchone()[0], "effect")
            self.assertEqual(conn.execute("SELECT media_type FROM library_root").fetchone()[0], "effect")
        scan_directory(self.db, self.music, "missing-ffprobe", media_type="music")
        with self.db.connect() as conn:
            self.assertEqual(conn.execute("SELECT media_type FROM asset").fetchone()[0], "music")

    def test_identical_effects_are_filtered_without_deleting_sources(self):
        first = self.music / "same-a.wav"
        second = self.music / "same-b.wav"
        different = self.music / "different.wav"
        write_wav(first, 100)
        write_wav(second, 100)
        write_wav(different, 200)
        scan_directory(self.db, self.music, "missing-ffprobe", media_type="effect")
        summary = prepare_effect_duplicates(self.db)
        self.assertEqual(summary["effect_files"], 3)
        self.assertEqual(summary["duplicate_files"], 1)
        self.assertEqual(summary["indexable_effects"], 2)
        result = build_embeddings(self.db, DiagnosticEmbedder())
        self.assertEqual(result["completed"], 2)
        with self.db.connect() as conn:
            duplicate = conn.execute(
                "SELECT duplicate_of,status FROM asset WHERE status='duplicate'"
            ).fetchone()
            embedded = conn.execute("SELECT COUNT(*) FROM embedding").fetchone()[0]
        self.assertIsNotNone(duplicate["duplicate_of"])
        self.assertEqual(duplicate["status"], "duplicate")
        self.assertEqual(embedded, 2)
        self.assertTrue(second.is_file())

    def test_reference_search_returns_same_audio_first(self):
        first = self.music / "first.wav"
        second = self.music / "second.wav"
        write_wav(first, 100)
        write_wav(second, 200)
        scan_directory(self.db, self.music, "missing-ffprobe")
        embedder = DiagnosticEmbedder()
        result = build_embeddings(self.db, embedder)
        self.assertEqual(result["completed"], 2)
        matches = search_assets(self.db, embedder, reference_path=first)
        self.assertEqual(matches[0]["filename"], "first.wav")

    def test_modified_file_invalidates_old_embedding(self):
        song = self.music / "change.wav"
        write_wav(song, 100)
        scan_directory(self.db, self.music, "missing-ffprobe")
        build_embeddings(self.db, DiagnosticEmbedder())
        time.sleep(0.01)
        write_wav(song, 300, 9000)
        scan_directory(self.db, self.music, "missing-ffprobe")
        with self.db.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM embedding").fetchone()[0]
        self.assertEqual(count, 0)

    def test_delete_asset_record_keeps_source_file(self):
        song = self.music / "delete-record.wav"
        write_wav(song)
        scan_directory(self.db, self.music, "missing-ffprobe")
        build_embeddings(self.db, DiagnosticEmbedder())
        with self.db.connect() as conn:
            asset_id = conn.execute("SELECT id FROM asset").fetchone()[0]
        self.assertTrue(self.db.delete_asset_record(asset_id))
        self.assertTrue(song.is_file())
        with self.db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM asset").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM embedding").fetchone()[0], 0)

    def test_batch_delete_cascades_vectors_and_derived_data(self):
        first = self.music / "batch-one.wav"
        second = self.music / "batch-two.wav"
        write_wav(first)
        write_wav(second)
        scan_directory(self.db, self.music, "missing-ffprobe")
        build_embeddings(self.db, DiagnosticEmbedder())
        with self.db.connect() as conn:
            ids = [row[0] for row in conn.execute("SELECT id FROM asset ORDER BY id")]
            conn.executemany(
                """INSERT INTO asset_tag(asset_id,namespace,code,label,score,rank_no)
                VALUES (?,'mood','test','测试',1,1)""", [(asset_id,) for asset_id in ids],
            )
            conn.executemany(
                """INSERT INTO asset_tag_analysis_state
                (asset_id,scheme_code,engine,result_count,completed_at)
                VALUES (?,'recommended','local',0,?)""",
                [(asset_id, time.time()) for asset_id in ids],
            )
        self.assertEqual(self.db.delete_asset_records(ids), 2)
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())
        with self.db.connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM asset").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM embedding").fetchone()[0], 0)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM asset_tag").fetchone()[0], 0)
            self.assertEqual(conn.execute(
                "SELECT COUNT(*) FROM asset_tag_analysis_state"
            ).fetchone()[0], 0)

    def test_lexical_score(self):
        self.assertGreater(lexical_score("温暖 钢琴", "温暖钢琴主题.wav"), 0.9)
        self.assertEqual(lexical_score("紧张", "温暖钢琴.wav"), 0.0)

    def test_chinese_query_expansion(self):
        expanded = expand_chinese_query("克制的悲伤，钢琴独白")
        self.assertIn("restrained subtle", expanded)
        self.assertIn("piano", expanded)
        self.assertIn("background music for monologue", expanded)

    def test_waveform_requires_ffmpeg(self):
        song = self.music / "wave.wav"
        write_wav(song)
        with self.assertRaisesRegex(RuntimeError, "FFmpeg"):
            generate_peaks(song, "definitely-missing-ffmpeg")

    def test_tempo_estimator_detects_regular_pulses(self):
        sample_rate = 16000
        audio = np.zeros(sample_rate * 12, dtype=np.float32)
        for start in range(0, audio.size, sample_rate // 2):
            audio[start:start + 300] = np.hanning(300)
        bpm, confidence = estimate_tempo(audio, sample_rate)
        self.assertAlmostEqual(bpm, 120, delta=4)
        self.assertGreater(confidence, 0.08)

    def test_music_tags_keep_only_clear_main_instrument(self):
        tags = LocalAudioAnalyzer._mapped_tags(
            {"piano": 0.81, "violin": 0.52, "acoustic guitar": 0.10},
            "instrument", INSTRUMENTS, 0.35, 1, min_margin=0.08,
        )
        self.assertEqual([(tag["code"], tag["rank_no"]) for tag in tags], [("piano", 1)])

    def test_music_tags_abstain_when_main_instrument_is_ambiguous(self):
        tags = LocalAudioAnalyzer._mapped_tags(
            {"piano": 0.70, "violin": 0.66},
            "instrument", INSTRUMENTS, 0.35, 1, min_margin=0.08,
        )
        self.assertEqual(tags, [])

    def test_instrument_names_do_not_overlap_generic_guitar(self):
        tags = LocalAudioAnalyzer._mapped_tags(
            {"acoustic guitar": 0.78, "guitar": 0.20},
            "instrument", INSTRUMENTS, 0.35, 1, min_margin=0.08,
        )
        self.assertEqual(tags[0]["code"], "acoustic_guitar")

    def test_tag_classification_and_filter(self):
        vector = np.array([1.0, 0.0], dtype=np.float32)
        labels = {"mood": [
            ("warm", "温暖", np.array([1.0, 0.0], dtype=np.float32)),
            ("tense", "紧张", np.array([0.0, 1.0], dtype=np.float32)),
        ]}
        tagged = classify(vector, labels, keep=1)
        self.assertEqual(tagged[0]["code"], "warm")

        song = self.music / "tagged.wav"
        write_wav(song)
        scan_directory(self.db, self.music, "missing-ffprobe")
        embedder = DiagnosticEmbedder()
        build_embeddings(self.db, embedder)
        with self.db.connect() as conn:
            asset_id = conn.execute("SELECT id FROM asset").fetchone()[0]
            conn.execute(
                "INSERT INTO asset_tag VALUES (?, 'mood', 'warm', '温暖', 0.9, 1)",
                (asset_id,),
            )
        self.assertEqual(len(search_assets(
            self.db, embedder, tag_filters={"mood": "warm"}
        )), 1)
        self.assertEqual(len(search_assets(
            self.db, embedder, tag_filters={"mood": "tense"}
        )), 0)


if __name__ == "__main__":
    unittest.main()
