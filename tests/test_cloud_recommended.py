from __future__ import annotations

import json
import tempfile
import unittest
import wave
import urllib.error
from email.message import Message
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from bgm_v2.cloud_recommended import AliyunRecommendedClassifier
from bgm_v2.db import Database
from bgm_v2.scanner import scan_directory


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(np.zeros(16000, dtype="<i2").tobytes())


class FakeResponse:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return self.body


class CloudRecommendedTests(unittest.TestCase):
    def test_http_error_detail_includes_status_request_id_and_body(self):
        headers = Message()
        headers["x-request-id"] = "req-123"
        error = urllib.error.HTTPError(
            "https://example.invalid", 429, "Too Many Requests", headers,
            BytesIO(b'{"code":"Throttling","message":"rate limit exceeded"}'),
        )
        detail = AliyunRecommendedClassifier._http_error_detail(error)
        self.assertIn("HTTP 429", detail)
        self.assertIn("req-123", detail)
        self.assertIn("Throttling", detail)

    def test_aliyun_result_is_validated_and_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "effect.wav"
            write_wav(audio)
            db = Database(root / "library.db")
            scan_directory(db, root, "missing", media_type="effect")
            with db.connect() as conn:
                asset = dict(conn.execute("SELECT * FROM asset").fetchone())
            candidates = [
                {"id": 10, "path": "Transition > Whoosh", "description": "Fast passing sound",
                 "examples_json": "[]"},
                {"id": 11, "path": "Ambience > City", "description": "Urban ambience",
                 "examples_json": "[]"},
            ]
            response = FakeResponse({
                "choices": [{"message": {"content": json.dumps({
                    "abstain": False,
                    "labels": [
                        {"node_id": 10, "confidence": 0.91, "evidence": "fast passing sound"},
                        {"node_id": 999, "confidence": 0.99, "evidence": "invalid node"},
                        {"node_id": 11, "confidence": 0.40, "evidence": "low confidence"},
                    ],
                })}}],
            })
            classifier = AliyunRecommendedClassifier(
                db, "secret", "https://example.invalid/v1", "qwen-test",
                "missing", min_confidence=0.78,
            )
            with patch("urllib.request.urlopen", return_value=response) as request:
                first = classifier.classify(asset, candidates)
                second = classifier.classify(asset, candidates)
            self.assertEqual(first, second)
            self.assertEqual([item["node_id"] for item in first], [10])
            self.assertEqual(request.call_count, 1)
            sent = json.loads(request.call_args.args[0].data.decode("utf-8"))
            self.assertEqual(sent["model"], "qwen-test")
            self.assertTrue(sent["messages"][0]["content"][0]["input_audio"]["data"].startswith(
                "data:audio/wav;base64,"
            ))

    def test_ucs_is_classified_category_then_subcategory_and_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = root / "effect.wav"
            write_wav(audio)
            db = Database(root / "library.db")
            scan_directory(db, root, "missing", media_type="effect")
            with db.connect() as conn:
                asset = dict(conn.execute("SELECT * FROM asset").fetchone())
            candidates = [
                {"id": 21, "code": "DOOROpen", "parent_id": 2,
                 "path": "DOORS > OPEN", "label": "OPEN",
                 "description": "Door opening", "examples_json": '["open door"]'},
                {"id": 22, "code": "DOORClose", "parent_id": 2,
                 "path": "DOORS > CLOSE", "label": "CLOSE",
                 "description": "Door closing", "examples_json": '["close door"]'},
                {"id": 31, "code": "ANMLDog", "parent_id": 3,
                 "path": "ANIMALS > DOG", "label": "DOG",
                 "description": "Dog sounds", "examples_json": "[]"},
            ]
            category_response = FakeResponse({
                "choices": [{"message": {"content": json.dumps({
                    "abstain": False, "category": "DOORS", "confidence": 0.93,
                    "evidence": "door movement",
                })}}],
            })
            child_response = FakeResponse({
                "choices": [{"message": {"content": json.dumps({
                    "abstain": False, "node_id": 22, "confidence": 0.88,
                    "evidence": "a door closes firmly",
                })}}],
            })
            classifier = AliyunRecommendedClassifier(
                db, "secret", "https://example.invalid/v1", "qwen-test", "missing",
            )
            with patch("urllib.request.urlopen",
                       side_effect=[category_response, child_response]) as request:
                first = classifier.classify_ucs(asset, candidates)
                second = classifier.classify_ucs(asset, candidates)
            self.assertEqual(first, second)
            self.assertEqual(first["node_id"], 22)
            self.assertEqual(first["category"], "DOORS")
            self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
