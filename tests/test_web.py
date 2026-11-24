from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from bgm_v2.config import Settings
from bgm_v2.web import AppState, make_handler


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        data = Path(self.temp.name)
        settings = Settings(data, data / "test.db", "127.0.0.1", 0, "ffmpeg", "ffprobe", "unused", 10)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(AppState(settings)))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def test_home_and_status(self):
        with urllib.request.urlopen(self.base + "/") as response:
            page = response.read().decode("utf-8")
            self.assertIn("BGM 本地检索实验室", page)
            self.assertIn("请选择素材类型", page)
            self.assertIn("03 · 筛选条件", page)
            self.assertIn('id="style"', page)
            self.assertIn('class="result-browser detail-closed"', page)
            self.assertIn('id="resultTableHead"', page)
            self.assertIn("自定义标签</span><span>时长", page)
            self.assertIn('id="detailPane"', page)
            self.assertIn('class="detail-pane hidden"', page)
            self.assertIn('class="result-browser detail-closed"', page)
            self.assertIn("搜索文件名、标签", page)
            self.assertIn("function showDetail", page)
            self.assertIn("function closeDetail", page)
            self.assertIn("function buildTagTree", page)
            self.assertIn("detailTagSection(x,'ucs','UCS')", page)
            self.assertIn("function batchDeleteRecords", page)
            self.assertIn("/api/assets/delete", page)
            self.assertIn("!event.target.closest('#detailPane')", page)
            self.assertIn("function chooseFolder", page)
            self.assertNotIn('id="roots"', page)
        with urllib.request.urlopen(self.base + "/api/status") as response:
            data = json.load(response)
        self.assertEqual(data["counts"]["total"], 0)
        self.assertFalse(data["job"]["running"])
        self.assertNotIn("roots", data)


if __name__ == "__main__":
    unittest.main()
