from __future__ import annotations

import json
import gc
import mimetypes
import re
import threading
import traceback
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .config import Settings
from .cloud_recommended import AliyunRecommendedClassifier
from .analysis import LocalAudioAnalyzer, analyze_effect_actions, analyze_library
from .db import Database
from .dedupe import prepare_effect_duplicates
from .embedder import Embedder, create_embedder
from .effect_tagging import auto_tag_effects, prepare_effect_tag_vectors
from .indexer import build_embeddings
from .scanner import scan_directory
from .search import search_assets
from .text_embedding import BgeTextEmbedder, TextEmbedder, build_text_embeddings
from .waveform import generate_peaks
from .taxonomy import (
    asset_nodes, create_custom_node, delete_custom_node, ensure_ucs,
    initialize_schemes, list_children, rename_custom_node, set_asset_node, set_asset_ucs,
    set_asset_recommended,
)


class JobStopped(Exception):
    pass


class AppState:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(settings.db_path)
        initialize_schemes(self.db, settings.data_dir.parent)
        self.embedder: Embedder | None = None
        self.embedder_key = ""
        self.text_embedder: TextEmbedder | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        try:
            saved = json.loads(self.db.get_setting("background_job", "{}") or "{}")
        except json.JSONDecodeError:
            saved = {}
        if saved.get("running"):
            saved.update({"running": False, "state": "interrupted",
                          "message": "上次任务中断，可点击继续建立检索库"})
        self.job: dict = saved or {
            "type": "idle", "running": False, "state": "idle",
            "progress": {}, "error": "",
        }
        self._persist_job()

    def get_text_embedder(self) -> TextEmbedder:
        with self.lock:
            if self.text_embedder is None:
                self.text_embedder = BgeTextEmbedder(self.settings.text_model)
            return self.text_embedder

    def refresh_text_embeddings(self, asset_ids: list[int] | None = None) -> None:
        if self.text_embedder is not None:
            build_text_embeddings(self.db, self.text_embedder, asset_ids)

    def get_embedder(self, provider: str) -> Embedder:
        model = self.settings.clap_model if provider == "clap" else "hash-v1"
        key = f"{provider}:{model}"
        with self.lock:
            if self.embedder is None or self.embedder_key != key:
                self.embedder = create_embedder(
                    provider, model, self.settings.ffmpeg, self.settings.ffprobe
                )
                self.embedder_key = key
            return self.embedder

    def release_embedder(self) -> None:
        with self.lock:
            self.embedder = None
            self.embedder_key = ""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def run_job(self, job_type: str, operation) -> None:
        with self.lock:
            if self.job["running"]:
                raise RuntimeError("已有后台任务正在运行")
            self.stop_event.clear()
            self.job = {"type": job_type, "running": True, "state": "running",
                        "progress": {}, "stages": {}, "error": "", "started_at": time.time()}
            self._persist_job()

        def target():
            try:
                operation(lambda value: self._progress(value))
                self.job["state"] = "completed"
                self.job["message"] = "任务已完成"
            except JobStopped:
                self.job["state"] = "stopped"
                self.job["message"] = "已安全停止，已完成进度将在下次继续"
            except Exception as exc:
                self.job["error"] = f"{exc}\n{traceback.format_exc(limit=2)}"
                self.job["state"] = "failed"
            finally:
                self.job["running"] = False
                self.job["finished_at"] = time.time()
                self._persist_job()

        threading.Thread(target=target, name=f"bgm-{job_type}", daemon=True).start()

    def _progress(self, value: dict) -> None:
        self.job["progress"] = value
        phase = value.get("phase")
        if phase:
            self.job.setdefault("stages", {})[phase] = value
        self.job["updated_at"] = time.time()
        self._persist_job()
        if self.stop_event.is_set():
            raise JobStopped()

    def request_stop(self) -> bool:
        if not self.job.get("running"):
            return False
        self.stop_event.set()
        self.job["state"] = "stopping"
        self.job["message"] = "正在完成当前小批次后安全停止"
        self._persist_job()
        return True

    def _persist_job(self) -> None:
        self.db.set_setting("background_job", json.dumps(self.job, ensure_ascii=False))


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")


def make_handler(state: AppState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BgmV2/0.1"

        def log_message(self, format: str, *args) -> None:
            return

        def send_json(self, value, status: int = 200) -> None:
            body = json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 1024 * 1024:
                raise ValueError("请求过大")
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/":
                    body = (Path(__file__).parent / "static" / "index.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/api/status":
                    self.send_json({
                        "counts": state.db.counts(),
                        "job": state.job,
                        "recommended_classifier": {
                            "provider": state.settings.recommended_provider,
                            "model": (state.settings.aliyun_model
                                      if state.settings.recommended_provider == "aliyun"
                                      else "local"),
                            "cloud_configured": bool(
                                state.settings.aliyun_api_key
                                and state.settings.aliyun_base_url
                            ),
                        },
                    })
                elif parsed.path == "/api/search":
                    self.handle_search(parse_qs(parsed.query))
                elif parsed.path == "/api/tag-tree":
                    query = parse_qs(parsed.query)
                    scheme = query.get("scheme", [""])[0]
                    parent = query.get("parent_id", [""])[0]
                    if scheme == "ucs":
                        ensure_ucs(state.db, state.settings.data_dir / "taxonomy")
                    self.send_json(list_children(
                        state.db, scheme, int(parent) if parent else None
                    ))
                elif parsed.path == "/api/effect-tags":
                    query = parse_qs(parsed.query)
                    asset_id = int(query.get("asset_id", ["0"])[0])
                    self.send_json({"nodes": asset_nodes(state.db, asset_id)})
                elif re.fullmatch(r"/api/audio/\d+", parsed.path):
                    self.handle_audio(int(parsed.path.rsplit("/", 1)[1]))
                elif re.fullmatch(r"/api/waveform/\d+", parsed.path):
                    self.handle_waveform(int(parsed.path.rsplit("/", 1)[1]))
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # Browsers routinely cancel an old audio Range request when the
                # user seeks, pauses, switches assets, or closes the detail pane.
                return
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)

        def do_POST(self) -> None:
            try:
                data = self.read_json()
                if self.path == "/api/select-folder":
                    try:
                        import tkinter as tk
                        from tkinter import filedialog
                    except ImportError as exc:
                        raise RuntimeError("当前 Python 环境不支持文件夹选择，请直接输入完整目录") from exc
                    window = tk.Tk()
                    try:
                        window.withdraw()
                        window.attributes("-topmost", True)
                        selected = filedialog.askdirectory(
                            parent=window, title="选择要扫描的素材文件夹", mustexist=True
                        )
                    finally:
                        window.destroy()
                    self.send_json({"path": selected or ""})
                elif self.path == "/api/scan":
                    root = Path(str(data.get("path", "")))
                    media_type = str(data.get("media_type", ""))
                    if media_type not in {"music", "effect"}:
                        raise ValueError("请选择素材类型：音乐或音效")
                    state.run_job("scan", lambda progress: scan_directory(
                        state.db, root, state.settings.ffprobe, progress, media_type
                    ))
                    self.send_json({"ok": True}, 202)
                elif self.path == "/api/index":
                    limit = data.get("limit")
                    item_limit = int(limit) if limit else None

                    def operation(progress):
                        progress({"phase": "duplicates", "message": "正在确认完全相同的音效文件"})
                        duplicate_summary = prepare_effect_duplicates(state.db)
                        progress({"phase": "duplicates", **duplicate_summary,
                                  "message": "完全相同音效过滤完成"})
                        progress({"phase": "vector", "message": "正在生成语义检索向量"})
                        embedder = state.get_embedder("clap")
                        build_embeddings(
                            state.db, embedder, item_limit,
                            lambda value: progress({"phase": "vector", **value}),
                        )
                        del embedder
                        state.release_embedder()
                        progress({"phase": "tags", "message": "正在分析音频标签"})
                        with state.db.connect() as conn:
                            pending_music = conn.execute(
                                """SELECT COUNT(*) FROM asset a LEFT JOIN audio_analysis x
                                ON x.asset_id=a.id WHERE a.missing=0 AND a.media_type='music'
                                AND (x.asset_id IS NULL OR x.status<>'ready')"""
                            ).fetchone()[0]
                            pending_effect = conn.execute(
                                """SELECT COUNT(*) FROM asset a LEFT JOIN effect_action_analysis x
                                ON x.asset_id=a.id WHERE a.missing=0 AND a.media_type='effect'
                                AND a.status<>'duplicate'
                                AND (x.asset_id IS NULL OR x.status<>'ready')"""
                            ).fetchone()[0]
                        if pending_music or pending_effect:
                            analyzer = LocalAudioAnalyzer(
                                state.settings.analysis_model, state.settings.ffmpeg
                            )
                            try:
                                analyze_library(
                                    state.db, analyzer, item_limit,
                                    lambda value: progress({"phase": "tags", **value}),
                                )
                                progress({
                                    "phase": "effect_actions",
                                    "message": (
                                        "正在生成本地音频描述并识别音效动作（AST）"
                                        f" · {analyzer.device.upper()} · 批次 {analyzer.batch_size}"
                                    ),
                                })
                                analyze_effect_actions(
                                    state.db, analyzer, item_limit,
                                    lambda value: progress({"phase": "effect_actions", **value}),
                                )
                            finally:
                                del analyzer
                                gc.collect()
                        else:
                            print("[本地音频分析] 已计算缓存全部命中，不加载AST模型", flush=True)
                            progress({"phase": "effect_actions", "queued": 0,
                                      "completed": 0, "skipped": state.db.counts()["effect"],
                                      "message": "本地音频分析已计算，全部跳过"})
                        progress({"phase": "warmup", "message": "正在重新加载检索模型"})
                        embedder = state.get_embedder("clap")
                        progress({"phase": "effect_tags", "message": "正在按动作交叉验证材质与音效标签"})
                        recommended_classifier = None
                        if state.settings.recommended_provider == "aliyun":
                            print(
                                "[推荐标签] 本次建库使用云端模型："
                                f"aliyun / {state.settings.aliyun_model}",
                                flush=True,
                            )
                            recommended_classifier = AliyunRecommendedClassifier(
                                state.db,
                                state.settings.aliyun_api_key,
                                state.settings.aliyun_base_url,
                                state.settings.aliyun_model,
                                state.settings.ffmpeg,
                                state.settings.cloud_timeout_seconds,
                                state.settings.cloud_min_confidence,
                                state.settings.cloud_max_concurrency,
                                state.settings.cloud_ucs_category_confidence,
                                state.settings.cloud_ucs_subcategory_confidence,
                            )
                        else:
                            print("[推荐标签] 本次建库使用本地模型", flush=True)
                        auto_tag_effects(
                            state.db, state.settings.data_dir / "taxonomy", embedder, item_limit,
                            progress=lambda value: progress({"phase": "effect_tags", **value}),
                            text_embedder=state.get_text_embedder(),
                            recommended_classifier=recommended_classifier,
                            cloud_fallback_local=state.settings.cloud_fallback_local,
                        )
                        progress({"phase": "text_vectors", "message": "正在同步文件名与标签语义向量"})
                        build_text_embeddings(
                            state.db, state.get_text_embedder(),
                            progress=lambda value: progress({"phase": "text_vectors", **value}),
                        )

                    state.run_job("build", operation)
                    self.send_json({"ok": True}, 202)
                elif self.path == "/api/index-preview":
                    self.send_json(prepare_effect_duplicates(state.db))
                elif self.path == "/api/job/stop":
                    self.send_json({"ok": True, "stopping": state.request_stop()})
                elif self.path == "/api/custom-tag":
                    action = str(data.get("action", "create"))
                    if action == "create":
                        node = create_custom_node(
                            state.db, str(data.get("label", "")),
                            int(data["parent_id"]) if data.get("parent_id") else None,
                        )
                        self.send_json({"ok": True, "node": node})
                    elif action == "rename":
                        rename_custom_node(state.db, int(data["node_id"]), str(data.get("label", "")))
                        state.refresh_text_embeddings()
                        self.send_json({"ok": True})
                    elif action == "delete":
                        delete_custom_node(state.db, int(data["node_id"]))
                        state.refresh_text_embeddings()
                        self.send_json({"ok": True})
                    else:
                        raise ValueError("未知的自定义标签操作")
                elif self.path == "/api/effect-tag":
                    set_asset_node(
                        state.db, int(data["asset_id"]), int(data["node_id"]),
                        bool(data.get("assigned", True)),
                    )
                    state.refresh_text_embeddings([int(data["asset_id"])])
                    self.send_json({"ok": True})
                elif self.path == "/api/ucs-tag":
                    node_id = int(data["node_id"]) if data.get("node_id") else None
                    set_asset_ucs(state.db, int(data["asset_id"]), node_id)
                    state.refresh_text_embeddings([int(data["asset_id"])])
                    self.send_json({"ok": True})
                elif self.path == "/api/recommended-tag":
                    set_asset_recommended(
                        state.db, int(data["asset_id"]), int(data["node_id"]),
                        bool(data.get("assigned", True)),
                    )
                    state.refresh_text_embeddings([int(data["asset_id"])])
                    self.send_json({"ok": True})
                elif self.path == "/api/asset/delete":
                    deleted = state.db.delete_asset_record(int(data["asset_id"]))
                    if not deleted:
                        raise ValueError("素材记录不存在")
                    self.send_json({"ok": True, "source_file_deleted": False})
                elif self.path == "/api/assets/delete":
                    raw_ids = data.get("asset_ids", [])
                    if not isinstance(raw_ids, list):
                        raise ValueError("asset_ids 必须是数组")
                    asset_ids = [int(value) for value in raw_ids]
                    if not asset_ids:
                        raise ValueError("请至少选择一条素材记录")
                    if len(asset_ids) > 500:
                        raise ValueError("一次最多删除 500 条素材记录")
                    deleted = state.db.delete_asset_records(asset_ids)
                    self.send_json({"ok": True, "deleted": deleted,
                                    "source_files_deleted": False})
                else:
                    self.send_error(404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)

        def handle_search(self, query_params: dict[str, list[str]]) -> None:
            get = lambda key, default="": query_params.get(key, [default])[0]
            embedder = state.get_embedder("clap")
            reference = None
            reference_vector = None
            reference_id = get("reference_id")
            if reference_id:
                with state.db.connect() as conn:
                    row = conn.execute(
                        """SELECT a.path, e.vector, e.dimension FROM asset a
                        LEFT JOIN embedding e ON e.asset_id=a.id AND e.provider=? AND e.model=?
                        WHERE a.id=? AND a.missing=0""",
                        (embedder.provider, embedder.model_name, int(reference_id)),
                    ).fetchone()
                if row is None:
                    raise ValueError("参考歌曲不存在")
                if row["vector"] is not None:
                    import numpy as np
                    reference_vector = np.frombuffer(
                        row["vector"], dtype="<f4", count=row["dimension"]
                    )
                else:
                    reference = Path(row["path"])
            min_seconds = get("min_seconds")
            max_seconds = get("max_seconds")
            min_bpm = get("min_bpm")
            max_bpm = get("max_bpm")
            tag_node_ids = [int(value) for value in get("tag_node_ids").split(",") if value]
            page = max(1, int(get("page", "1")))
            page_size = max(1, min(50, int(get("limit", "20"))))
            items = search_assets(
                state.db,
                embedder,
                query=get("q"),
                reference_path=reference,
                reference_vector=reference_vector,
                extension=get("extension"),
                min_duration_ms=int(float(min_seconds) * 1000) if min_seconds else None,
                max_duration_ms=int(float(max_seconds) * 1000) if max_seconds else None,
                media_type=get("media_type", "music"),
                min_bpm=float(min_bpm) if min_bpm else None,
                max_bpm=float(max_bpm) if max_bpm else None,
                tag_node_ids=tag_node_ids,
                tag_filters={
                    "instrument": get("instrument"), "mood": get("mood"),
                    "rhythm": get("rhythm"), "style": get("style"),
                },
                limit=page_size + 1,
                offset=(page - 1) * page_size,
                text_embedder=state.get_text_embedder(),
            )
            self.send_json({"items": items[:page_size], "page": page,
                            "has_more": len(items) > page_size})

        def handle_audio(self, asset_id: int) -> None:
            with state.db.connect() as conn:
                row = conn.execute("SELECT path FROM asset WHERE id=? AND missing=0", (asset_id,)).fetchone()
            if row is None:
                self.send_error(404)
                return
            path = Path(row["path"])
            if not path.is_file():
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            range_header = self.headers.get("Range")
            if range_header:
                match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if not match:
                    self.send_error(416)
                    return
                if match.group(1):
                    start = int(match.group(1))
                if match.group(2):
                    end = min(int(match.group(2)), size - 1)
                if start > end or start >= size:
                    self.send_error(416)
                    return
            length = end - start + 1
            self.send_response(HTTPStatus.PARTIAL_CONTENT if range_header else HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if range_header:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as stream:
                stream.seek(start)
                remaining = length
                while remaining:
                    chunk = stream.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)

        def handle_waveform(self, asset_id: int) -> None:
            with state.db.connect() as conn:
                row = conn.execute(
                    """SELECT a.path, a.modified_ns, w.peaks, w.points,
                    w.source_modified_ns FROM asset a
                    LEFT JOIN waveform w ON w.asset_id=a.id
                    WHERE a.id=? AND a.missing=0""",
                    (asset_id,),
                ).fetchone()
            if row is None:
                raise ValueError("音频不存在")
            if row["peaks"] is not None and row["source_modified_ns"] == row["modified_ns"]:
                peaks = np.frombuffer(row["peaks"], dtype="<f4", count=row["points"])
            else:
                peaks = generate_peaks(Path(row["path"]), state.settings.ffmpeg)
                with state.db.connect() as conn:
                    conn.execute(
                        """INSERT OR REPLACE INTO waveform
                        (asset_id, source_modified_ns, points, peaks, created_at)
                        VALUES (?, ?, ?, ?, ?)""",
                        (asset_id, row["modified_ns"], peaks.size,
                         peaks.astype("<f4").tobytes(), time.time()),
                    )
            self.send_json({"peaks": [round(float(value), 4) for value in peaks]})

    return Handler


def serve(settings: Settings) -> None:
    settings.ensure_dirs()
    state = AppState(settings)
    if settings.recommended_provider == "aliyun":
        print(
            "[推荐标签配置] provider=aliyun，"
            f"model={settings.aliyun_model}，"
            f"API Key={'已配置' if settings.aliyun_api_key else '未配置'}，"
            f"Base URL={'已配置' if settings.aliyun_base_url else '未配置'}，"
            f"并发上限={settings.cloud_max_concurrency}，"
            f"失败回退本地={'开启' if settings.cloud_fallback_local else '关闭'}",
            flush=True,
        )
    else:
        print("[推荐标签配置] provider=local，本地模型，不会调用云端", flush=True)
    print("正在加载 BGE 中文文本检索模型，请稍候……")
    text_embedder = state.get_text_embedder()
    print("正在加载 CLAP 检索模型，请稍候……")
    embedder = state.get_embedder("clap")
    print("CLAP 检索模型加载完成。")

    last_message = {"value": ""}
    def show_init_progress(value: dict) -> None:
        message = value.get("message")
        if not message:
            total = int(value.get("total", 0))
            cached = int(value.get("cached", 0))
            generated = int(value.get("generated", 0))
            message = f"标签向量初始化 {cached + generated}/{total}"
        if message != last_message["value"]:
            print(message)
            last_message["value"] = message

    print("正在校验标签向量初始化缓存……")
    tag_result = prepare_effect_tag_vectors(
        state.db, settings.data_dir / "taxonomy", embedder, text_embedder,
        show_init_progress,
    )
    print(
        f"标签向量初始化完成：CLAP {tag_result['clap_total']}，"
        f"BGE {tag_result['bge_total']}。缓存有效时不会重复生成。"
    )
    text_result = build_text_embeddings(state.db, text_embedder)
    print(f"素材文本向量就绪：新增/更新 {text_result['generated']}，缓存 {text_result['cached']}。")
    server = ThreadingHTTPServer((settings.host, settings.port), make_handler(state))
    print(f"BGM V2 检索验证版：http://{settings.host}:{settings.port}")
    print("按 Ctrl+C 停止。源音频不会被修改。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
