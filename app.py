"""
Trauma Behaviour Video Annotator — backend.

Local annotation server. Each video is a folder of ordered frames under --data.
  - Open a folder: every video in it (and its sub-folders) is extracted in sequence
  - Upload a single video / frames / zip, import annotations JSON or COCO
  - Trauma / No-Trauma video label, multi-select frame behaviours, comments, context
  - Person-of-interest boxes: manual, YOLO detect, Auto-VIP (whole video), track forward
  - Exports: per-video JSON + COCO, merged dataset JSON and metadata CSV/XLSX

Run:  py app.py  [--data ./data] [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import vip

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

BEHAVIOURS = ["flashback", "avoidance", "negative_emotion", "hyper_arousal", "normal"]
VIDEO_LABELS = ["trauma", "no_trauma"]
OBJECT_LABELS = ["person_of_interest", "person", "other"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".mpg", ".mpeg"}
CONTEXT_NAMES = ["context.txt", "scenario.txt", "background.txt"]
ANNOTATION_FILE = "annotations.json"
META_FILE = "meta.json"
COCO_SUFFIX = "_coco.json"
UNDO_FILE = "annotations.undo.json"
EXPORT_DIRNAME = "_exports"

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DATA_DIR = (APP_DIR / "data").resolve()

_obj_id_seq = itertools.count(1)


def set_data_dir(path: str | Path) -> Path:
    global DATA_DIR
    DATA_DIR = Path(path).resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


# ----------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------

def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def is_safe_name(name: str) -> bool:
    return name not in ("", ".", "..") and "/" not in name and "\\" not in name


def ascii_id(text: str) -> str:
    """Folder-safe ASCII id. OpenCV and some tools break on non-ASCII paths on Windows."""
    s = re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_-")
    s = re.sub(r"_{2,}", "_", s)
    return s[:60].strip("_-")


def make_video_id(rel: Path | str) -> str:
    rel = Path(rel)
    parts = [*rel.parent.parts[-1:], rel.stem]
    base = ascii_id("_".join(parts))
    if len(re.sub(r"[^A-Za-z0-9]", "", base)) < 3:
        base = f"video_{hashlib.sha1(str(rel).encode('utf-8')).hexdigest()[:8]}"
    return base


def unique_video_id(base: str) -> str:
    candidate, n = base, 2
    while (DATA_DIR / candidate).exists():
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            # Windows: a concurrent reader briefly holds the file
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def attachment_headers(filename: str) -> dict:
    fallback = filename.encode("ascii", "ignore").decode() or "download.json"
    return {
        "Content-Disposition": f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"
    }


def json_download(payload: Any, filename: str) -> Response:
    body = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    return Response(body, media_type="application/json", headers=attachment_headers(filename))


_locks: dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()


def vlock(vid: str) -> threading.RLock:
    """Serialises load→modify→save per video so concurrent requests never drop edits."""
    with _locks_guard:
        return _locks.setdefault(vid, threading.RLock())


# ----------------------------------------------------------------------------
# Video folders, frames, metadata
# ----------------------------------------------------------------------------

def video_dir(vid: str, must_exist: bool = True) -> Path:
    if not is_safe_name(vid) or vid.startswith(("_", ".")):
        raise HTTPException(400, "Invalid video id")
    d = DATA_DIR / vid
    if must_exist and not d.is_dir():
        raise HTTPException(404, f"Video '{vid}' not found")
    return d


def list_frames(d: Path) -> list[str]:
    if not d.is_dir():
        return []
    names = [p.name for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    return sorted(names, key=natural_key)


def read_meta(vid: str) -> dict:
    return read_json(DATA_DIR / vid / META_FILE, {})


def write_meta(vid: str, meta: dict) -> dict:
    d = DATA_DIR / vid
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_text(d / META_FILE, json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def update_meta(vid: str, **fields) -> dict:
    with vlock(vid):
        meta = read_meta(vid)
        meta.update(fields)
        return write_meta(vid, meta)


def library_dirs() -> list[Path]:
    if not DATA_DIR.is_dir():
        return []
    return [p for p in DATA_DIR.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))]


def read_context_file(d: Path) -> str:
    for name in CONTEXT_NAMES:
        p = d / name
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
    return ""


def label_from_path(path: str) -> str | None:
    p = path.lower().replace("_", " ").replace("-", " ")
    if re.search(r"\b(no|non|without)\s*trauma", p):
        return "no_trauma"
    if "trauma" in p:
        return "trauma"
    return None


def ids_from_path(path: str) -> tuple[str, str]:
    """Pick P001 / S001 style participant and session ids out of a path, if present."""
    def find(letter: str) -> str:
        m = re.search(rf"(?<![A-Za-z0-9]){letter}[_\-\s]?(\d{{1,4}})(?![0-9])", path, re.IGNORECASE)
        return f"{letter.upper()}{int(m.group(1)):03d}" if m else ""
    return find("P"), find("S")


# ----------------------------------------------------------------------------
# Annotation model
# ----------------------------------------------------------------------------

def empty_slot() -> dict:
    return {"behaviours": [], "comment": "", "bbox": None, "objects": []}


def new_obj_id(prefix: str = "obj") -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{next(_obj_id_seq)}_{uuid.uuid4().hex[:6]}"


def is_legacy_box(o: dict) -> bool:
    """Boxes converted from the old single-bbox format (server ids "obj_…")."""
    return o.get("source") == "legacy" or (o.get("source") == "manual" and str(o.get("id", "")).startswith("obj_"))


def is_user_box(o: dict) -> bool:
    return o.get("source") == "manual" and not is_legacy_box(o)


def is_locked_poi(o: dict) -> bool:
    """A POI the user chose (or drew/moved); automatic POI selection must not replace it."""
    return bool(o.get("is_poi")) and (bool(o.get("poi_locked")) or is_user_box(o))


def clean_behaviours(values) -> list[str]:
    out = list(dict.fromkeys(b for b in (values or []) if b in BEHAVIOURS))
    if "normal" in out and len(out) > 1:
        out = [b for b in out if b != "normal"]
    return out


def normalize_object(obj: dict) -> dict | None:
    if not isinstance(obj, dict):
        return None
    bb = obj.get("bbox") or obj.get("box")
    if not bb or len(bb) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bb)
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    label = obj.get("label") if obj.get("label") in OBJECT_LABELS else "person"
    return {
        "id": str(obj.get("id") or new_obj_id()),
        "bbox": [x1, y1, x2, y2],
        "label": label,
        "behaviours": clean_behaviours(obj.get("behaviours")),
        "confirmed": bool(obj.get("confirmed", True)),
        "source": str(obj.get("source") or "manual"),
        "conf": float(obj["conf"]) if obj.get("conf") is not None else None,
        "is_poi": bool(obj.get("is_poi", False)) or label == "person_of_interest",
        "poi_locked": bool(obj.get("poi_locked", False)),
    }


def sync_poi(slot: dict) -> dict:
    """Exactly one POI per frame; its box is mirrored to slot.bbox and it carries the frame behaviours."""
    objects = slot.get("objects") or []
    seen: set[str] = set()
    for o in objects:
        if o["id"] in seen:
            o["id"] = new_obj_id()
        seen.add(o["id"])

    poi = next((o for o in objects if o.get("is_poi")), None)
    for o in objects:
        o["is_poi"] = o is poi
        o["poi_locked"] = o is poi and bool(o.get("poi_locked"))
        if o is poi:
            o["label"] = "person_of_interest"
            o["confirmed"] = True
        elif o.get("label") == "person_of_interest":
            o["label"] = "person"

    if poi is not None:
        if not slot["behaviours"] and poi.get("behaviours"):
            slot["behaviours"] = clean_behaviours(poi["behaviours"])  # legacy per-object labels
        poi["behaviours"] = list(slot["behaviours"])
        slot["bbox"] = list(poi["bbox"])
    else:
        slot["bbox"] = None
    slot["objects"] = objects
    return slot


def normalize_slot(slot: Any) -> dict:
    if not isinstance(slot, dict):
        return empty_slot()
    behaviours = slot.get("behaviours")
    if behaviours is None:
        legacy = slot.get("behaviour")
        behaviours = [legacy] if isinstance(legacy, str) else (legacy or [])
    out = {
        "behaviours": clean_behaviours(behaviours),
        "comment": str(slot.get("comment") or ""),
        "bbox": None,
        "objects": [],
    }
    objects = [n for n in (normalize_object(o) for o in slot.get("objects") or []) if n]
    legacy_box = slot.get("bbox")
    if not objects and "objects" not in slot and legacy_box and len(legacy_box) == 4:
        objects.append(normalize_object({"bbox": legacy_box, "is_poi": True, "source": "legacy"}))
    out["objects"] = objects
    return sync_poi(out)


def annotation_path(d: Path) -> Path:
    return d / ANNOTATION_FILE


def load_annotations(vid: str) -> dict:
    d = video_dir(vid)
    frames = list_frames(d)
    data = read_json(annotation_path(d), {})
    if not isinstance(data, dict):
        data = {}
    data["video_id"] = vid
    data.setdefault("video_label", None)
    data.setdefault("video_comment", "")
    data.setdefault("participant_id", "")
    data.setdefault("session_id", "")
    if "context" not in data:
        data["context"] = read_context_file(d)
    data.setdefault("frame_dims", {})
    raw_frames = data.get("frames") or {}
    data["frames"] = {name: normalize_slot(raw_frames.get(name)) for name in frames}
    # keep slots for frames that vanished (e.g. partial re-extract) so nothing is silently lost
    for name, slot in raw_frames.items():
        if name not in data["frames"]:
            data["frames"][name] = normalize_slot(slot)
    data["_frame_order"] = frames
    return data


def save_annotations(vid: str, data: dict) -> None:
    d = video_dir(vid)
    out = {k: v for k, v in data.items() if not k.startswith("_")}
    atomic_write_text(annotation_path(d), json.dumps(out, indent=2, ensure_ascii=False))
    _stats_cache.pop(vid, None)


def persist(vid: str, data: dict, coco_now: bool = False) -> None:
    save_annotations(vid, data)
    if coco_now:
        write_coco_file(vid, data)
    else:
        mark_coco_dirty(vid)


def snapshot_for_undo(vid: str, data: dict, frame_names: list[str], action: str) -> None:
    snap = {
        "video_id": vid,
        "action": action,
        "ts": time.time(),
        "frames": {n: data["frames"].get(n, empty_slot()) for n in frame_names},
    }
    atomic_write_text(video_dir(vid) / UNDO_FILE, json.dumps(snap))


# ----------------------------------------------------------------------------
# COCO
# ----------------------------------------------------------------------------

def coco_path(d: Path, vid: str) -> Path:
    return d / f"{vid}{COCO_SUFFIX}"


def frame_size(vid: str, name: str, data: dict, meta: dict) -> list[int]:
    dims = data.setdefault("frame_dims", {})
    if isinstance(dims.get(name), list) and len(dims[name]) == 2:
        return dims[name]
    if meta.get("width") and meta.get("height"):
        return [int(meta["width"]), int(meta["height"])]
    try:
        with Image.open(DATA_DIR / vid / name) as im:
            wh = [int(im.width), int(im.height)]
    except Exception:
        wh = [0, 0]
    dims[name] = wh
    return wh


def build_coco(vid: str, data: dict) -> dict:
    meta = read_meta(vid)
    frames = data.get("_frame_order") or list_frames(video_dir(vid))
    cat_map = {name: i + 1 for i, name in enumerate(OBJECT_LABELS)}
    images, annotations = [], []
    ann_id = 1
    for i, name in enumerate(frames):
        w, h = frame_size(vid, name, data, meta)
        slot = data["frames"].get(name) or empty_slot()
        behaviours = slot.get("behaviours") or []
        images.append({
            "id": i + 1,
            "file_name": name,
            "width": w,
            "height": h,
            "frame_index": i,
            "behaviours": behaviours,
            "behaviour": behaviours[0] if len(behaviours) == 1 else None,
            "comment": slot.get("comment", ""),
        })
        for obj in slot.get("objects") or []:
            if not (obj.get("confirmed") or obj.get("is_poi")):
                continue
            x1, y1, x2, y2 = obj["bbox"]
            obj_beh = obj.get("behaviours") or []
            annotations.append({
                "id": ann_id,
                "image_id": i + 1,
                "category_id": cat_map.get(obj.get("label"), cat_map["other"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                "iscrowd": 0,
                "behaviours": obj_beh,
                "behaviour": obj_beh[0] if len(obj_beh) == 1 else None,
                "object_id": obj.get("id"),
                "is_poi": bool(obj.get("is_poi")),
                "poi_locked": bool(obj.get("poi_locked")),
                "score": obj.get("conf"),
                "source": obj.get("source"),
            })
            ann_id += 1
    return {
        "info": {
            "description": f"Annotated objects for {vid}",
            "video_id": vid,
            "filename": meta.get("source_name") or vid,
            "participant_id": data.get("participant_id", ""),
            "session_id": data.get("session_id", ""),
            "video_label": data.get("video_label"),
            "video_comment": data.get("video_comment", ""),
            "context": data.get("context", ""),
            "fps": meta.get("fps"),
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in sorted(cat_map.items(), key=lambda x: x[1])],
    }


def write_coco_file(vid: str, data: dict | None = None) -> Path:
    with vlock(vid):
        if data is None:
            data = load_annotations(vid)
        coco = build_coco(vid, data)
        path = coco_path(video_dir(vid), vid)
        atomic_write_text(path, json.dumps(coco, indent=2, ensure_ascii=False))
        return path


_coco_dirty: set[str] = set()
_coco_cv = threading.Condition()


def mark_coco_dirty(vid: str) -> None:
    with _coco_cv:
        _coco_dirty.add(vid)
        _coco_cv.notify()


def _coco_writer_loop() -> None:
    """Keeps <video>_coco.json in step with annotations.json without rebuilding on every click."""
    while True:
        with _coco_cv:
            while not _coco_dirty:
                _coco_cv.wait()
        time.sleep(1.5)  # coalesce bursts of edits
        with _coco_cv:
            batch = list(_coco_dirty)
            _coco_dirty.clear()
        for vid in batch:
            try:
                write_coco_file(vid)
            except Exception as e:  # noqa: BLE001
                print(f"[coco] {vid}: {e}")


def flush_coco() -> None:
    with _coco_cv:
        batch = list(_coco_dirty)
        _coco_dirty.clear()
    for vid in batch:
        try:
            write_coco_file(vid)
        except Exception as e:  # noqa: BLE001
            print(f"[coco] {vid}: {e}")


# ----------------------------------------------------------------------------
# Library stats
# ----------------------------------------------------------------------------

_stats_cache: dict[str, tuple] = {}


def video_stats(vid: str) -> dict:
    d = DATA_DIR / vid
    frames = list_frames(d)
    ann = annotation_path(d)
    mtime = ann.stat().st_mtime_ns if ann.is_file() else 0
    key = (mtime, len(frames))
    cached = _stats_cache.get(vid)
    if cached and cached[0] == key:
        return cached[1]
    labelled = boxed = 0
    video_label = None
    raw = read_json(ann, {}) if mtime else {}
    slots = raw.get("frames") or {}
    for name in frames:
        s = slots.get(name) or {}
        if s.get("behaviours") or s.get("behaviour"):
            labelled += 1
        if s.get("bbox"):
            boxed += 1
    video_label = raw.get("video_label")
    stats = {
        "num_frames": len(frames),
        "labelled": labelled,
        "boxed": boxed,
        "video_label": video_label,
        "has_annotations": bool(mtime),
    }
    _stats_cache[vid] = (key, stats)
    return stats


def video_summary(vid: str) -> dict:
    meta = read_meta(vid)
    stats = video_stats(vid)
    status = meta.get("status") or ("ready" if stats["num_frames"] else "empty")
    if status == "ready" and not stats["num_frames"]:
        status = "empty"
    return {
        "id": vid,
        "name": meta.get("source_name") or vid,
        "group": meta.get("group") or "Uploads",
        "order": meta.get("order", 10**6),
        "status": status,
        "error": meta.get("error"),
        "fps": meta.get("fps"),
        **stats,
    }


# ----------------------------------------------------------------------------
# Frame extraction
# ----------------------------------------------------------------------------

def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def probe_video(path: Path) -> dict:
    info = {"fps": None, "width": None, "height": None, "frames": None, "duration": None}
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or None
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
            info.update(
                fps=round(fps, 3) if fps else None,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None,
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None,
                frames=n,
                duration=(n / fps) if (n and fps) else None,
            )
        cap.release()
    except Exception:
        pass
    if info["fps"] is None and shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                capture_output=True, text=True, timeout=60,
            )
            j = json.loads(out.stdout or "{}")
            st = (j.get("streams") or [{}])[0]
            num, _, den = (st.get("r_frame_rate") or "0/1").partition("/")
            fps = float(num) / float(den or 1) if float(den or 1) else None
            dur = float((j.get("format") or {}).get("duration") or 0) or None
            info.update(fps=round(fps, 3) if fps else None, width=st.get("width"),
                        height=st.get("height"), duration=dur,
                        frames=int(dur * fps) if (dur and fps) else None)
        except Exception:
            pass
    return info


def clear_frames(d: Path) -> None:
    for p in d.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            try:
                p.unlink()
            except OSError:
                pass


class JobCancelled(Exception):
    pass


def _extract_ffmpeg(src: Path, out_dir: Path, fps: float | None, duration: float | None,
                    progress: Callable[[float], None] | None) -> int:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(src)]
    if fps and fps > 0:
        cmd += ["-vf", f"fps={fps}"]
    cmd += ["-q:v", "2", "-start_number", "1", str(out_dir / "frame_%06d.jpg"),
            "-progress", "pipe:1", "-nostats"]
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True, errors="replace")
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                    try:
                        t = int(line.split("=", 1)[1]) / 1e6
                    except ValueError:
                        continue
                    if progress and duration:
                        progress(min(0.99, t / duration))
            proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        if proc.returncode != 0:
            err.seek(0)
            raise RuntimeError(err.read().decode("utf-8", "replace")[-600:] or "ffmpeg failed")
    return len(list_frames(out_dir))


def _extract_opencv(src: Path, out_dir: Path, fps: float | None,
                    progress: Callable[[float], None] | None) -> int:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open the video")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = max(1.0, src_fps / fps) if fps and fps > 0 else 1.0
    idx = written = 0
    next_take = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx >= next_take:
                written += 1
                ok_enc, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                if ok_enc:
                    (out_dir / f"frame_{written:06d}.jpg").write_bytes(buf.tobytes())
                next_take += interval
            idx += 1
            if progress and total and idx % 10 == 0:
                progress(min(0.99, idx / total))
    finally:
        cap.release()
    return written


def extract_frames(src: Path, out_dir: Path, fps: float | None = None,
                   progress: Callable[[float], None] | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(src)
    clear_frames(out_dir)
    n, engine, last_err = 0, None, None
    if ffmpeg_available():
        try:
            n = _extract_ffmpeg(src, out_dir, fps, info["duration"], progress)
            engine = "ffmpeg"
        except JobCancelled:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[extract] ffmpeg failed, trying OpenCV: {e}")
            clear_frames(out_dir)
    if n == 0 and opencv_available():
        n = _extract_opencv(src, out_dir, fps, progress)
        engine = "opencv"
    if n == 0:
        raise RuntimeError(f"No frames extracted ({last_err or 'install ffmpeg or opencv-python'})")
    return {
        "num_frames": n,
        "fps": float(fps) if fps and fps > 0 else info["fps"],
        "src_fps": info["fps"],
        "width": info["width"],
        "height": info["height"],
        "duration": info["duration"],
        "engine": engine,
    }


# ----------------------------------------------------------------------------
# Background jobs
# ----------------------------------------------------------------------------

_jobs: dict[str, dict] = {}
_job_queues: dict[str, queue.Queue] = {"import": queue.Queue(), "ml": queue.Queue()}
_workers_started = False


def submit_job(kind: str, lane: str, title: str, fn: Callable[[dict], Any], video_id: str | None = None) -> dict:
    job = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "lane": lane,
        "title": title,
        "video_id": video_id,
        "status": "queued",
        "progress": 0.0,
        "message": "Waiting…",
        "result": None,
        "error": None,
        "created": time.time(),
        "finished": None,
        "cancel": False,
    }
    _jobs[job["id"]] = job
    _prune_jobs()
    _job_queues[lane].put((job, fn))
    return public_job(job)


def public_job(job: dict) -> dict:
    return {k: v for k, v in job.items() if k != "cancel"} | {"cancel_requested": job["cancel"]}


def check_cancel(job: dict | None) -> None:
    if job and job.get("cancel"):
        raise JobCancelled()


def _prune_jobs() -> None:
    done = sorted((j for j in _jobs.values() if j["finished"]), key=lambda j: j["finished"])
    for j in done[:-30]:
        _jobs.pop(j["id"], None)


def _job_worker(lane: str) -> None:
    q = _job_queues[lane]
    while True:
        job, fn = q.get()
        if job["cancel"]:
            job.update(status="cancelled", message="Cancelled", finished=time.time())
            continue
        job.update(status="running", message="Starting…")
        try:
            job["result"] = fn(job)
            job.update(status="done", progress=1.0)
            if job["message"] in ("Starting…", ""):
                job["message"] = "Done"
        except JobCancelled:
            job.update(status="cancelled", message="Cancelled")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            job.update(status="error", error=str(e), message=str(e))
        job["finished"] = time.time()


def start_background() -> None:
    global _workers_started
    if _workers_started:
        return
    _workers_started = True
    for lane in _job_queues:
        threading.Thread(target=_job_worker, args=(lane,), daemon=True, name=f"jobs-{lane}").start()
    threading.Thread(target=_coco_writer_loop, daemon=True, name="coco-writer").start()


# ----------------------------------------------------------------------------
# Folder import
# ----------------------------------------------------------------------------

def scan_videos(root: Path) -> list[Path]:
    """All videos under root in natural order: a folder's own files, then its sub-folders."""
    out: list[Path] = []
    data_dir = DATA_DIR.resolve()
    for dirpath, dirnames, filenames in os.walk(root):
        keep = []
        for d in dirnames:
            full = Path(dirpath, d)
            if d.startswith((".", "_", "$")):
                continue
            try:
                if full.resolve() == data_dir:
                    continue
            except OSError:
                continue
            keep.append(d)
        dirnames[:] = sorted(keep, key=natural_key)
        for f in sorted(filenames, key=natural_key):
            if Path(f).suffix.lower() in VIDEO_EXTS and not f.startswith("."):
                out.append(Path(dirpath, f))
    return out


def source_index() -> dict[str, str]:
    idx = {}
    for d in library_dirs():
        src = read_meta(d.name).get("source_path")
        if src:
            idx[os.path.normcase(src)] = d.name
    return idx


def prefill_annotations(vid: str, rel_path: str, label_from_folder: bool) -> None:
    with vlock(vid):
        data = load_annotations(vid)
        changed = False
        if label_from_folder and not data.get("video_label"):
            lab = label_from_path(rel_path)
            if lab:
                data["video_label"] = lab
                changed = True
        pid, sid = ids_from_path(rel_path)
        if pid and not data.get("participant_id"):
            data["participant_id"], changed = pid, True
        if sid and not data.get("session_id"):
            data["session_id"], changed = sid, True
        if changed or not annotation_path(video_dir(vid)).is_file():
            persist(vid, data)


def import_folder(job: dict | None, root: Path, fps: float | None, label_from_folder: bool,
                  auto_vip: bool, reextract: bool, vip_model: str = vip.DEFAULT_WEIGHTS,
                  log: Callable[[str], None] | None = None) -> dict:
    videos = scan_videos(root)
    if not videos:
        raise ValueError(f"No video files found in {root}")
    known = source_index()
    plan: list[tuple[str, Path, Path]] = []
    for order, p in enumerate(videos):
        rel = p.relative_to(root)
        group = "/".join([root.name or str(root), *rel.parent.parts])
        src = str(p.resolve())
        vid = known.get(os.path.normcase(src))
        if vid is None:
            vid = unique_video_id(make_video_id(rel))
            (DATA_DIR / vid).mkdir(parents=True, exist_ok=True)
            known[os.path.normcase(src)] = vid
        has_frames = bool(list_frames(DATA_DIR / vid))
        with vlock(vid):
            meta = read_meta(vid)
            meta.update(
                video_id=vid, source_path=src, source_name=p.name, rel_path=str(rel),
                group=group, order=order, root=str(root),
            )
            if not has_frames or reextract:
                meta.update(status="queued", error=None)
            write_meta(vid, meta)
        plan.append((vid, p, rel))

    total = len(plan)
    imported, skipped, failed = [], [], []
    try:
        for i, (vid, p, rel) in enumerate(plan):
            check_cancel(job)
            if job:
                job["message"] = f"{i + 1}/{total} · {p.name}"
                job["progress"] = i / total
            if log:
                log(f"[{i + 1}/{total}] {rel}")
            d = DATA_DIR / vid
            if list_frames(d) and not reextract:
                skipped.append(vid)
            else:
                update_meta(vid, status="extracting", error=None)

                def on_progress(f: float, i=i) -> None:
                    check_cancel(job)
                    if job:
                        job["progress"] = (i + f) / total

                try:
                    with vlock(vid):
                        backup_annotations(d)
                    info = extract_frames(p, d, fps, on_progress)
                except JobCancelled:
                    update_meta(vid, status="cancelled")
                    raise
                except Exception as e:  # noqa: BLE001
                    update_meta(vid, status="error", error=str(e)[:300])
                    failed.append({"video_id": vid, "error": str(e)[:300]})
                    if log:
                        log(f"    ERROR: {e}")
                    continue
                update_meta(vid, frames_version=int(time.time()), **info)
                imported.append(vid)
            prefill_annotations(vid, str(rel), label_from_folder)
            update_meta(vid, status="ready")
            if auto_vip:
                if job is None:
                    run_auto_vip(None, vid, vip_model, 640, log=log)
                else:
                    submit_job("auto_vip", "ml", f"Auto-VIP · {p.name}",
                               lambda j, vid=vid: run_auto_vip(j, vid, vip_model, 640), video_id=vid)
    except JobCancelled:
        for vid, _, _ in plan:
            if read_meta(vid).get("status") == "queued":
                update_meta(vid, status="cancelled")
        raise
    if job:
        job["message"] = f"{total} video(s): {len(imported)} extracted, {len(skipped)} already present, {len(failed)} failed"
    return {"total": total, "imported": imported, "skipped": skipped, "failed": failed}


def backup_annotations(d: Path) -> str | None:
    path = annotation_path(d)
    if not path.is_file():
        return None
    dest = d / f"annotations.backup.{time.strftime('%Y%m%d_%H%M%S')}.json"
    shutil.copy2(path, dest)
    return dest.name


# ----------------------------------------------------------------------------
# ML: detection model cache, Auto-VIP, tracking
# ----------------------------------------------------------------------------

_det_models: dict[str, Any] = {}
_det_lock = threading.Lock()


def get_detector(name: str):
    try:
        import ultralytics  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"YOLO unavailable ({e}). Install with: py -m pip install ultralytics") from e
    name = name if name in vip.WEIGHT_CHOICES else vip.DEFAULT_WEIGHTS
    if name not in _det_models:
        _det_models[name] = vip.load_model(name)
    return _det_models[name]


LYING_MIN_FRAMES = 5


FOLLOW_STOP_TEXT = {"end": "to the end", "scene_cut": "stopped at a scene cut", "lost": "stopped: person lost"}


def run_auto_vip(job: dict | None, vid: str, model_name: str, imgsz: int,
                 log: Callable[[str], None] | None = None, *, start: int = 0,
                 seed_frame: int | None = None, seed_box: list[float] | None = None,
                 lying: bool = True, lying_conf: float = 0.3, action: str = "auto_vip") -> dict:
    """Detect + track every person in frames[start:] and pick the POI in each frame: the largest
    visible person, or (with `seed_box`) the person under that box on `seed_frame`, followed through
    their shot. Where that person can't be followed, the frame keeps the POI it had (if any)."""
    d = video_dir(vid)
    all_frames = list_frames(d)
    if not all_frames:
        raise ValueError("Video has no frames")
    start = max(0, min(int(start), len(all_frames) - 1))
    frames = all_frames[start:]
    follow = seed_box is not None
    if follow:
        seed_frame = start if seed_frame is None else int(seed_frame)
        if not start <= seed_frame < len(all_frames):
            raise ValueError("The POI frame must lie inside the range")
        if len(seed_box) != 4:
            raise ValueError("Invalid POI box")
    if job:
        job["message"] = "Loading model…"
    t_start = time.time()
    model = vip.load_model(model_name)
    plain = vip.load_model(model_name) if lying or follow else None  # `model` carries ByteTrack callbacks
    if follow:
        seed_img = vip.read_image(d / all_frames[seed_frame])
        if seed_img is None:
            raise ValueError(f"Cannot read {all_frames[seed_frame]}")
        if vip.match_seed(vip.detect_persons(plain, seed_img, conf=0.2, imgsz=imgsz, lying=lying), seed_box) is None:
            raise ValueError(
                f"The red POI box on frame {seed_frame + 1} doesn't cover a person the detector can find"
                f"{'' if lying else ' (Find people lying down is off)'}. Use Track forward (T) for people "
                "the detector can't see, or pick another box.")
    lying_tracker = vip.LyingTracker()
    first = vip.read_image(d / frames[0])
    if first is None:
        raise ValueError(f"Cannot read {frames[0]}")
    raw: list[list[vip.Person]] = []
    cut_at: list[bool] = []
    shot_of: list[int] = []
    prev_sig = None
    id_offset = max_tid = 0
    cuts = 0
    t0 = time.time()
    for i, name in enumerate(frames):
        check_cancel(job)
        img = first if i == 0 else vip.read_image(d / name)
        if img is None:
            raw.append([])
            cut_at.append(False)
            shot_of.append(cuts)
            continue
        sig = vip.frame_signature(img)
        cut = vip.is_scene_cut(prev_sig, sig)
        prev_sig = sig
        if cut:
            # fresh tracker per shot; offset keeps IDs unique within the video
            vip.reset_tracker(model)
            lying_tracker.reset()
            id_offset = max_tid
            cuts += 1
        people = vip.track_persons(model, img, imgsz=imgsz)
        for p in people:
            p.tid += id_offset
            max_tid = max(max_tid, p.tid)
        if lying:
            lying_people = vip.merge_lying(people, vip.detect_lying(plain, img, conf=lying_conf, imgsz=imgsz))
            if lying_people:
                people += lying_tracker.update(lying_people)
        raw.append(people)
        cut_at.append(cut)
        shot_of.append(cuts)
        if job and (i % 5 == 0 or i == len(frames) - 1):
            rate = (i + 1) / max(1e-6, time.time() - t0)
            job["progress"] = (i + 1) / len(frames) * 0.97
            job["message"] = f"Frame {start + i + 1}/{len(all_frames)} · {rate:.1f} fps"
        if log and i % 200 == 0:
            log(f"    auto-VIP {vid}: frame {start + i + 1}/{len(all_frames)}")

    seed = None
    if follow:
        seed = vip.match_seed(raw[seed_frame - start], seed_box)
        if seed is None:
            raise ValueError(f"The red POI box on frame {seed_frame + 1} matched none of the tracked people. "
                             "Use Track forward (T) instead, or pick another box.")

    # A real lying person stays put for a while (or for most of a short shot);
    # rotated-pass blips on hands or shadows do not.
    shot_len: dict[int, int] = defaultdict(int)
    for s in shot_of:
        shot_len[s] += 1
    lying_hits: dict[int, int] = defaultdict(int)
    lying_shot: dict[int, int] = {}
    for people, s in zip(raw, shot_of):
        for p in people:
            if vip.is_lying_tid(p.tid):
                lying_hits[p.tid] += 1
                lying_shot.setdefault(p.tid, s)
    short = {cid for cid, n in lying_hits.items()
             if n < LYING_MIN_FRAMES and not (n >= 3 and n >= 0.6 * shot_len[lying_shot[cid]])}
    short.discard(seed.tid if seed else None)
    if short:
        raw = [[p for p in people if p.tid not in short] for people in raw]
    lying_frames = sum(1 for people in raw if any(vip.is_lying_tid(p.tid) for p in people))

    per_frame: list[tuple[list[tuple[int, vip.Person]], int | None]] = []
    span = None
    if follow:
        found, span = vip.follow_identity(raw, shot_of, seed_frame - start, seed)
        span = {**span, "first": span["first"] + start, "last": span["last"] + start,
                "seed_frame": seed_frame, "frames": len(found)}
        for k, people in enumerate(raw):
            poi = found.get(k)
            per_frame.append(([(seed.tid if p is poi else p.tid, p) for p in people],
                              seed.tid if poi is not None else None))
    else:
        selector = vip.VipSelector(first.shape[1], first.shape[0])
        for people, cut in zip(raw, cut_at):
            vip_id = selector.update(people, scene_cut=cut)
            per_frame.append((selector.visible_people(people), vip_id))

    with vlock(vid):
        data = load_annotations(vid)
        snapshot_for_undo(vid, data, frames, action)
        # IDs must not clash with automatic IDs kept on frames outside the range
        track_off, lie_off = 0, 0
        for name in all_frames[:start]:
            for o in data["frames"][name]["objects"]:
                m = re.fullmatch(r"(track|lie)_(\d+)", str(o["id"]))
                if m and m[1] == "track":
                    track_off = max(track_off, int(m[2]))
                elif m:
                    lie_off = max(lie_off, int(m[2]) + 1)
        poi_frames = box_frames = kept_poi = 0
        for name, (people, vip_id) in zip(frames, per_frame):
            slot = data["frames"][name]
            manual = [o for o in slot["objects"] if is_user_box(o) or is_locked_poi(o)]
            manual_poi = any(o.get("is_poi") for o in manual)
            vip_visible = any(cid == vip_id for cid, _ in people)
            old_poi = None
            if follow and not vip_visible:
                old_poi = next((o for o in slot["objects"] if o.get("is_poi") and o not in manual
                                and not is_legacy_box(o)), None)
            legacy = []
            for o in slot["objects"]:
                if o not in manual and is_legacy_box(o) and not any(vip.iou(o["bbox"], p.box) > 0.5 for _, p in people):
                    if vip_visible or manual_poi:
                        o["is_poi"] = False
                    legacy.append(o)
            legacy_poi = any(o.get("is_poi") for o in legacy)
            objects = list(manual) + legacy
            for cid, p in people:
                if any(vip.iou(p.box, m["bbox"]) > 0.5 for m in manual):
                    continue
                is_poi = not (manual_poi or legacy_poi) and cid == vip_id
                objects.append({
                    "id": (f"lie_{cid - vip.LYING_TID_BASE + lie_off}" if vip.is_lying_tid(cid)
                           else f"track_{cid + track_off}"),
                    "bbox": [round(v, 2) for v in p.box],
                    "label": "person_of_interest" if is_poi else "person",
                    "behaviours": [],
                    "confirmed": True,
                    "source": "auto_vip",
                    "conf": round(p.conf, 4),
                    "is_poi": is_poi,
                })
            if old_poi is not None and not (manual_poi or legacy_poi):
                fresh = [o for o in objects if o not in manual and o not in legacy]
                best = max(fresh, key=lambda o: vip.iou(o["bbox"], old_poi["bbox"]), default=None)
                if best is not None and vip.iou(best["bbox"], old_poi["bbox"]) > 0.5:
                    best["is_poi"] = True
                else:
                    objects.append(old_poi)
                kept_poi += 1
            slot["objects"] = objects
            sync_poi(slot)
            if slot["bbox"]:
                poi_frames += 1
            if objects:
                box_frames += 1
        persist(vid, data, coco_now=True)
    update_meta(vid, vip_done=True)
    elapsed = time.time() - t_start
    if job:
        if span:
            job["message"] = (f"POI followed on {span['frames']} frames ({span['first'] + 1}–{span['last'] + 1}, "
                              f"{FOLLOW_STOP_TEXT.get(span['stop_reason'], span['stop_reason'])})"
                              f" · people boxed in {box_frames}/{len(frames)} frames · {cuts} scene cut(s)"
                              f" · {elapsed:.0f} s")
        else:
            job["message"] = (f"VIP found in {poi_frames}/{len(frames)} frames · {cuts} scene cut(s)"
                              f" · lying person in {lying_frames} · {elapsed:.0f} s")
    return {"video_id": vid, "poi_frames": poi_frames, "frames": len(frames), "scene_cuts": cuts,
            "lying_frames": lying_frames, "box_frames": box_frames, "start": start, "follow": span,
            "kept_poi": kept_poi, "elapsed": round(elapsed, 1)}


class TrackRequest(BaseModel):
    seed_box: list[float] | None = None
    num_frames: int = 0
    propagate_bbox: bool = True
    propagate_behaviours: bool = True
    propagate_comment: bool = False
    overwrite_bbox: bool = True
    overwrite_labels: bool = True
    stop_on_label_change: bool = False
    max_gap: int = 15
    iou_thresh: float = 0.3
    imgsz: int = 640
    model: str = vip.DEFAULT_WEIGHTS


def _apply_track_results(vid: str, seed_name: str, results: list[tuple[str, list[float] | None, str]],
                         req: TrackRequest, track_id: str, locked: bool = False) -> None:
    with vlock(vid):
        data = load_annotations(vid)
        seed = data["frames"][seed_name]
        seed_beh = list(seed["behaviours"])
        seed_comment = seed["comment"]
        snapshot_for_undo(vid, data, [n for n, _, _ in results], "track")
        for name, box, source in results:
            slot = data["frames"][name]
            if box is not None and req.propagate_bbox:
                existing_poi = next((o for o in slot["objects"] if o.get("is_poi")), None)
                if name == seed_name or req.overwrite_bbox or existing_poi is None:
                    others = [o for o in slot["objects"] if o["id"] != track_id and not o.get("is_poi")]
                    others = [o for o in others if vip.iou(o["bbox"], box) < 0.6]
                    slot["objects"] = others + [{
                        "id": track_id,
                        "bbox": [round(v, 2) for v in box],
                        "label": "person_of_interest",
                        "behaviours": [],
                        "confirmed": True,
                        "source": source,
                        "conf": None,
                        "is_poi": True,
                        "poi_locked": locked,
                    }]
            if name != seed_name:
                if req.propagate_behaviours and (req.overwrite_labels or not slot["behaviours"]):
                    slot["behaviours"] = list(seed_beh)
                if req.propagate_comment and (req.overwrite_labels or not slot["comment"]):
                    slot["comment"] = seed_comment
            sync_poi(slot)
        persist(vid, data, coco_now=True)


def run_track(job: dict | None, vid: str, idx: int, req: TrackRequest) -> dict:
    d = video_dir(vid)
    with vlock(vid):
        data = load_annotations(vid)
    frames = data["_frame_order"]
    if idx < 0 or idx >= len(frames):
        raise ValueError("Frame index out of range")
    seed_name = frames[idx]
    seed = data["frames"][seed_name]
    seed_box = req.seed_box or seed.get("bbox")
    poi = next((o for o in seed["objects"] if o.get("is_poi")), None)
    track_id = poi["id"] if poi else new_obj_id("track")
    locked = poi is None or is_locked_poi(poi)
    end = len(frames) if req.num_frames <= 0 else min(len(frames), idx + 1 + req.num_frames)

    results: list[tuple[str, list[float] | None, str]] = [(seed_name, seed_box, "manual" if poi is None else poi["source"])]
    stop_reason = "end"

    if not req.propagate_bbox:
        for i in range(idx + 1, end):
            name = frames[i]
            if req.stop_on_label_change and data["frames"][name]["behaviours"] and \
                    data["frames"][name]["behaviours"] != seed["behaviours"]:
                stop_reason = "label_change"
                break
            results.append((name, None, ""))
        _apply_track_results(vid, seed_name, results, req, track_id, locked)
        return {"covered": len(results), "stop_reason": stop_reason, "stopped_at": idx + len(results) - 1}

    if not seed_box or len(seed_box) != 4:
        raise ValueError("Draw or select a person-of-interest box on this frame first")
    if job:
        job["message"] = "Loading model…"
    model = vip.load_model(req.model)
    seed_img = vip.read_image(d / seed_name)
    if seed_img is None:
        raise ValueError(f"Cannot read {seed_name}")
    tracker = vip.SingleTargetTracker(model, seed_img, seed_box, imgsz=req.imgsz,
                                      iou_thresh=req.iou_thresh, max_gap=req.max_gap,
                                      lying_model=vip.load_model(req.model))
    results[0] = (seed_name, tracker.box, results[0][2])
    counts: dict[str, int] = {}
    t0 = time.time()
    for i in range(idx + 1, end):
        check_cancel(job)
        name = frames[i]
        if req.stop_on_label_change and data["frames"][name]["behaviours"] and \
                data["frames"][name]["behaviours"] != seed["behaviours"]:
            stop_reason = "label_change"
            break
        img = vip.read_image(d / name)
        if img is None:
            stop_reason = "unreadable"
            break
        box, source = tracker.step(img)
        if box is None:
            stop_reason = source
            break
        results.append((name, box, source))
        counts[source] = counts.get(source, 0) + 1
        if job and ((i - idx) % 3 == 0 or i == end - 1):
            done = i - idx
            job["progress"] = done / max(1, end - idx - 1) * 0.97
            job["message"] = f"Frame {i + 1}/{len(frames)} · {done / max(1e-6, time.time() - t0):.1f} fps"
    if stop_reason in ("lost", "scene_cut"):
        # trailing "hold" frames were only a guess
        while len(results) > 1 and results[-1][2] == "hold":
            results.pop()
    _apply_track_results(vid, seed_name, results, req, track_id, locked)
    stopped_at = idx + len(results) - 1
    if job:
        job["message"] = f"Tracked {len(results)} frame(s) · stopped: {stop_reason}"
    return {"covered": len(results), "stop_reason": stop_reason, "stopped_at": stopped_at,
            "track_id": track_id, "sources": counts}


# ----------------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------------

def export_dir() -> Path:
    d = DATA_DIR / EXPORT_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def video_record(vid: str) -> dict:
    """Merged per-video record: video id, trauma label, behaviours and boxes per frame."""
    with vlock(vid):
        data = load_annotations(vid)
    meta = read_meta(vid)
    fps = meta.get("fps")
    frames = data["_frame_order"]
    out_frames = {}
    for i, name in enumerate(frames):
        slot = data["frames"][name]
        out_frames[name] = {
            "index": i,
            "time": round(i / fps, 3) if fps else None,
            "behaviours": slot["behaviours"],
            "comment": slot["comment"],
            "bbox": slot["bbox"],
            "objects": slot["objects"],
        }
    return {
        "video_id": vid,
        "filename": meta.get("source_name") or vid,
        "source_path": meta.get("source_path"),
        "group": meta.get("group"),
        "participant_id": data.get("participant_id", ""),
        "session_id": data.get("session_id", ""),
        "video_label": data.get("video_label"),
        "video_comment": data.get("video_comment", ""),
        "context": data.get("context", ""),
        "fps": fps,
        "num_frames": len(frames),
        "frame_dims": data.get("frame_dims", {}),
        "frames": out_frames,
    }


METADATA_COLUMNS = ["#", "participant_id", "session_id", "filename", "modality", "start_time",
                    "end_time", "behavioral", "trauma_label", "video_id", "start_frame", "end_frame"]


def metadata_rows(records: list[dict]) -> list[dict]:
    """One row per contiguous run of identical frame behaviours (one blank row if unlabelled)."""
    rows = []
    for rec in records:
        trauma = {"trauma": "trauma", "no_trauma": "no trauma"}.get(rec.get("video_label") or "", "")
        fps = rec.get("fps")
        base = {
            "participant_id": rec.get("participant_id", ""),
            "session_id": rec.get("session_id", ""),
            "filename": rec.get("filename"),
            "modality": "video",
            "trauma_label": trauma,
            "video_id": rec["video_id"],
        }
        segments = []
        cur, start = None, 0
        names = list(rec["frames"].keys())
        for i, name in enumerate(names + [None]):
            lab = ";".join(rec["frames"][name]["behaviours"]) if name else None
            if lab != cur:
                if cur:
                    segments.append((start, i - 1, cur))
                cur, start = lab, i
        if not segments:
            rows.append({**base, "start_time": "", "end_time": "", "behavioral": "",
                         "start_frame": "", "end_frame": ""})
            continue
        for s, e, lab in segments:
            rows.append({
                **base,
                "start_time": fmt_time(s / fps) if fps else "",
                "end_time": fmt_time((e + 1) / fps) if fps else "",
                "behavioral": lab.replace("_", " "),
                "start_frame": s + 1,
                "end_frame": e + 1,
            })
    for n, r in enumerate(rows, 1):
        r["#"] = n
    return rows


def write_metadata(rows: list[dict], folder: Path) -> list[str]:
    written = []
    csv_path = folder / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=METADATA_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    written.append(csv_path.name)
    try:
        from openpyxl import Workbook  # type: ignore

        wb = Workbook()
        ws = wb.active
        ws.title = "metadata"
        ws.append(METADATA_COLUMNS)
        for r in rows:
            ws.append([r.get(c, "") for c in METADATA_COLUMNS])
        wb.save(folder / "metadata.xlsx")
        written.append("metadata.xlsx")
    except ImportError:
        pass
    return written


def ordered_video_ids(only_ready: bool = True) -> list[str]:
    items = [video_summary(d.name) for d in library_dirs()]
    items = [v for v in items if v["num_frames"] or not only_ready]
    items.sort(key=lambda v: (natural_key(v["group"]), v["order"], natural_key(v["name"])))
    return [v["id"] for v in items]


def export_all() -> dict:
    flush_coco()
    folder = export_dir()
    vids_dir = folder / "videos"
    vids_dir.mkdir(exist_ok=True)
    records = []
    for vid in ordered_video_ids():
        rec = video_record(vid)
        records.append(rec)
        (vids_dir / f"{vid}_annotations.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        with vlock(vid):
            coco = build_coco(vid, load_annotations(vid))
        (vids_dir / f"{vid}{COCO_SUFFIX}").write_text(
            json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")
    dataset = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "behaviours": BEHAVIOURS,
        "video_labels": VIDEO_LABELS,
        "videos": records,
    }
    (folder / "all_annotations.json").write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    files = ["all_annotations.json", *write_metadata(metadata_rows(records), folder)]
    return {"folder": str(folder), "files": files, "videos": len(records)}


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    start_background()
    yield
    flush_coco()


app = FastAPI(title="Trauma Behaviour Video Annotator", lifespan=lifespan)


class FrameAnnotation(BaseModel):
    frame: str
    behaviours: list[str] | None = None
    comment: str | None = None
    objects: list[dict] | None = None


class VideoMeta(BaseModel):
    video_label: str | None = None
    video_comment: str | None = None
    context: str | None = None
    participant_id: str | None = None
    session_id: str | None = None


LOW_CONF_FLOOR = 0.15


class DetectRequest(BaseModel):
    conf: float = 0.35
    imgsz: int = 960
    augment: bool = False
    lying: bool = True
    model: str = vip.DEFAULT_WEIGHTS


class FillRequest(BaseModel):
    start: int
    end: int
    behaviours: list[str]
    only_empty: bool = False


class AutoVipRequest(BaseModel):
    model: str = vip.DEFAULT_WEIGHTS
    imgsz: int = 640


class DetectTrackRequest(BaseModel):
    model: str = vip.DEFAULT_WEIGHTS
    imgsz: int = 640
    conf: float = 0.3  # floor for lying-down (rotated-pass) boxes, which ByteTrack doesn't score
    lying: bool = True
    start: int = 0
    poi: str = "largest"  # "largest" | "follow"
    seed_frame: int | None = None
    seed_box: list[float] | None = None


class FolderRequest(BaseModel):
    path: str
    fps: float | None = None
    label_from_folder: bool = True
    auto_vip: bool = False
    reextract: bool = False


class ProbeRequest(BaseModel):
    filename: str
    video_id: str | None = None


@app.get("/api/config")
def api_config():
    import importlib.util
    return {
        "behaviours": BEHAVIOURS,
        "video_labels": VIDEO_LABELS,
        "object_labels": OBJECT_LABELS,
        "yolo_models": vip.WEIGHT_CHOICES,
        "yolo_ready": importlib.util.find_spec("ultralytics") is not None,
        "extract_ready": ffmpeg_available() or opencv_available(),
        "ffmpeg_ready": ffmpeg_available(),
        "data_dir": str(DATA_DIR),
        "export_dir": str(DATA_DIR / EXPORT_DIRNAME),
        "detect_defaults": {"conf": 0.35, "imgsz": 960, "augment": False, "model": vip.DEFAULT_WEIGHTS},
    }


@app.get("/api/videos")
def api_videos():
    items = [video_summary(d.name) for d in library_dirs()]
    items = [v for v in items if v["num_frames"] or v["status"] != "empty"]
    items.sort(key=lambda v: (natural_key(v["group"]), v["order"], natural_key(v["name"])))
    return {"videos": items}


@app.get("/api/video/{vid}")
def api_video(vid: str):
    with vlock(vid):
        data = load_annotations(vid)
    if not data["_frame_order"]:
        raise HTTPException(409, "This video has no frames yet (still queued or extracting)")
    meta = read_meta(vid)
    order = data["_frame_order"]
    return {
        "video_id": vid,
        "name": meta.get("source_name") or vid,
        "group": meta.get("group") or "Uploads",
        "source_path": meta.get("source_path"),
        "fps": meta.get("fps"),
        "frames_version": meta.get("frames_version", 0),
        "frames": order,
        "annotations": {n: data["frames"][n] for n in order},
        "video_label": data.get("video_label"),
        "video_comment": data.get("video_comment", ""),
        "context": data.get("context", ""),
        "participant_id": data.get("participant_id", ""),
        "session_id": data.get("session_id", ""),
        "has_undo": (video_dir(vid) / UNDO_FILE).is_file(),
        "undo_action": read_json(video_dir(vid) / UNDO_FILE, {}).get("action"),
    }


@app.get("/api/frame/{vid}/{name}")
def api_frame(vid: str, name: str):
    if not is_safe_name(name) or Path(name).suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(400, "Invalid frame name")
    p = video_dir(vid) / name
    if not p.is_file():
        raise HTTPException(404, "Frame not found")
    return FileResponse(p, headers={"Cache-Control": "private, max-age=86400"})


@app.post("/api/annotate/{vid}")
def api_annotate(vid: str, ann: FrameAnnotation):
    with vlock(vid):
        data = load_annotations(vid)
        if ann.frame not in data["frames"]:
            raise HTTPException(404, f"Unknown frame '{ann.frame}'")
        slot = data["frames"][ann.frame]
        if ann.behaviours is not None:
            bad = [b for b in ann.behaviours if b not in BEHAVIOURS]
            if bad:
                raise HTTPException(400, f"Unknown behaviours: {bad}")
            slot["behaviours"] = clean_behaviours(ann.behaviours)
        if ann.comment is not None:
            slot["comment"] = ann.comment
        if ann.objects is not None:
            slot["objects"] = [n for n in (normalize_object(o) for o in ann.objects) if n]
        sync_poi(slot)
        persist(vid, data)
        return {"ok": True, "frame": ann.frame, "annotation": slot}


@app.post("/api/fill/{vid}")
def api_fill(vid: str, req: FillRequest):
    with vlock(vid):
        data = load_annotations(vid)
        frames = data["_frame_order"]
        start, end = max(0, req.start), min(len(frames) - 1, req.end)
        if end < start:
            raise HTTPException(400, "Empty range")
        names = frames[start:end + 1]
        snapshot_for_undo(vid, data, names, "fill")
        beh = clean_behaviours(req.behaviours)
        changed = 0
        for n in names:
            slot = data["frames"][n]
            if req.only_empty and slot["behaviours"]:
                continue
            slot["behaviours"] = list(beh)
            sync_poi(slot)
            changed += 1
        persist(vid, data)
        return {"ok": True, "changed": changed, "start": start, "end": end,
                "annotations": {n: data["frames"][n] for n in names}}


@app.post("/api/video-meta/{vid}")
def api_video_meta(vid: str, meta: VideoMeta):
    with vlock(vid):
        data = load_annotations(vid)
        if meta.video_label is not None:
            if meta.video_label == "":
                data["video_label"] = None
            elif meta.video_label in VIDEO_LABELS:
                data["video_label"] = meta.video_label
            else:
                raise HTTPException(400, f"Unknown video label '{meta.video_label}'")
        for field in ("video_comment", "context", "participant_id", "session_id"):
            value = getattr(meta, field)
            if value is not None:
                data[field] = value.strip() if field.endswith("_id") else value
        persist(vid, data)
    return {"ok": True}


@app.post("/api/detect/{vid}/{name}")
def api_detect(vid: str, name: str, req: DetectRequest):
    p = video_dir(vid) / name
    if not is_safe_name(name) or not p.is_file():
        raise HTTPException(404, "Frame not found")
    img = vip.read_image(p)
    if img is None:
        raise HTTPException(400, "Cannot read frame image")
    conf = max(0.1, min(0.95, float(req.conf)))
    low_conf = False
    with _det_lock:
        model = get_detector(req.model)
        people = vip.detect_persons(model, img, conf=conf, imgsz=req.imgsz, augment=req.augment,
                                    lying=req.lying)
        if not people and not req.augment:
            people = vip.detect_persons(model, img, conf=conf, imgsz=req.imgsz, augment=True,
                                        lying=req.lying)
        if not people:
            # Close-ups, lying or covered people often score just under the threshold:
            # offer the best weak candidates for the user to confirm instead of nothing.
            weak = vip.detect_persons(model, img, conf=LOW_CONF_FLOOR, imgsz=req.imgsz, augment=True,
                                      lying=req.lying)
            for p in weak:
                if len(people) < 3 and all(vip.iou(p.box, q.box) < 0.3 for q in people):
                    people.append(p)
            low_conf = bool(people)
    dets = [{
        "id": new_obj_id("det"),
        "bbox": p.box,
        "conf": p.conf,
        "label": "person",
        "source": "detection",
        "confirmed": True,
        "is_poi": False,
        "behaviours": [],
    } for p in people]
    tip = None
    if low_conf:
        tip = (f"No person at ≥{int(conf * 100)}% — showing the best weak match "
               f"({int(people[0].conf * 100)}%). Check the box; delete it (Del) if wrong.")
    elif not dets:
        tip = (f"No person found (even at {int(LOW_CONF_FLOOR * 100)}%). Draw the box by hand, "
               "or try model small/medium in Detection settings.")
    return {"detections": dets, "count": len(dets), "tip": tip, "low_conf": low_conf,
            "image_size": [int(img.shape[1]), int(img.shape[0])]}


def ensure_no_tracking_job(vid: str) -> None:
    active = [j for j in _jobs.values() if j["video_id"] == vid and j["kind"] in ("auto_vip", "detect_track", "track")
              and j["status"] in ("queued", "running")]
    if active:
        raise HTTPException(409, "A tracking job is already running for this video")


@app.post("/api/auto-vip/{vid}")
def api_auto_vip(vid: str, req: AutoVipRequest):
    video_dir(vid)
    ensure_no_tracking_job(vid)
    name = read_meta(vid).get("source_name") or vid
    job = submit_job("auto_vip", "ml", f"Auto-VIP · {name}",
                     lambda j: run_auto_vip(j, vid, req.model, req.imgsz), video_id=vid)
    return {"job": job}


@app.post("/api/detect-track/{vid}")
def api_detect_track(vid: str, req: DetectTrackRequest):
    """One click: detect + track every person in all frames (or from `start`) and choose the POI."""
    if req.poi not in ("largest", "follow"):
        raise HTTPException(400, "poi must be 'largest' or 'follow'")
    with vlock(vid):
        data = load_annotations(vid)
    frames = data["_frame_order"]
    if not frames:
        raise HTTPException(409, "This video has no frames yet")
    start = max(0, min(req.start, len(frames) - 1))
    seed_frame = seed_box = None
    if req.poi == "follow":
        seed_frame = start if req.seed_frame is None else req.seed_frame
        if not start <= seed_frame < len(frames):
            raise HTTPException(400, "The POI frame must lie inside the range")
        seed_box = req.seed_box or data["frames"][frames[seed_frame]].get("bbox")
        if not seed_box or len(seed_box) != 4:
            raise HTTPException(400, f"No red POI box on frame {seed_frame + 1} to follow")
    ensure_no_tracking_job(vid)
    name = read_meta(vid).get("source_name") or vid
    conf = max(0.1, min(0.95, float(req.conf)))
    job = submit_job("detect_track", "ml", f"Detect + track · {name}",
                     lambda j: run_auto_vip(j, vid, req.model, req.imgsz, start=start, seed_frame=seed_frame,
                                            seed_box=seed_box, lying=req.lying, lying_conf=conf,
                                            action="detect_track"),
                     video_id=vid)
    return {"job": job}


@app.post("/api/track/{vid}/{idx}")
def api_track(vid: str, idx: int, req: TrackRequest):
    if not (req.propagate_bbox or req.propagate_behaviours or req.propagate_comment):
        raise HTTPException(400, "Select at least one thing to propagate")
    video_dir(vid)
    if not req.propagate_bbox:
        try:
            return {"result": run_track(None, vid, idx, req)}
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
    name = read_meta(vid).get("source_name") or vid
    job = submit_job("track", "ml", f"Track · {name}", lambda j: run_track(j, vid, idx, req), video_id=vid)
    return {"job": job}


@app.post("/api/undo/{vid}")
def api_undo(vid: str):
    with vlock(vid):
        undo_path = video_dir(vid) / UNDO_FILE
        snap = read_json(undo_path, None)
        if not snap:
            raise HTTPException(404, "Nothing to undo")
        data = load_annotations(vid)
        for name, slot in (snap.get("frames") or {}).items():
            if name in data["frames"]:
                data["frames"][name] = normalize_slot(slot)
        persist(vid, data, coco_now=True)
        undo_path.unlink(missing_ok=True)
    return {"ok": True, "restored": len(snap.get("frames") or {}), "action": snap.get("action")}


# ---- Jobs -------------------------------------------------------------------

@app.get("/api/jobs")
def api_jobs():
    jobs = sorted(_jobs.values(), key=lambda j: j["created"], reverse=True)
    return {"jobs": [public_job(j) for j in jobs]}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return public_job(job)


@app.post("/api/jobs/{job_id}/cancel")
def api_job_cancel(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    job["cancel"] = True
    return public_job(job)


# ---- Import -----------------------------------------------------------------

@app.get("/api/browse-folder")
def api_browse_folder():
    """Native folder picker. Runs in a subprocess because Tk must own its thread."""
    script = (
        "import tkinter as tk\nfrom tkinter import filedialog\n"
        "r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True); r.update()\n"
        "p = filedialog.askdirectory(parent=r, title='Select a folder containing videos', mustexist=True)\n"
        "r.destroy()\nimport sys\nsys.stdout.buffer.write((p or '').encode('utf-8'))\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, timeout=900)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(501, f"Folder picker unavailable ({e}). Paste the folder path instead.") from e
    if out.returncode != 0:
        raise HTTPException(501, "Folder picker unavailable. Paste the folder path instead.")
    path = out.stdout.decode("utf-8", "replace").strip()
    return {"path": str(Path(path)) if path else ""}


@app.post("/api/import/folder")
def api_import_folder(req: FolderRequest):
    root = Path(req.path.strip().strip('"')).expanduser()
    if not root.is_dir():
        raise HTTPException(400, f"Folder not found: {root}")
    root = root.resolve()
    videos = scan_videos(root)
    if not videos:
        raise HTTPException(400, f"No video files ({', '.join(sorted(VIDEO_EXTS))}) found in {root}")
    if not (ffmpeg_available() or opencv_available()):
        raise HTTPException(503, "Cannot extract frames: install ffmpeg or opencv-python")
    fps = req.fps if req.fps and req.fps > 0 else None
    # Show every video in the sidebar right away, before extraction starts
    known = source_index()
    for order, p in enumerate(videos):
        if os.path.normcase(str(p.resolve())) in known:
            continue
        rel = p.relative_to(root)
        vid = unique_video_id(make_video_id(rel))
        write_meta(vid, {
            "video_id": vid, "source_path": str(p.resolve()), "source_name": p.name,
            "rel_path": str(rel), "group": "/".join([root.name, *rel.parent.parts]),
            "order": order, "root": str(root), "status": "queued",
        })
        known[os.path.normcase(str(p.resolve()))] = vid
    job = submit_job(
        "import", "import", f"Import folder · {root.name}",
        lambda j: import_folder(j, root, fps, req.label_from_folder, req.auto_vip, req.reextract),
    )
    return {"job": job, "count": len(videos)}


def _empty_stats() -> dict:
    return {"exists": False, "num_frames": 0, "has_annotations": False, "labelled": 0, "boxed": 0}


def _probe_stats(base: str) -> dict:
    if not (DATA_DIR / base).is_dir():
        return _empty_stats()
    s = video_stats(base)
    return {"exists": True, **{k: s[k] for k in ("num_frames", "has_annotations", "labelled", "boxed")}}


@app.post("/api/import/video/probe")
def api_import_video_probe(req: ProbeRequest):
    base = make_video_id(Path(req.video_id or req.filename).name)
    stats = _probe_stats(base)
    return {
        "video_id": base,
        "alternate_id": unique_video_id(base) if stats["exists"] else base,
        **stats,
        "extract_ready": ffmpeg_available() or opencv_available(),
    }


def _resolve_import_target(base: str, mode: str) -> tuple[str, str, Path, str | None]:
    stats = _probe_stats(base)
    if mode == "auto":
        if stats["exists"] and (stats["num_frames"] or stats["has_annotations"]):
            raise HTTPException(409, detail={
                "message": "This video is already in the library. Choose what to do.",
                "probe": {**stats, "video_id": base, "alternate_id": unique_video_id(base)},
            })
        mode = "create"
    if mode == "create_new":
        base, mode = unique_video_id(base), "create"
    if mode not in ("create", "reextract_keep", "reextract_wipe", "open"):
        raise HTTPException(400, f"Unknown mode '{mode}'")
    out_dir = DATA_DIR / base
    backup = None
    if mode != "open":
        out_dir.mkdir(parents=True, exist_ok=True)
        if mode in ("reextract_keep", "reextract_wipe"):
            backup = backup_annotations(out_dir)
        if mode == "reextract_wipe":
            annotation_path(out_dir).unlink(missing_ok=True)
            (out_dir / UNDO_FILE).unlink(missing_ok=True)
        if mode == "create" and list_frames(out_dir):
            raise HTTPException(409, "Folder already has frames; choose re-extract or open")
    return base, mode, out_dir, backup


@app.post("/api/import/video")
async def api_import_video(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    video_id: str | None = Form(None),
    fps: float | None = Form(None),
):
    if not (ffmpeg_available() or opencv_available()):
        raise HTTPException(503, "Cannot extract frames: install ffmpeg or opencv-python")
    raw_name = file.filename or "video.mp4"
    ext = Path(raw_name).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported video type '{ext}'")
    base = make_video_id(Path(video_id or raw_name).name) if not video_id else video_id
    if not is_safe_name(base):
        raise HTTPException(400, "Invalid video id")
    base, mode, out_dir, backup = _resolve_import_target(base, mode)
    if mode == "open":
        return {"ok": True, "mode": "open", "video_id": base}

    uploads = out_dir / "_uploads"
    uploads.mkdir(exist_ok=True)
    src = uploads / f"source{ext}"
    with src.open("wb") as f:
        shutil.copyfileobj(file.file, f, 1024 * 1024)
    write_meta(base, {**read_meta(base), "video_id": base, "source_name": raw_name,
                      "source_path": str(src), "group": "Uploads", "order": int(time.time()),
                      "status": "queued", "error": None})
    fps_val = fps if fps and fps > 0 else None

    def work(job: dict) -> dict:
        update_meta(base, status="extracting")
        try:
            info = extract_frames(src, out_dir, fps_val,
                                  lambda f: (check_cancel(job), job.__setitem__("progress", f)))
        except JobCancelled:
            update_meta(base, status="cancelled")
            raise
        except Exception as e:
            update_meta(base, status="error", error=str(e)[:300])
            raise
        update_meta(base, status="ready", frames_version=int(time.time()), **info)
        with vlock(base):
            data = load_annotations(base)
            persist(base, data, coco_now=True)
        job["message"] = f"Extracted {info['num_frames']} frames"
        return {"video_id": base, **info}

    job = submit_job("import", "import", f"Extract · {raw_name}", work, video_id=base)
    return {"ok": True, "mode": mode, "video_id": base, "backup": backup, "job": job}


@app.post("/api/import/frames")
async def api_import_frames(
    files: list[UploadFile] = File(...),
    mode: str = Form("auto"),
    video_id: str | None = Form(None),
):
    if not files:
        raise HTTPException(400, "No files uploaded")
    first = files[0].filename or "frames"
    is_zip = len(files) == 1 and first.lower().endswith(".zip")
    base = video_id or make_video_id(Path(first).stem if is_zip else f"frames_{Path(first).parent.name or 'import'}")
    if not is_safe_name(base):
        raise HTTPException(400, "Invalid video id")
    base, mode, out_dir, backup = _resolve_import_target(base, mode)
    if mode == "open":
        return {"ok": True, "mode": "open", "video_id": base}
    clear_frames(out_dir)
    written = 0
    if is_zip:
        raw = await files[0].read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            imgs = sorted(
                (n for n in zf.namelist() if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS
                 and not Path(n).name.startswith(".")),
                key=natural_key,
            )
            for i, n in enumerate(imgs, 1):
                (out_dir / f"frame_{i:06d}{Path(n).suffix.lower()}").write_bytes(zf.read(n))
                written += 1
    else:
        imgs = sorted((f for f in files if Path(f.filename or "").suffix.lower() in IMAGE_EXTS),
                      key=lambda f: natural_key(f.filename or ""))
        for i, f in enumerate(imgs, 1):
            (out_dir / f"frame_{i:06d}{Path(f.filename or 'x.jpg').suffix.lower()}").write_bytes(await f.read())
            written += 1
    if written <= 0:
        raise HTTPException(400, "No image frames found in upload")
    write_meta(base, {**read_meta(base), "video_id": base, "source_name": first, "group": "Uploads",
                      "order": int(time.time()), "status": "ready", "frames_version": int(time.time()),
                      "num_frames": written})
    with vlock(base):
        persist(base, load_annotations(base), coco_now=True)
    return {"ok": True, "mode": mode, "video_id": base, "num_frames": written, "backup": backup}


@app.post("/api/import/json")
async def api_import_json(
    file: UploadFile = File(...),
    video_id: str | None = Form(None),
    merge: bool = Form(True),
):
    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8-sig"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid JSON: {e}") from e
    if isinstance(payload, dict) and isinstance(payload.get("videos"), list):
        match = next((v for v in payload["videos"] if v.get("video_id") == video_id), None)
        if match is None:
            raise HTTPException(400, f"Dataset file has no entry for video '{video_id}'")
        payload = match

    is_coco = "images" in payload and "annotations" in payload
    info = (payload.get("info") or {}) if is_coco else payload
    vid = video_id or info.get("video_id")
    if not vid or not (DATA_DIR / vid).is_dir():
        raise HTTPException(404, f"Video folder '{vid}' not found — import its frames first")

    with vlock(vid):
        data = load_annotations(vid)
        if not merge:
            data["frames"] = {n: empty_slot() for n in data["_frame_order"]}
        if info.get("video_label") in VIDEO_LABELS:
            data["video_label"] = info["video_label"]
        for field in ("video_comment", "context", "participant_id", "session_id"):
            if field in info:
                data[field] = info.get(field) or ""

        if is_coco:
            cats = {c["id"]: c["name"] for c in payload.get("categories", [])}
            id_to_name = {im["id"]: im["file_name"] for im in payload.get("images", [])}
            fresh_objects: dict[str, list] = {}
            for im in payload.get("images", []):
                name = im.get("file_name")
                if name not in data["frames"]:
                    continue
                beh = im.get("behaviours")
                if beh is None and im.get("behaviour"):
                    beh = [im["behaviour"]]
                if beh is not None:
                    data["frames"][name]["behaviours"] = clean_behaviours(beh)
                if "comment" in im:
                    data["frames"][name]["comment"] = im.get("comment") or ""
            for a in payload.get("annotations", []):
                name = id_to_name.get(a.get("image_id"))
                bb = a.get("bbox")
                if name not in data["frames"] or not bb or len(bb) != 4:
                    continue
                x, y, w, h = bb
                label = cats.get(a.get("category_id"), "person")
                fresh_objects.setdefault(name, []).append({
                    "id": a.get("object_id") or new_obj_id(),
                    "bbox": [x, y, x + w, y + h],
                    "label": label if label in OBJECT_LABELS else "other",
                    "confirmed": True,
                    "source": a.get("source") or "import",
                    "conf": a.get("score"),
                    "is_poi": bool(a.get("is_poi")) or label == "person_of_interest",
                    "poi_locked": bool(a.get("poi_locked")),
                })
            for name, objs in fresh_objects.items():
                data["frames"][name]["objects"] = [n for n in (normalize_object(o) for o in objs) if n]
        else:
            for name, slot in (payload.get("frames") or {}).items():
                if name in data["frames"]:
                    data["frames"][name] = normalize_slot(slot)
        for slot in data["frames"].values():
            sync_poi(slot)
        persist(vid, data, coco_now=True)
    return {"ok": True, "format": "coco" if is_coco else "annotations", "video_id": vid}


# ---- Export -----------------------------------------------------------------

@app.get("/api/export/{vid}")
def api_export(vid: str):
    rec = video_record(vid)
    (export_dir() / f"{vid}_annotations.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_download(rec, f"{vid}_annotations.json")


@app.get("/api/export-coco/{vid}")
def api_export_coco(vid: str):
    with vlock(vid):
        data = load_annotations(vid)
        coco = build_coco(vid, data)
        atomic_write_text(coco_path(video_dir(vid), vid), json.dumps(coco, indent=2, ensure_ascii=False))
    (export_dir() / f"{vid}{COCO_SUFFIX}").write_text(
        json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_download(coco, f"{vid}{COCO_SUFFIX}")


@app.post("/api/export-all")
def api_export_all():
    return export_all()


@app.get("/api/exports/{name}")
def api_export_file(name: str):
    if not is_safe_name(name):
        raise HTTPException(400, "Invalid file name")
    p = export_dir() / name
    if not p.is_file():
        raise HTTPException(404, "Export not found — run Export all first")
    return FileResponse(p, filename=name)


@app.post("/api/open-exports")
def api_open_exports():
    folder = export_dir()
    try:
        if sys.platform.startswith("win"):
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Could not open folder: {e}") from e
    return {"ok": True, "folder": str(folder)}


STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main():
    parser = argparse.ArgumentParser(description="Trauma Behaviour Video Annotator")
    parser.add_argument("--data", default=str(APP_DIR / "data"), help="Folder that stores extracted videos")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    set_data_dir(args.data)
    print(f"Data folder : {DATA_DIR}")
    print(f"Extraction  : ffmpeg={ffmpeg_available()} opencv={opencv_available()}")
    print(f"Open        : http://{args.host}:{args.port}/")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
