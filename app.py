"""
Trauma Behaviour Video Annotator — backend.

Local annotation server. Videos are folders of ordered frames. Supports:
  - Import MP4 (extract frames in-app) and frame/zip upload with conflict handling
  - Multi-select behaviour labels + Trauma / No-Trauma video label
  - Person-of-interest bounding boxes (manual, YOLO detect, track forward)
  - Track options: box / behaviours / comment, range, overwrite modes
  - Dual write: annotations.json + COCO mirror

Run:  py app.py  [--data ./data] [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

BEHAVIOURS = ["flashback", "avoidance", "negative_emotion", "hyper_arousal", "normal"]
VIDEO_LABELS = ["trauma", "no_trauma"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
CONTEXT_NAMES = ["context.txt", "scenario.txt", "background.txt"]
ANNOTATION_FILE = "annotations.json"
COCO_SUFFIX = "_coco.json"
UNDO_FILE = "annotations.undo.json"

DATA_DIR = Path("data")

_io_lock = threading.Lock()
_yolo_models: dict[str, Any] = {}
_yolo_error: str | None = None
_extract_jobs: dict[str, dict] = {}
DEFAULT_YOLO = "yolo11n.pt"
YOLO_CHOICES = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def is_safe_name(name: str) -> bool:
    return name not in ("", ".", "..") and "/" not in name and "\\" not in name


def sanitize_video_id(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^\w\-]+", "_", stem).strip("_")
    return stem or f"video_{int(time.time())}"


def unique_video_id(base: str) -> str:
    candidate = base
    n = 2
    while (DATA_DIR / candidate).exists():
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def video_dir(vid: str) -> Path:
    if not is_safe_name(vid):
        raise HTTPException(400, "Invalid video id")
    d = DATA_DIR / vid
    if not d.is_dir():
        raise HTTPException(404, f"Video '{vid}' not found")
    return d


def list_frames(d: Path) -> list[str]:
    names = [p.name for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(names, key=natural_key)


def read_context_file(d: Path) -> str:
    for name in CONTEXT_NAMES:
        p = d / name
        if p.is_file():
            try:
                return p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return ""
    return ""


def annotation_path(d: Path) -> Path:
    return d / ANNOTATION_FILE


OBJECT_LABELS = ["person_of_interest", "person", "other"]
_obj_id_seq = itertools.count(1)


def empty_slot() -> dict:
    return {"behaviours": [], "comment": "", "bbox": None, "objects": []}


def _new_obj_id() -> str:
    # Must be unique even when many boxes are created in the same millisecond
    return f"obj_{int(time.time() * 1000)}_{next(_obj_id_seq)}_{uuid.uuid4().hex[:8]}"


def normalize_object(obj: dict) -> dict | None:
    if not isinstance(obj, dict):
        return None
    bb = obj.get("bbox")
    if not bb or len(bb) != 4:
        return None
    behaviours = [b for b in (obj.get("behaviours") or []) if b in BEHAVIOURS]
    label = obj.get("label") or "person"
    if label not in OBJECT_LABELS and not isinstance(label, str):
        label = "person"
    return {
        "id": str(obj.get("id") or _new_obj_id()),
        "bbox": [float(v) for v in bb],
        "label": label,
        "behaviours": behaviours,
        "confirmed": bool(obj.get("confirmed", False)),
        "source": obj.get("source") or "manual",
        "conf": float(obj["conf"]) if obj.get("conf") is not None else None,
        "is_poi": bool(obj.get("is_poi", False)),
    }


def ensure_unique_object_ids(objects: list[dict]) -> list[dict]:
    """Repair duplicate ids so deleting one box cannot wipe several."""
    seen: set[str] = set()
    for o in objects:
        oid = str(o.get("id") or "")
        if not oid or oid in seen:
            o["id"] = _new_obj_id()
        seen.add(str(o["id"]))
    return objects


def sync_poi_bbox(slot: dict) -> dict:
    """Exactly one POI (red). Others are never person_of_interest / is_poi."""
    objects = slot.get("objects") or []
    ensure_unique_object_ids(objects)

    poi = next((o for o in objects if o.get("is_poi") and o.get("bbox")), None)
    if poi is None:
        poi = next(
            (o for o in objects if o.get("label") == "person_of_interest" and o.get("bbox")),
            None,
        )
    if poi is None:
        # Do NOT auto-promote every confirmed person — only if a single confirmed exists
        confirmed = [o for o in objects if o.get("confirmed") and o.get("bbox")]
        if len(confirmed) == 1:
            poi = confirmed[0]

    if poi is not None:
        poi_id = str(poi["id"])
        slot["bbox"] = list(poi["bbox"])
        for o in objects:
            is_poi = str(o["id"]) == poi_id
            o["is_poi"] = is_poi
            if is_poi:
                o["label"] = "person_of_interest"
                o["confirmed"] = True
            elif o.get("label") == "person_of_interest":
                o["label"] = "person"
    else:
        slot["bbox"] = None
        for o in objects:
            o["is_poi"] = False
            if o.get("label") == "person_of_interest":
                o["label"] = "person"
    slot["objects"] = objects
    return slot


def normalize_slot(slot: dict) -> dict:
    """Migrate legacy single `behaviour` → `behaviours` list; ensure objects[]."""
    if not isinstance(slot, dict):
        return empty_slot()
    out = dict(slot)
    if "behaviours" not in out:
        legacy = out.pop("behaviour", None)
        if legacy and isinstance(legacy, str):
            out["behaviours"] = [legacy]
        elif isinstance(legacy, list):
            out["behaviours"] = [b for b in legacy if b in BEHAVIOURS]
        else:
            out["behaviours"] = []
    else:
        out["behaviours"] = [b for b in (out.get("behaviours") or []) if b in BEHAVIOURS]
    out.setdefault("comment", "")
    out.setdefault("bbox", None)
    out.pop("behaviour", None)

    had_objects_key = "objects" in out
    objects = []
    for obj in out.get("objects") or []:
        n = normalize_object(obj)
        if n:
            objects.append(n)
    # Seed from legacy single bbox ONLY when migrating old files (no objects key yet)
    if not objects and not had_objects_key and out.get("bbox") and len(out["bbox"]) == 4:
        objects.append({
            "id": _new_obj_id(),
            "bbox": [float(v) for v in out["bbox"]],
            "label": "person_of_interest",
            "behaviours": list(out.get("behaviours") or []),
            "confirmed": True,
            "source": "manual",
            "conf": None,
            "is_poi": True,
        })
    out["objects"] = objects
    sync_poi_bbox(out)
    return out


def load_annotations(vid: str) -> dict:
    d = video_dir(vid)
    frames = list_frames(d)
    path = annotation_path(d)

    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
    else:
        data = {}

    data.setdefault("video_id", vid)
    data.setdefault("video_label", None)
    data.setdefault("video_comment", "")
    if "context" not in data:
        data["context"] = read_context_file(d)
    data.setdefault("frames", {})
    data.setdefault("frame_dims", {})

    normalized = {}
    for name, slot in data["frames"].items():
        normalized[name] = normalize_slot(slot)
    data["frames"] = normalized

    for name in frames:
        data["frames"].setdefault(name, empty_slot())

    data["_frame_order"] = frames
    return data


def save_annotations(vid: str, data: dict) -> None:
    d = video_dir(vid)
    out = {k: v for k, v in data.items() if not k.startswith("_")}
    with _io_lock:
        annotation_path(d).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def coco_path(d: Path, vid: str) -> Path:
    return d / f"{vid}{COCO_SUFFIX}"


def _frame_size(d: Path, name: str, data: dict) -> list[int]:
    dims = data.setdefault("frame_dims", {})
    if name in dims and isinstance(dims[name], list) and len(dims[name]) == 2:
        return dims[name]
    try:
        with Image.open(d / name) as im:
            wh = [int(im.width), int(im.height)]
    except Exception:
        wh = [0, 0]
    dims[name] = wh
    return wh


def build_coco(vid: str, data: dict) -> dict:
    d = video_dir(vid)
    frames = data.get("_frame_order") or list_frames(d)
    cat_map = {name: i + 1 for i, name in enumerate(OBJECT_LABELS)}

    images, annotations = [], []
    ann_id = 1
    for i, name in enumerate(frames):
        w, h = _frame_size(d, name, data)
        slot = normalize_slot(data["frames"].get(name, {}))
        behaviours = slot.get("behaviours") or []
        images.append({
            "id": i + 1,
            "file_name": name,
            "width": w,
            "height": h,
            "behaviours": behaviours,
            "behaviour": behaviours[0] if len(behaviours) == 1 else None,
            "comment": slot.get("comment", ""),
        })
        objs = [o for o in (slot.get("objects") or []) if o.get("confirmed") or o.get("is_poi")]
        if not objs and slot.get("bbox") and len(slot["bbox"]) == 4:
            objs = [{
                "bbox": slot["bbox"],
                "label": "person_of_interest",
                "behaviours": behaviours,
                "id": "legacy",
            }]
        for obj in objs:
            bb = obj.get("bbox")
            if not bb or len(bb) != 4:
                continue
            x1, y1, x2, y2 = bb
            label = obj.get("label") or "person_of_interest"
            if label not in cat_map:
                label = "other"
            obj_beh = obj.get("behaviours") or behaviours
            annotations.append({
                "id": ann_id,
                "image_id": i + 1,
                "category_id": cat_map[label],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                "iscrowd": 0,
                "behaviours": obj_beh,
                "behaviour": obj_beh[0] if len(obj_beh) == 1 else None,
                "object_id": obj.get("id"),
                "is_poi": bool(obj.get("is_poi")),
            })
            ann_id += 1

    return {
        "info": {
            "description": f"Annotated objects for {vid}",
            "video_id": vid,
            "video_label": data.get("video_label"),
            "video_comment": data.get("video_comment", ""),
            "context": data.get("context", ""),
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in sorted(cat_map.items(), key=lambda x: x[1])],
    }


def write_coco_file(vid: str, data: dict, coco: dict) -> None:
    d = video_dir(vid)
    with _io_lock:
        coco_path(d, vid).write_text(
            json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def persist(vid: str, data: dict, *, write_coco: bool = True) -> None:
    """Save annotations. COCO rebuild is optional — skip on frequent UI edits for speed."""
    if write_coco:
        coco = None
        try:
            coco = build_coco(vid, data)
        except Exception as e:  # noqa: BLE001
            print(f"[coco] build failed for {vid}: {e}")
        save_annotations(vid, data)
        if coco is not None:
            try:
                write_coco_file(vid, data, coco)
            except Exception as e:  # noqa: BLE001
                print(f"[coco] write failed for {vid}: {e}")
    else:
        save_annotations(vid, data)


def ensure_coco(vid: str, data: dict) -> None:
    d = video_dir(vid)
    if not coco_path(d, vid).is_file():
        persist(vid, data)


def frame_path(vid: str, idx: int) -> Path:
    d = video_dir(vid)
    frames = list_frames(d)
    if idx < 0 or idx >= len(frames):
        raise HTTPException(404, "Frame index out of range")
    return d / frames[idx]


def backup_annotations(d: Path) -> str | None:
    path = annotation_path(d)
    if not path.is_file():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = d / f"annotations.backup.{stamp}.json"
    shutil.copy2(path, dest)
    return dest.name


def snapshot_for_undo(vid: str, data: dict, frame_names: list[str]) -> None:
    """Save prior slots for an undo of the last track."""
    d = video_dir(vid)
    snap = {
        "video_id": vid,
        "ts": time.time(),
        "frames": {n: normalize_slot(data["frames"].get(n, {})) for n in frame_names},
    }
    with _io_lock:
        (d / UNDO_FILE).write_text(json.dumps(snap), encoding="utf-8")


def annotation_stats(vid: str) -> dict:
    d = DATA_DIR / vid
    if not d.is_dir():
        return {"exists": False, "num_frames": 0, "has_annotations": False, "labelled": 0, "boxed": 0}
    frames = list_frames(d)
    has_ann = annotation_path(d).is_file()
    labelled = boxed = 0
    if has_ann or frames:
        try:
            data = load_annotations(vid) if frames else {"frames": {}}
            for name in frames:
                slot = data["frames"].get(name, {})
                if slot.get("behaviours"):
                    labelled += 1
                if slot.get("bbox"):
                    boxed += 1
        except HTTPException:
            pass
    return {
        "exists": True,
        "num_frames": len(frames),
        "has_annotations": has_ann,
        "labelled": labelled,
        "boxed": boxed,
    }


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def opencv_available() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Frame extraction
# ----------------------------------------------------------------------------

def clear_frames(d: Path) -> None:
    for p in d.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            try:
                p.unlink()
            except OSError:
                pass


def extract_with_ffmpeg(video_path: Path, out_dir: Path, fps: float | None, job_id: str) -> int:
    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if fps and fps > 0:
        cmd += ["-vf", f"fps={fps}"]
    cmd += ["-q:v", "2", pattern]
    _extract_jobs[job_id]["status"] = "extracting"
    _extract_jobs[job_id]["engine"] = "ffmpeg"
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-800:] if proc.stderr else "ffmpeg failed")
    return len(list_frames(out_dir))


def extract_with_opencv(video_path: Path, out_dir: Path, fps: float | None, job_id: str) -> int:
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video with OpenCV")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    interval = 1.0
    if fps and fps > 0 and src_fps > 0:
        interval = max(1.0, src_fps / fps)

    _extract_jobs[job_id]["status"] = "extracting"
    _extract_jobs[job_id]["engine"] = "opencv"
    _extract_jobs[job_id]["total_hint"] = total

    idx = 0
    written = 0
    next_take = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= next_take:
            written += 1
            out = out_dir / f"frame_{written:06d}.jpg"
            cv2.imwrite(str(out), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            next_take += interval
            _extract_jobs[job_id]["written"] = written
            if total:
                _extract_jobs[job_id]["progress"] = min(99, int(100 * idx / total))
        idx += 1

    cap.release()
    return written


def extract_frames(video_path: Path, out_dir: Path, fps: float | None, job_id: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if ffmpeg_available():
        try:
            n = extract_with_ffmpeg(video_path, out_dir, fps, job_id)
            if n > 0:
                return n
        except Exception as e:  # noqa: BLE001
            print(f"[extract] ffmpeg failed, trying OpenCV: {e}")
    if opencv_available():
        return extract_with_opencv(video_path, out_dir, fps, job_id)
    raise RuntimeError(
        "No frame extractor available. Install ffmpeg or `pip install opencv-python`."
    )


# ----------------------------------------------------------------------------
# YOLO + tracker
# ----------------------------------------------------------------------------

def get_yolo(weights: str = DEFAULT_YOLO):
    """Load (and cache) a YOLO weight file. Lying / bed poses need lower conf + larger imgsz."""
    global _yolo_error
    name = weights if weights in YOLO_CHOICES else DEFAULT_YOLO
    if name in _yolo_models:
        return _yolo_models[name], None
    try:
        from ultralytics import YOLO  # type: ignore
        _yolo_models[name] = YOLO(name)
        _yolo_error = None
        return _yolo_models[name], None
    except Exception as e:  # noqa: BLE001
        _yolo_error = (
            f"YOLO unavailable: {e}. Install with `pip install ultralytics` "
            "to enable detection and tracking."
        )
        return None, _yolo_error


def detect_persons(
    model,
    img_path: Path,
    conf: float = 0.35,
    imgsz: int = 960,
    augment: bool = False,
) -> list[dict]:
    """Detect people. Default min confidence 35%. Augment off by default (faster)."""
    min_conf = max(0.35, float(conf))  # never accept below 35%
    res = model.predict(
        str(img_path),
        classes=[0],
        conf=min_conf,
        imgsz=max(320, min(1920, int(imgsz))),
        augment=bool(augment),
        verbose=False,
    )
    out: list[dict] = []
    if not res:
        return out
    for b in res[0].boxes:
        score = float(b.conf[0])
        if score < 0.35:
            continue
        xyxy = [float(v) for v in b.xyxy[0].tolist()]
        cls_id = int(b.cls[0]) if b.cls is not None else 0
        out.append({
            "id": _new_obj_id(),
            "box": xyxy,
            "bbox": xyxy,
            "conf": score,
            "cls": cls_id,
            "label": "person",
            "source": "detection",
            "confirmed": False,
            "is_poi": False,
            "behaviours": [],
        })
    out.sort(key=lambda d: d["conf"], reverse=True)
    return out


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid_dist(a: list[float], b: list[float]) -> float:
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _make_csrt():
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    for factory in (
        getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None),
        getattr(cv2, "TrackerCSRT_create", None),
    ):
        if callable(factory):
            try:
                return factory()
            except Exception:
                continue
    return None


class ForwardTracker:
    def __init__(self, seed_box, iou_thresh=0.15, max_gap=8):
        self.box = [float(v) for v in seed_box]
        self.iou_thresh = iou_thresh
        self.max_gap = max_gap
        self.gap = 0
        self.csrt = None
        self._cv2 = None

    def _match(self, dets: list[dict]):
        best, best_iou = None, 0.0
        for d in dets:
            i = iou(self.box, d["box"])
            if i > best_iou:
                best, best_iou = d, i
        if best is not None and best_iou >= self.iou_thresh:
            return best["box"]
        w = self.box[2] - self.box[0]
        h = self.box[3] - self.box[1]
        limit = 1.5 * max(w, h)
        near, near_d = None, limit
        for d in dets:
            dist = centroid_dist(self.box, d["box"])
            if dist < near_d:
                near, near_d = d, dist
        return near["box"] if near is not None else None

    def _csrt_update(self, img_path: Path):
        if self._cv2 is None:
            try:
                import cv2  # type: ignore
                self._cv2 = cv2
            except Exception:
                return None
        cv2 = self._cv2
        frame = cv2.imread(str(img_path))
        if frame is None:
            return None
        if self.csrt is None:
            self.csrt = _make_csrt()
            if self.csrt is None:
                return None
            x1, y1, x2, y2 = self.box
            self.csrt.init(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            return None
        ok, bb = self.csrt.update(frame)
        if not ok:
            return None
        x, y, w, h = bb
        return [float(x), float(y), float(x + w), float(y + h)]

    def step(self, model, img_path: Path):
        dets = detect_persons(model, img_path)
        matched = self._match(dets)
        if matched is not None:
            self.box = matched
            self.gap = 0
            self.csrt = None
            return matched, "detection"
        est = self._csrt_update(img_path)
        self.gap += 1
        if est is not None:
            self.box = est
            if self.gap > self.max_gap:
                return None, "lost"
            return est, "tracker"
        if self.gap > self.max_gap:
            return None, "lost"
        return self.box, "hold"


# ----------------------------------------------------------------------------
# API models
# ----------------------------------------------------------------------------

class FrameAnnotation(BaseModel):
    behaviours: list[str] | None = None
    behaviour: str | None = None  # legacy single
    comment: str | None = None
    bbox: list[float] | None = None
    objects: list[dict] | None = None


class VideoMeta(BaseModel):
    video_label: str | None = None
    video_comment: str | None = None
    context: str | None = None


class DetectRequest(BaseModel):
    conf: float = 0.35
    imgsz: int = 960
    augment: bool = False
    model: str = DEFAULT_YOLO


class ApplyObjectsRequest(BaseModel):
    """Confirm / save an object list on the current frame, optionally copy to following frames."""
    objects: list[dict]
    num_frames: int = 1          # 1 = this frame only; >1 copies boxes forward statically
    mode: str = "replace"        # replace | merge
    set_frame_behaviours: bool = False


class TrackRequest(BaseModel):
    seed_box: list[float] | None = None
    num_frames: int = 0
    propagate_bbox: bool = True
    propagate_behaviours: bool = True
    propagate_comment: bool = False
    overwrite_bbox: bool = True
    overwrite_labels: bool = True
    stop_on_label_change: bool = False
    max_gap: int = 8
    iou_thresh: float = 0.15
    dry_run: bool = False


class ProbeRequest(BaseModel):
    filename: str
    video_id: str | None = None


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------

app = FastAPI(title="Trauma Behaviour Video Annotator")


@app.get("/api/config")
def api_config():
    import importlib.util
    yolo_ready = importlib.util.find_spec("ultralytics") is not None
    return {
        "behaviours": BEHAVIOURS,
        "video_labels": VIDEO_LABELS,
        "object_labels": OBJECT_LABELS,
        "yolo_models": YOLO_CHOICES,
        "yolo_ready": yolo_ready,
        "ffmpeg_ready": ffmpeg_available(),
        "opencv_ready": opencv_available(),
        "extract_ready": ffmpeg_available() or opencv_available(),
        "detect_defaults": {"conf": 0.35, "imgsz": 960, "augment": False, "model": DEFAULT_YOLO},
    }


@app.get("/api/videos")
def api_videos():
    if not DATA_DIR.is_dir():
        return {"videos": []}
    out = []
    for d in sorted([p for p in DATA_DIR.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
        frames = list_frames(d)
        if not frames:
            continue
        data = load_annotations(d.name)
        labelled = sum(1 for f in frames if data["frames"].get(f, {}).get("behaviours"))
        boxed = sum(1 for f in frames if data["frames"].get(f, {}).get("bbox"))
        out.append({
            "id": d.name,
            "num_frames": len(frames),
            "labelled": labelled,
            "boxed": boxed,
            "video_label": data.get("video_label"),
        })
    return {"videos": out}


@app.get("/api/video/{vid}")
def api_video(vid: str):
    data = load_annotations(vid)
    ensure_coco(vid, data)
    return {
        "video_id": data["video_id"],
        "frames": data["_frame_order"],
        "annotations": data["frames"],
        "video_label": data.get("video_label"),
        "video_comment": data.get("video_comment", ""),
        "context": data.get("context", ""),
        "has_undo": (video_dir(vid) / UNDO_FILE).is_file(),
    }


@app.get("/api/frame/{vid}/{idx}")
def api_frame(vid: str, idx: int):
    p = frame_path(vid, idx)
    media = "image/jpeg"
    ext = p.suffix.lower()
    if ext == ".png":
        media = "image/png"
    elif ext == ".webp":
        media = "image/webp"
    elif ext == ".bmp":
        media = "image/bmp"
    return FileResponse(p, media_type=media)


@app.post("/api/annotate/{vid}/{idx}")
def api_annotate(vid: str, idx: int, ann: FrameAnnotation):
    data = load_annotations(vid)
    frames = data["_frame_order"]
    if idx < 0 or idx >= len(frames):
        raise HTTPException(404, "Frame index out of range")
    name = frames[idx]
    slot = normalize_slot(data["frames"].setdefault(name, empty_slot()))

    if ann.behaviours is not None:
        bad = [b for b in ann.behaviours if b not in BEHAVIOURS]
        if bad:
            raise HTTPException(400, f"Unknown behaviours: {bad}")
        behaviours = list(dict.fromkeys(ann.behaviours))
        if "normal" in behaviours and len(behaviours) > 1:
            behaviours = ["normal"]
        slot["behaviours"] = behaviours
    elif ann.behaviour is not None:
        if ann.behaviour == "":
            slot["behaviours"] = []
        elif ann.behaviour in BEHAVIOURS:
            slot["behaviours"] = [ann.behaviour]
        else:
            raise HTTPException(400, f"Unknown behaviour '{ann.behaviour}'")

    if ann.comment is not None:
        slot["comment"] = ann.comment
    if ann.objects is not None:
        objects = []
        for obj in ann.objects:
            n = normalize_object(obj)
            if n:
                objects.append(n)
        slot["objects"] = objects
        sync_poi_bbox(slot)
    elif ann.bbox is not None:
        slot["bbox"] = ann.bbox if len(ann.bbox) == 4 else None
        # Keep / update POI object to match
        if slot["bbox"]:
            poi = next((o for o in slot.get("objects") or [] if o.get("is_poi")), None)
            if poi:
                poi["bbox"] = list(slot["bbox"])
            else:
                slot.setdefault("objects", []).append({
                    "id": _new_obj_id(),
                    "bbox": list(slot["bbox"]),
                    "label": "person_of_interest",
                    "behaviours": list(slot.get("behaviours") or []),
                    "confirmed": True,
                    "source": "manual",
                    "conf": None,
                    "is_poi": True,
                })
                sync_poi_bbox(slot)
        else:
            slot["objects"] = [o for o in (slot.get("objects") or []) if not o.get("is_poi")]

    data["frames"][name] = normalize_slot(slot)
    # Fast path: do not rebuild COCO on every keystroke / object edit
    persist(vid, data, write_coco=False)
    return {"ok": True, "frame": name, "annotation": data["frames"][name]}


@app.post("/api/clear-bbox/{vid}/{idx}")
def api_clear_bbox(vid: str, idx: int):
    data = load_annotations(vid)
    frames = data["_frame_order"]
    if idx < 0 or idx >= len(frames):
        raise HTTPException(404, "Frame index out of range")
    slot = normalize_slot(data["frames"][frames[idx]])
    slot["bbox"] = None
    slot["objects"] = [o for o in slot.get("objects") or [] if not o.get("is_poi")]
    for o in slot["objects"]:
        o["is_poi"] = False
    data["frames"][frames[idx]] = slot
    persist(vid, data, write_coco=False)
    return {"ok": True}


@app.post("/api/video-meta/{vid}")
def api_video_meta(vid: str, meta: VideoMeta):
    data = load_annotations(vid)
    if meta.video_label is not None:
        if meta.video_label == "":
            data["video_label"] = None
        elif meta.video_label in VIDEO_LABELS:
            data["video_label"] = meta.video_label
        else:
            raise HTTPException(400, f"Unknown video label '{meta.video_label}'")
    if meta.video_comment is not None:
        data["video_comment"] = meta.video_comment
    if meta.context is not None:
        data["context"] = meta.context
    persist(vid, data, write_coco=False)
    return {"ok": True}


@app.post("/api/write-context/{vid}")
def api_write_context(vid: str):
    d = video_dir(vid)
    data = load_annotations(vid)
    target = None
    for name in CONTEXT_NAMES:
        if (d / name).is_file():
            target = d / name
            break
    if target is None:
        target = d / CONTEXT_NAMES[0]
    with _io_lock:
        target.write_text(data.get("context", ""), encoding="utf-8")
    return {"ok": True, "file": target.name}


@app.post("/api/detect/{vid}/{idx}")
def api_detect(vid: str, idx: int, req: DetectRequest):
    model, err = get_yolo(req.model)
    if err:
        raise HTTPException(503, err)
    p = frame_path(vid, idx)
    dets = detect_persons(
        model, p, conf=req.conf, imgsz=req.imgsz, augment=req.augment
    )
    tip = None
    if not dets:
        tip = (
            "No person ≥35% confidence. Try a larger model (small/medium), "
            "raise Size to 1280+, keep Augment on, or draw the box manually."
        )
    return {
        "detections": dets,
        "count": len(dets),
        "settings": {
            "conf": req.conf,
            "imgsz": req.imgsz,
            "augment": req.augment,
            "model": req.model if req.model in YOLO_CHOICES else DEFAULT_YOLO,
        },
        "tip": tip,
    }


@app.post("/api/apply-objects/{vid}/{idx}")
def api_apply_objects(vid: str, idx: int, req: ApplyObjectsRequest):
    """Save confirmed object list on this frame; optionally copy to following frames."""
    data = load_annotations(vid)
    frames = data["_frame_order"]
    if idx < 0 or idx >= len(frames):
        raise HTTPException(404, "Frame index out of range")

    objects = []
    for obj in req.objects:
        n = normalize_object(obj)
        if n:
            # Applying means these are kept — mark confirmed unless explicitly false
            if "confirmed" not in obj:
                n["confirmed"] = True
            objects.append(n)
    if not objects:
        raise HTTPException(400, "No valid objects to apply")

    # Exactly one POI preferred
    if not any(o.get("is_poi") for o in objects):
        poi = next((o for o in objects if o.get("label") == "person_of_interest"), None)
        if poi is None:
            poi = objects[0]
            poi["label"] = "person_of_interest"
        poi["is_poi"] = True

    end = min(len(frames), idx + max(1, req.num_frames))
    covered = []
    snapshot_for_undo(vid, data, frames[idx:end])

    for i in range(idx, end):
        name = frames[i]
        slot = normalize_slot(data["frames"][name])
        # Keep the SAME object id across frames so one person = one track id
        cloned = []
        for o in objects:
            c = dict(o)
            c["id"] = str(o.get("id") or _new_obj_id())
            if i != idx:
                c["source"] = "copied"
            cloned.append(c)
        if req.mode == "merge":
            existing = [o for o in (slot.get("objects") or []) if o.get("confirmed")]
            # drop overlapping existing (IoU > 0.5) OR same id already present
            clone_ids = {str(c["id"]) for c in cloned}
            kept = []
            for e in existing:
                if str(e.get("id")) in clone_ids:
                    continue
                if any(iou(e["bbox"], c["bbox"]) > 0.5 for c in cloned):
                    continue
                kept.append(e)
            slot["objects"] = kept + cloned
        else:
            slot["objects"] = cloned
        if req.set_frame_behaviours:
            # Union of object behaviours onto frame
            beh = []
            for o in slot["objects"]:
                for b in o.get("behaviours") or []:
                    if b not in beh:
                        beh.append(b)
            if beh:
                slot["behaviours"] = beh
        sync_poi_bbox(slot)
        data["frames"][name] = normalize_slot(slot)
        covered.append(name)

    persist(vid, data)
    return {
        "ok": True,
        "covered": len(covered),
        "frames": covered,
        "annotation": data["frames"][frames[idx]],
        "has_undo": True,
    }


@app.post("/api/track/{vid}/{idx}")
def api_track(vid: str, idx: int, req: TrackRequest):
    if not (req.propagate_bbox or req.propagate_behaviours or req.propagate_comment):
        raise HTTPException(400, "Select at least one propagate option")

    data = load_annotations(vid)
    frames = data["_frame_order"]
    d = video_dir(vid)
    if idx < 0 or idx >= len(frames):
        raise HTTPException(404, "Frame index out of range")

    seed_name = frames[idx]
    seed_slot = normalize_slot(data["frames"][seed_name])
    seed_box = req.seed_box or seed_slot.get("bbox")
    if req.propagate_bbox and (not seed_box or len(seed_box) != 4):
        raise HTTPException(400, "Need a seed bounding box to propagate boxes")

    seed_behaviours = list(seed_slot.get("behaviours") or [])
    seed_comment = seed_slot.get("comment", "")

    # Stable track id for the POI — same person keeps the same id across frames
    seed_objects = list(seed_slot.get("objects") or [])
    poi_obj = next((o for o in seed_objects if o.get("is_poi")), None)
    if poi_obj is None:
        poi_obj = next(
            (o for o in seed_objects if o.get("label") == "person_of_interest"),
            None,
        )
    if poi_obj is None and seed_box:
        poi_obj = {
            "id": _new_obj_id(),
            "bbox": [float(v) for v in seed_box],
            "label": "person_of_interest",
            "behaviours": list(seed_behaviours),
            "confirmed": True,
            "source": "seed",
            "conf": None,
            "is_poi": True,
        }
        seed_objects.append(poi_obj)
    track_id = str(poi_obj["id"]) if poi_obj else _new_obj_id()
    if poi_obj is not None:
        poi_obj["id"] = track_id
        poi_obj["is_poi"] = True
        poi_obj["label"] = "person_of_interest"
        poi_obj["confirmed"] = True
        if seed_box and len(seed_box) == 4:
            poi_obj["bbox"] = [float(v) for v in seed_box]

    def upsert_tracked_poi(frame_name: str, box: list[float] | None, source: str) -> None:
        """Write/update the tracked person with the SAME id on this frame."""
        if box is None or len(box) != 4:
            return
        slot = normalize_slot(data["frames"][frame_name])
        others = [
            o for o in (slot.get("objects") or [])
            if str(o.get("id")) != track_id
        ]
        for o in others:
            o["is_poi"] = False
            if o.get("label") == "person_of_interest":
                o["label"] = "person"
        tracked = {
            "id": track_id,
            "bbox": [float(v) for v in box],
            "label": "person_of_interest",
            "behaviours": list(
                ((poi_obj or {}).get("behaviours") or None) or seed_behaviours
            ),
            "confirmed": True,
            "source": source,
            "conf": (poi_obj or {}).get("conf"),
            "is_poi": True,
        }
        if req.propagate_behaviours and req.overwrite_labels:
            tracked["behaviours"] = list(seed_behaviours)
        slot["objects"] = others + [tracked]
        slot["bbox"] = list(tracked["bbox"])
        sync_poi_bbox(slot)
        data["frames"][frame_name] = normalize_slot(slot)

    end = len(frames) if req.num_frames <= 0 else min(len(frames), idx + 1 + req.num_frames)
    covered_names = frames[idx:end]

    if not req.dry_run:
        snapshot_for_undo(vid, data, covered_names)
        # Persist seed POI with stable id
        if poi_obj is not None:
            for o in seed_objects:
                if str(o.get("id")) == track_id:
                    o.update(poi_obj)
                    break
            else:
                seed_objects.append(poi_obj)
            for o in seed_objects:
                o["is_poi"] = str(o.get("id")) == track_id
                if not o["is_poi"] and o.get("label") == "person_of_interest":
                    o["label"] = "person"
            seed_slot["objects"] = seed_objects
            if seed_box:
                seed_slot["bbox"] = [float(v) for v in seed_box]
            sync_poi_bbox(seed_slot)
            data["frames"][seed_name] = normalize_slot(seed_slot)

    model = None
    tracker = None
    if req.propagate_bbox:
        model, err = get_yolo()
        if err:
            raise HTTPException(503, err)
        tracker = ForwardTracker(seed_box, iou_thresh=req.iou_thresh, max_gap=req.max_gap)
        if not req.dry_run:
            data["frames"][seed_name]["bbox"] = [float(v) for v in seed_box]
            upsert_tracked_poi(seed_name, seed_box, "seed")

    results = {}
    stop_reason = "end"
    stopped_at = end - 1

    # Seed frame result
    results[seed_name] = {
        "box": seed_box if req.propagate_bbox else seed_slot.get("bbox"),
        "source": "seed",
        "track_id": track_id,
        "behaviours_applied": False,
        "comment_applied": False,
    }
    if req.propagate_behaviours and (req.overwrite_labels or not seed_behaviours):
        if not req.dry_run and req.overwrite_labels:
            data["frames"][seed_name]["behaviours"] = list(seed_behaviours)
        results[seed_name]["behaviours_applied"] = True
    if req.propagate_comment and req.overwrite_labels:
        if not req.dry_run:
            data["frames"][seed_name]["comment"] = seed_comment
        results[seed_name]["comment_applied"] = True

    for i in range(idx + 1, end):
        name = frames[i]
        slot = normalize_slot(data["frames"][name])

        if req.stop_on_label_change and slot.get("behaviours"):
            if list(slot["behaviours"]) != list(seed_behaviours):
                stop_reason = "label_change"
                stopped_at = i - 1
                break

        entry: dict[str, Any] = {
            "box": None,
            "source": None,
            "track_id": track_id,
            "behaviours_applied": False,
            "comment_applied": False,
        }

        if req.propagate_bbox and tracker is not None and model is not None:
            existing = slot.get("bbox")
            # Prefer existing object with same track id
            same = next(
                (o for o in (slot.get("objects") or []) if str(o.get("id")) == track_id),
                None,
            )
            if same and same.get("bbox") and not req.overwrite_bbox:
                tracker.box = same["bbox"]
                entry["box"] = same["bbox"]
                entry["source"] = "kept"
            elif existing and not req.overwrite_bbox:
                tracker.box = existing
                entry["box"] = existing
                entry["source"] = "kept"
                if not req.dry_run:
                    upsert_tracked_poi(name, existing, "kept")
            else:
                box, source = tracker.step(model, d / name)
                if box is None:
                    stop_reason = "lost"
                    stopped_at = i - 1
                    break
                entry["box"] = box
                entry["source"] = source
                if not req.dry_run:
                    data["frames"][name]["bbox"] = [float(v) for v in box]
                    upsert_tracked_poi(name, box, source)

        if req.propagate_behaviours:
            existing_b = slot.get("behaviours") or []
            if existing_b and not req.overwrite_labels:
                pass
            else:
                entry["behaviours_applied"] = True
                if not req.dry_run:
                    data["frames"][name]["behaviours"] = list(seed_behaviours)
                    # Also keep object-level behaviours on the tracked id
                    objs = data["frames"][name].get("objects") or []
                    for o in objs:
                        if str(o.get("id")) == track_id:
                            o["behaviours"] = list(seed_behaviours)

        if req.propagate_comment:
            if slot.get("comment") and not req.overwrite_labels:
                pass
            else:
                entry["comment_applied"] = True
                if not req.dry_run:
                    data["frames"][name]["comment"] = seed_comment

        results[name] = entry
        stopped_at = i

    if not req.dry_run:
        persist(vid, data)

    return {
        "ok": True,
        "dry_run": req.dry_run,
        "results": results,
        "covered": len(results),
        "stopped_at": stopped_at,
        "stop_reason": stop_reason,
        "track_id": track_id,
        "has_undo": not req.dry_run,
    }


@app.post("/api/undo-track/{vid}")
def api_undo_track(vid: str):
    d = video_dir(vid)
    undo_path = d / UNDO_FILE
    if not undo_path.is_file():
        raise HTTPException(404, "Nothing to undo")
    try:
        snap = json.loads(undo_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise HTTPException(400, f"Corrupt undo file: {e}") from e

    data = load_annotations(vid)
    for name, slot in snap.get("frames", {}).items():
        if name in data["frames"]:
            data["frames"][name] = normalize_slot(slot)
    persist(vid, data)
    try:
        undo_path.unlink()
    except OSError:
        pass
    return {"ok": True, "restored": len(snap.get("frames", {}))}


# ---- Import: video / frames / json -----------------------------------------

@app.post("/api/import/video/probe")
def api_import_video_probe(req: ProbeRequest):
    base = sanitize_video_id(req.video_id or req.filename)
    exact = DATA_DIR / base
    stats = annotation_stats(base) if exact.is_dir() else {
        "exists": False, "num_frames": 0, "has_annotations": False, "labelled": 0, "boxed": 0
    }
    modes = ["create"]
    if stats["exists"]:
        modes = ["open"]
        if stats["num_frames"]:
            modes.append("reextract_keep")
            modes.append("reextract_wipe")
        else:
            modes.append("reextract_wipe")
    return {
        "video_id": base,
        "alternate_id": unique_video_id(base) if stats["exists"] else base,
        **stats,
        "suggested_modes": modes,
        "extract_ready": ffmpeg_available() or opencv_available(),
    }


@app.post("/api/import/video")
async def api_import_video(
    file: UploadFile = File(...),
    mode: str = Form("auto"),
    video_id: str | None = Form(None),
    fps: float | None = Form(None),
):
    if not (ffmpeg_available() or opencv_available()):
        raise HTTPException(
            503,
            "Cannot extract frames: install ffmpeg or opencv-python.",
        )

    raw_name = file.filename or "video.mp4"
    ext = Path(raw_name).suffix.lower()
    if ext not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported video type '{ext}'. Use mp4/mov/avi/mkv.")

    base = sanitize_video_id(video_id or raw_name)
    stats = annotation_stats(base) if (DATA_DIR / base).is_dir() else {
        "exists": False, "num_frames": 0, "has_annotations": False, "labelled": 0, "boxed": 0
    }

    # Resolve mode
    if mode == "auto":
        if not stats["exists"] or stats["num_frames"] == 0:
            mode = "create"
        elif stats["has_annotations"]:
            raise HTTPException(
                409,
                detail={
                    "message": "Video already annotated. Choose a mode.",
                    "probe": {**stats, "video_id": base, "suggested_modes": [
                        "open", "reextract_keep", "reextract_wipe", "create_new"
                    ]},
                },
            )
        else:
            raise HTTPException(
                409,
                detail={
                    "message": "Frames already exist. Choose a mode.",
                    "probe": {**stats, "video_id": base, "suggested_modes": [
                        "open", "reextract_keep", "reextract_wipe", "create_new"
                    ]},
                },
            )

    if mode == "open":
        if not stats["exists"] or stats["num_frames"] == 0:
            raise HTTPException(404, "Nothing to open")
        return {"ok": True, "mode": "open", "video_id": base, "num_frames": stats["num_frames"]}

    if mode == "create_new":
        base = unique_video_id(base)
        mode = "create"

    if mode in ("reextract_wipe", "reextract_keep", "create"):
        pass
    else:
        raise HTTPException(400, f"Unknown mode '{mode}'")

    out_dir = DATA_DIR / base
    out_dir.mkdir(parents=True, exist_ok=True)

    backup = None
    if mode == "reextract_wipe":
        backup = backup_annotations(out_dir)
        ann = annotation_path(out_dir)
        if ann.is_file():
            ann.unlink()
        clear_frames(out_dir)
    elif mode == "reextract_keep":
        backup = backup_annotations(out_dir)
        clear_frames(out_dir)
    elif mode == "create":
        if list_frames(out_dir):
            raise HTTPException(409, "Folder already has frames; use reextract or open")

    # Save upload then extract
    job_id = f"{base}_{int(time.time() * 1000)}"
    _extract_jobs[job_id] = {"status": "uploading", "progress": 0, "written": 0, "video_id": base}

    uploads = out_dir / "_uploads"
    uploads.mkdir(exist_ok=True)
    video_path = uploads / f"source{ext}"
    content = await file.read()
    video_path.write_bytes(content)
    _extract_jobs[job_id]["status"] = "uploaded"
    _extract_jobs[job_id]["progress"] = 5

    try:
        n = extract_frames(video_path, out_dir, fps if fps and fps > 0 else None, job_id)
    except Exception as e:  # noqa: BLE001
        _extract_jobs[job_id]["status"] = "error"
        _extract_jobs[job_id]["error"] = str(e)
        raise HTTPException(500, f"Extraction failed: {e}") from e

    if n <= 0:
        raise HTTPException(500, "No frames extracted from video")

    # Rebuild annotation slots; keep prior when reextract_keep
    data = load_annotations(base)
    if mode == "reextract_wipe":
        data["frames"] = {name: empty_slot() for name in data["_frame_order"]}
        data["video_label"] = None
        data["video_comment"] = ""
    persist(base, data)

    _extract_jobs[job_id]["status"] = "done"
    _extract_jobs[job_id]["progress"] = 100
    _extract_jobs[job_id]["written"] = n

    return {
        "ok": True,
        "mode": mode,
        "video_id": base,
        "num_frames": n,
        "job_id": job_id,
        "backup": backup,
        "engine": _extract_jobs[job_id].get("engine"),
    }


@app.get("/api/import/job/{job_id}")
def api_import_job(job_id: str):
    job = _extract_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job


@app.post("/api/import/frames")
async def api_import_frames(
    files: list[UploadFile] = File(...),
    mode: str = Form("auto"),
    video_id: str | None = Form(None),
):
    if not files:
        raise HTTPException(400, "No files uploaded")

    # Zip or loose images
    first = files[0]
    name0 = first.filename or "frames"
    base = sanitize_video_id(video_id or Path(name0).stem if len(files) == 1 else (video_id or "frames_import"))
    if len(files) == 1 and (first.filename or "").lower().endswith(".zip"):
        base = sanitize_video_id(video_id or Path(first.filename).stem)

    stats = annotation_stats(base) if (DATA_DIR / base).is_dir() else {
        "exists": False, "num_frames": 0, "has_annotations": False, "labelled": 0, "boxed": 0
    }

    if mode == "auto":
        if stats["exists"] and (stats["num_frames"] or stats["has_annotations"]):
            raise HTTPException(
                409,
                detail={
                    "message": "Target folder already has data. Choose a mode.",
                    "probe": {
                        **stats,
                        "video_id": base,
                        "alternate_id": unique_video_id(base),
                        "suggested_modes": ["open", "reextract_keep", "reextract_wipe", "create_new"],
                    },
                },
            )
        mode = "create"

    if mode == "open":
        return {"ok": True, "mode": "open", "video_id": base, "num_frames": stats["num_frames"]}

    if mode == "create_new":
        base = unique_video_id(base)
        mode = "create"

    out_dir = DATA_DIR / base
    out_dir.mkdir(parents=True, exist_ok=True)
    backup = None

    if mode == "reextract_wipe":
        backup = backup_annotations(out_dir)
        if annotation_path(out_dir).is_file():
            annotation_path(out_dir).unlink()
        clear_frames(out_dir)
    elif mode == "reextract_keep":
        backup = backup_annotations(out_dir)
        clear_frames(out_dir)
    elif mode != "create":
        raise HTTPException(400, f"Unknown mode '{mode}'")

    written = 0
    if len(files) == 1 and (files[0].filename or "").lower().endswith(".zip"):
        raw = await files[0].read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            imgs = [
                n for n in zf.namelist()
                if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS
                and not Path(n).name.startswith(".")
            ]
            imgs.sort(key=natural_key)
            for i, name in enumerate(imgs, 1):
                data_bytes = zf.read(name)
                ext = Path(name).suffix.lower()
                out = out_dir / f"frame_{i:06d}{ext}"
                out.write_bytes(data_bytes)
                written += 1
    else:
        imgs = []
        for f in files:
            fn = f.filename or ""
            if Path(fn).suffix.lower() in IMAGE_EXTS:
                imgs.append(f)
        imgs.sort(key=lambda f: natural_key(f.filename or ""))
        for i, f in enumerate(imgs, 1):
            ext = Path(f.filename or "x.jpg").suffix.lower()
            out = out_dir / f"frame_{i:06d}{ext}"
            out.write_bytes(await f.read())
            written += 1

    if written <= 0:
        raise HTTPException(400, "No image frames found in upload")

    data = load_annotations(base)
    if mode == "reextract_wipe":
        data["frames"] = {name: empty_slot() for name in data["_frame_order"]}
        data["video_label"] = None
        data["video_comment"] = ""
    persist(base, data)

    return {"ok": True, "mode": mode, "video_id": base, "num_frames": written, "backup": backup}


@app.post("/api/import/json")
async def api_import_json(
    file: UploadFile = File(...),
    video_id: str | None = Form(None),
    merge: bool = Form(True),
):
    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Invalid JSON: {e}") from e

    # Working format or COCO
    if "images" in payload and "annotations" in payload:
        vid = video_id or payload.get("info", {}).get("video_id") or sanitize_video_id(file.filename or "import")
        if not (DATA_DIR / vid).is_dir():
            raise HTTPException(404, f"Video folder '{vid}' not found — import frames first")
        data = load_annotations(vid)
        info = payload.get("info") or {}
        if info.get("video_label") in VIDEO_LABELS:
            data["video_label"] = info["video_label"]
        if "video_comment" in info:
            data["video_comment"] = info.get("video_comment") or ""
        if "context" in info:
            data["context"] = info.get("context") or ""

        id_to_name = {im["id"]: im["file_name"] for im in payload.get("images", [])}
        img_meta = {im["file_name"]: im for im in payload.get("images", [])}
        for name, im in img_meta.items():
            if name not in data["frames"]:
                continue
            slot = normalize_slot(data["frames"][name])
            behaviours = im.get("behaviours")
            if behaviours is None and im.get("behaviour"):
                behaviours = [im["behaviour"]]
            if behaviours is not None:
                slot["behaviours"] = [b for b in behaviours if b in BEHAVIOURS]
            if "comment" in im:
                slot["comment"] = im.get("comment") or ""
            data["frames"][name] = slot

        for ann in payload.get("annotations", []):
            name = id_to_name.get(ann.get("image_id"))
            if not name or name not in data["frames"]:
                continue
            bb = ann.get("bbox")
            if bb and len(bb) == 4:
                x, y, w, h = bb
                data["frames"][name]["bbox"] = [x, y, x + w, y + h]
            behaviours = ann.get("behaviours")
            if behaviours is None and ann.get("behaviour"):
                behaviours = [ann["behaviour"]]
            if behaviours is not None:
                data["frames"][name]["behaviours"] = [b for b in behaviours if b in BEHAVIOURS]

        persist(vid, data)
        return {"ok": True, "format": "coco", "video_id": vid}

    # Working annotations.json
    vid = video_id or payload.get("video_id") or sanitize_video_id(file.filename or "import")
    if not (DATA_DIR / vid).is_dir():
        raise HTTPException(404, f"Video folder '{vid}' not found — import frames first")
    data = load_annotations(vid)
    if not merge:
        data["frames"] = {n: empty_slot() for n in data["_frame_order"]}
    if payload.get("video_label") in VIDEO_LABELS or payload.get("video_label") is None:
        if "video_label" in payload:
            data["video_label"] = payload.get("video_label")
    if "video_comment" in payload:
        data["video_comment"] = payload.get("video_comment") or ""
    if "context" in payload:
        data["context"] = payload.get("context") or ""
    for name, slot in (payload.get("frames") or {}).items():
        if name in data["frames"]:
            data["frames"][name] = normalize_slot(slot)
    persist(vid, data)
    return {"ok": True, "format": "annotations", "video_id": vid}


@app.get("/api/export/{vid}")
def api_export(vid: str):
    data = load_annotations(vid)
    out = {k: v for k, v in data.items() if not k.startswith("_")}
    payload = json.dumps(out, indent=2, ensure_ascii=False)
    return StreamingResponse(
        io.BytesIO(payload.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{vid}_annotations.json"'},
    )


@app.get("/api/export-coco/{vid}")
def api_export_coco(vid: str):
    data = load_annotations(vid)
    coco = build_coco(vid, data)
    save_annotations(vid, data)
    payload = json.dumps(coco, indent=2, ensure_ascii=False)
    return StreamingResponse(
        io.BytesIO(payload.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{vid}{COCO_SUFFIX}"'},
    )


@app.post("/api/write-coco/{vid}")
def api_write_coco(vid: str):
    data = load_annotations(vid)
    persist(vid, data)
    d = video_dir(vid)
    return {"ok": True, "file": coco_path(d, vid).name}


@app.post("/api/write-coco-all")
def api_write_coco_all():
    written = []
    if DATA_DIR.is_dir():
        for d in sorted([p for p in DATA_DIR.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
            if not list_frames(d):
                continue
            persist(d.name, load_annotations(d.name))
            written.append(d.name)
    return {"ok": True, "written": written, "count": len(written)}


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


def main():
    global DATA_DIR
    parser = argparse.ArgumentParser(description="Trauma Behaviour Video Annotator")
    parser.add_argument("--data", default="data", help="Folder containing video subfolders")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    DATA_DIR = Path(args.data).resolve()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Serving annotations from: {DATA_DIR}")
    print(f"Open http://{args.host}:{args.port}/ in your browser.")
    print(f"Extract ready: ffmpeg={ffmpeg_available()} opencv={opencv_available()}")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
