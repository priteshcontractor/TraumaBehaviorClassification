"""
Trauma Behaviour Video Annotator — backend.

A local, single-user annotation server. Each "video" is a folder of ordered
image frames plus an optional context text file. Frames are labelled per-frame
(behaviour) and per-video (Trauma / No-Trauma), with a single "person of
interest" bounding box that can be drawn by hand, seeded from YOLO11 detections,
or propagated forward across frames by a single-object tracker.

The base tool runs with only FastAPI + Pillow. Detection/tracking activate
automatically if `ultralytics` (YOLO11) and OpenCV are installed.

Run:  python app.py  [--data ./data] [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

BEHAVIOURS = ["flashback", "avoidance", "negative_emotion", "hyper_arousal", "normal"]
VIDEO_LABELS = ["trauma", "no_trauma"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CONTEXT_NAMES = ["context.txt", "scenario.txt", "background.txt"]
ANNOTATION_FILE = "annotations.json"
COCO_SUFFIX = "_coco.json"  # written per video folder as "<vid>_coco.json"

DATA_DIR = Path("data")  # overridden by --data at startup

# A lock so concurrent saves to the same annotations file don't interleave.
_io_lock = threading.Lock()

# Lazily-loaded YOLO model (only imported when detection is first requested).
_yolo_model: Any | None = None
_yolo_error: str | None = None


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def natural_key(s: str) -> list:
    """Sort key so frame_2 comes before frame_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def is_safe_name(name: str) -> bool:
    """Reject path traversal in video ids."""
    return name not in ("", ".", "..") and "/" not in name and "\\" not in name


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


def load_annotations(vid: str) -> dict:
    """Load (or initialise) the annotation record for a video."""
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
    data.setdefault("video_label", None)          # "trauma" | "no_trauma" | None
    data.setdefault("video_comment", "")
    # Seed context from the sidecar text file the first time only.
    if "context" not in data:
        data["context"] = read_context_file(d)
    data.setdefault("frames", {})                  # frame_name -> {behaviour, comment, bbox}
    data.setdefault("frame_dims", {})              # frame_name -> [w, h] (cached, persisted)

    # Ensure every frame on disk has a slot; keep any existing values.
    for name in frames:
        data["frames"].setdefault(name, {"behaviour": None, "comment": "", "bbox": None})

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
    """Return [w, h] for a frame, caching the result in data['frame_dims']."""
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
    """Build a COCO detection dict from the annotation record.

    Single category 'person_of_interest'. Behaviour, per-frame comment and the
    video-level fields are carried as attributes so the COCO file is a complete
    record. Frame dimensions are read once and cached in data['frame_dims'].
    """
    d = video_dir(vid)
    frames = data.get("_frame_order") or list_frames(d)

    images, annotations = [], []
    ann_id = 1
    for i, name in enumerate(frames):
        w, h = _frame_size(d, name, data)
        slot = data["frames"].get(name, {})
        images.append({
            "id": i + 1,
            "file_name": name,
            "width": w,
            "height": h,
            "behaviour": slot.get("behaviour"),
            "comment": slot.get("comment", ""),
        })
        bb = slot.get("bbox")
        if bb and len(bb) == 4:
            x1, y1, x2, y2 = bb
            annotations.append({
                "id": ann_id,
                "image_id": i + 1,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],   # COCO xywh, pixels
                "area": max(0.0, x2 - x1) * max(0.0, y2 - y1),
                "iscrowd": 0,
                "behaviour": slot.get("behaviour"),
            })
            ann_id += 1

    return {
        "info": {
            "description": f"Person-of-interest boxes for {vid}",
            "video_id": vid,
            "video_label": data.get("video_label"),
            "video_comment": data.get("video_comment", ""),
            "context": data.get("context", ""),
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person_of_interest"}],
    }


def write_coco_file(vid: str, data: dict, coco: dict) -> None:
    d = video_dir(vid)
    with _io_lock:
        coco_path(d, vid).write_text(
            json.dumps(coco, indent=2, ensure_ascii=False), encoding="utf-8"
        )


def persist(vid: str, data: dict) -> None:
    """Save the working annotations AND the COCO mirror to the video folder.

    COCO building fills the dimension cache first so it persists in
    annotations.json; a failure to write COCO never blocks the primary save.
    """
    coco = None
    try:
        coco = build_coco(vid, data)  # fills data['frame_dims']
    except Exception as e:  # noqa: BLE001
        print(f"[coco] build failed for {vid}: {e}")
    save_annotations(vid, data)
    if coco is not None:
        try:
            write_coco_file(vid, data, coco)
        except Exception as e:  # noqa: BLE001
            print(f"[coco] write failed for {vid}: {e}")


def ensure_coco(vid: str, data: dict) -> None:
    """Write a COCO file on video open if one doesn't exist yet (backfills
    pre-existing annotations)."""
    d = video_dir(vid)
    if not coco_path(d, vid).is_file():
        persist(vid, data)


def frame_path(vid: str, idx: int) -> Path:
    d = video_dir(vid)
    frames = list_frames(d)
    if idx < 0 or idx >= len(frames):
        raise HTTPException(404, "Frame index out of range")
    return d / frames[idx]


# ----------------------------------------------------------------------------
# YOLO11 detection + single-object forward tracking
# ----------------------------------------------------------------------------

def get_yolo():
    """Load YOLO11 on first use. Returns (model, error_message)."""
    global _yolo_model, _yolo_error
    if _yolo_model is not None or _yolo_error is not None:
        return _yolo_model, _yolo_error
    try:
        from ultralytics import YOLO  # type: ignore
        # yolo11n is small/fast; swap for yolo11m/l/x for higher accuracy.
        _yolo_model = YOLO("yolo11n.pt")
    except Exception as e:  # noqa: BLE001 — surface any import/download failure
        _yolo_error = (
            f"YOLO unavailable: {e}. Install with `pip install ultralytics` "
            "to enable detection and tracking."
        )
    return _yolo_model, _yolo_error


def detect_persons(model, img_path: Path, conf: float = 0.25) -> list[dict]:
    """Run person-class detection on a single frame. Boxes are [x1,y1,x2,y2]."""
    res = model.predict(str(img_path), classes=[0], conf=conf, verbose=False)
    out: list[dict] = []
    if not res:
        return out
    for b in res[0].boxes:
        xyxy = [float(v) for v in b.xyxy[0].tolist()]
        out.append({"box": xyxy, "conf": float(b.conf[0])})
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
    """Return an OpenCV CSRT tracker instance, or None if unavailable.

    CSRT bridges short gaps where YOLO fails to detect the person (e.g. brief
    occlusion). Its location moved across OpenCV versions, so try each spot.
    """
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
    """
    Propagate a single seed box forward.

    For each subsequent frame we run YOLO and pick the person detection that
    best matches the running box (IoU first, centroid fallback). When detection
    is lost we fall back to a CSRT visual tracker for a short grace period, then
    stop so the annotator can re-seed.
    """

    def __init__(self, seed_box, iou_thresh=0.15, max_gap=8):
        self.box = [float(v) for v in seed_box]
        self.iou_thresh = iou_thresh
        self.max_gap = max_gap
        self.gap = 0
        self.csrt = None  # type: ignore
        self._cv2 = None

    def _match(self, dets: list[dict]):
        best, best_iou = None, 0.0
        for d in dets:
            i = iou(self.box, d["box"])
            if i > best_iou:
                best, best_iou = d, i
        if best is not None and best_iou >= self.iou_thresh:
            return best["box"]
        # No overlap — fall back to nearest centroid within a sane distance.
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
            return None  # just initialised on the last known-good frame
        ok, bb = self.csrt.update(frame)
        if not ok:
            return None
        x, y, w, h = bb
        return [float(x), float(y), float(x + w), float(y + h)]

    def step(self, model, img_path: Path):
        """Advance to `img_path`. Returns (box, source) or (None, 'lost')."""
        dets = detect_persons(model, img_path)
        matched = self._match(dets)
        if matched is not None:
            self.box = matched
            self.gap = 0
            self.csrt = None  # reset visual tracker to the fresh detection
            return matched, "detection"
        # Detection lost — try to coast with CSRT.
        est = self._csrt_update(img_path)
        self.gap += 1
        if est is not None:
            self.box = est
            if self.gap > self.max_gap:
                return None, "lost"
            return est, "tracker"
        if self.gap > self.max_gap:
            return None, "lost"
        return self.box, "hold"  # keep last box briefly, hope detection returns


# ----------------------------------------------------------------------------
# API models
# ----------------------------------------------------------------------------

class FrameAnnotation(BaseModel):
    behaviour: str | None = None
    comment: str | None = None
    bbox: list[float] | None = None  # [x1, y1, x2, y2] in image pixels


class VideoMeta(BaseModel):
    video_label: str | None = None
    video_comment: str | None = None
    context: str | None = None


class DetectRequest(BaseModel):
    conf: float = 0.25


class TrackRequest(BaseModel):
    seed_box: list[float]        # [x1,y1,x2,y2] at the start frame
    num_frames: int = 0          # 0 => to end of video
    overwrite: bool = True       # replace existing bboxes on covered frames


# ----------------------------------------------------------------------------
# App + routes
# ----------------------------------------------------------------------------

app = FastAPI(title="Trauma Behaviour Video Annotator")


@app.get("/api/config")
def api_config():
    import importlib.util
    # Truthful availability check without downloading model weights.
    yolo_ready = importlib.util.find_spec("ultralytics") is not None
    return {
        "behaviours": BEHAVIOURS,
        "video_labels": VIDEO_LABELS,
        "yolo_ready": yolo_ready,
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
        labelled = sum(1 for f in frames if data["frames"].get(f, {}).get("behaviour"))
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
    ensure_coco(vid, data)  # backfill a COCO file if none exists yet
    return {
        "video_id": data["video_id"],
        "frames": data["_frame_order"],
        "annotations": data["frames"],
        "video_label": data.get("video_label"),
        "video_comment": data.get("video_comment", ""),
        "context": data.get("context", ""),
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
    slot = data["frames"].setdefault(name, {"behaviour": None, "comment": "", "bbox": None})
    if ann.behaviour is not None:
        if ann.behaviour == "" :
            slot["behaviour"] = None
        elif ann.behaviour in BEHAVIOURS:
            slot["behaviour"] = ann.behaviour
        else:
            raise HTTPException(400, f"Unknown behaviour '{ann.behaviour}'")
    if ann.comment is not None:
        slot["comment"] = ann.comment
    if ann.bbox is not None:
        slot["bbox"] = ann.bbox if len(ann.bbox) == 4 else None
    # bbox explicitly cleared by sending [] ? Treat empty list as clear.
    persist(vid, data)
    return {"ok": True, "frame": name, "annotation": slot}


@app.post("/api/clear-bbox/{vid}/{idx}")
def api_clear_bbox(vid: str, idx: int):
    data = load_annotations(vid)
    frames = data["_frame_order"]
    if idx < 0 or idx >= len(frames):
        raise HTTPException(404, "Frame index out of range")
    data["frames"][frames[idx]]["bbox"] = None
    persist(vid, data)
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
    persist(vid, data)
    return {"ok": True}


@app.post("/api/write-context/{vid}")
def api_write_context(vid: str):
    """Write the current (possibly edited) context back to context.txt on disk."""
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
    model, err = get_yolo()
    if err:
        raise HTTPException(503, err)
    p = frame_path(vid, idx)
    dets = detect_persons(model, p, conf=req.conf)
    return {"detections": dets}


@app.post("/api/track/{vid}/{idx}")
def api_track(vid: str, idx: int, req: TrackRequest):
    """Seed a box at frame `idx` and propagate forward, saving boxes as we go."""
    model, err = get_yolo()
    if err:
        raise HTTPException(503, err)
    if len(req.seed_box) != 4:
        raise HTTPException(400, "seed_box must be [x1,y1,x2,y2]")

    data = load_annotations(vid)
    frames = data["_frame_order"]
    d = video_dir(vid)
    if idx < 0 or idx >= len(frames):
        raise HTTPException(404, "Frame index out of range")

    # Save the seed on the start frame.
    data["frames"][frames[idx]]["bbox"] = [float(v) for v in req.seed_box]

    end = len(frames) if req.num_frames <= 0 else min(len(frames), idx + 1 + req.num_frames)
    tracker = ForwardTracker(req.seed_box)
    results = {frames[idx]: {"box": req.seed_box, "source": "seed"}}

    for i in range(idx + 1, end):
        name = frames[i]
        if not req.overwrite and data["frames"][name].get("bbox"):
            # Respect an existing manual box: sync tracker to it and continue.
            tracker.box = data["frames"][name]["bbox"]
            results[name] = {"box": tracker.box, "source": "kept"}
            continue
        box, source = tracker.step(model, d / name)
        if box is None:
            break
        data["frames"][name]["bbox"] = [float(v) for v in box]
        results[name] = {"box": box, "source": source}

    persist(vid, data)
    return {"ok": True, "results": results, "covered": len(results)}


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
    """Download the COCO detection JSON for one video (built fresh so it always
    matches the current annotations)."""
    data = load_annotations(vid)
    coco = build_coco(vid, data)
    save_annotations(vid, data)  # persist any dims filled while building
    payload = json.dumps(coco, indent=2, ensure_ascii=False)
    return StreamingResponse(
        io.BytesIO(payload.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{vid}{COCO_SUFFIX}"'},
    )


@app.post("/api/write-coco/{vid}")
def api_write_coco(vid: str):
    """Write (or refresh) the on-disk COCO file for one video folder."""
    data = load_annotations(vid)
    persist(vid, data)
    d = video_dir(vid)
    return {"ok": True, "file": coco_path(d, vid).name}


@app.post("/api/write-coco-all")
def api_write_coco_all():
    """Write COCO files for every video folder. Useful after a first run on
    pre-existing annotations."""
    written = []
    if DATA_DIR.is_dir():
        for d in sorted([p for p in DATA_DIR.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name)):
            if not list_frames(d):
                continue
            data = load_annotations(d.name)
            persist(d.name, data)
            written.append(d.name)
    return {"ok": True, "written": written, "count": len(written)}


# Serve the static frontend at the root. Mounted last so /api/* wins.
app.mount("/", StaticFiles(directory="static", html=True), name="static")


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

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
