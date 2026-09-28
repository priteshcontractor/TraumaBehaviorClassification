"""
Person detection, VIP (person-of-interest) selection and single-target tracking.

Shared by the annotator server (app.py) and the batch CLI (trauma_vip_annotation.py).

The VIP is the largest visible person. Score (per tracked person, per frame):
    importance = 0.9 * area + 0.05 * centrality + 0.05 * persistence   (area relative to the largest person)
    score      = 0.5 * previous_score + 0.5 * importance               (light temporal smoothing)
The VIP only switches when the top-scoring person's box is >= 1.15x the current VIP's area for
2 consecutive frames (so two people of nearly equal size don't flicker), when the current VIP has
been missing for longer than `lost_patience` frames, or at a scene cut (identity cannot be carried
across shots, so the new shot's largest person is chosen). While the VIP is briefly hidden, the
largest visible person is returned for that frame without taking over the VIP identity.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = "yolo11s.pt"
WEIGHT_CHOICES = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]


# ----------------------------------------------------------------------------
# Basics
# ----------------------------------------------------------------------------

def resolve_weights(name: str | None) -> str:
    """Prefer weight files shipped next to the app so the working directory does not matter."""
    name = name if name in WEIGHT_CHOICES else DEFAULT_WEIGHTS
    local = APP_DIR / name
    return str(local) if local.is_file() else name


def load_model(name: str | None = None):
    """A fresh YOLO instance. Tracking registers callbacks on the model, so never share
    a tracking model with plain detection."""
    from ultralytics import YOLO  # type: ignore

    return YOLO(resolve_weights(name))


def read_image(path: Path | str) -> np.ndarray | None:
    """cv2.imread cannot open non-ASCII paths on Windows; decode from bytes instead."""
    import cv2  # type: ignore

    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def box_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def centroid_dist(a, b) -> float:
    return math.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)


@dataclass
class Person:
    box: list[float]
    conf: float
    tid: int | None = None
    rotated: bool = False  # found on a 90°-rotated copy of the frame (lying-person pass)


# ----------------------------------------------------------------------------
# Detection / multi-object tracking
# ----------------------------------------------------------------------------

def effective_imgsz(img, imgsz: int) -> int:
    """Upscaling a small frame past its native size blurs it and lowers YOLO scores."""
    native = int(math.ceil(max(img.shape[:2]) / 32) * 32)
    return max(320, min(1920, int(imgsz), native))


def _predict_boxes(model, img, conf: float, imgsz: int, augment: bool) -> list[tuple[list[float], float]]:
    res = model.predict(
        img,
        classes=[0],
        conf=float(conf),
        imgsz=effective_imgsz(img, imgsz),
        augment=bool(augment),
        verbose=False,
    )
    if not res or res[0].boxes is None:
        return []
    return [([float(v) for v in b.xyxy[0].tolist()], float(b.conf[0])) for b in res[0].boxes]


def _unrotate_box(box: list[float], rot: str, w: int, h: int) -> list[float]:
    """Map a box found on a 90°-rotated copy back to the original (w x h) frame."""
    x1, y1, x2, y2 = box
    if rot == "cw":    # original (x, y) -> (h - y, x)
        return [y1, h - x2, y2, h - x1]
    return [w - y2, x1, w - y1, x2]  # ccw: original (x, y) -> (y, w - x)


def _rotated_boxes(model, img, conf: float, imgsz: int, augment: bool) -> list[tuple[list[float], float]]:
    """Horizontal bodies found on copies of the frame rotated 90° both ways."""
    import cv2  # type: ignore

    h, w = img.shape[:2]
    found = []
    for rot, code in (("cw", cv2.ROTATE_90_CLOCKWISE), ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE)):
        for box, c in _predict_boxes(model, cv2.rotate(img, code), conf, imgsz, augment):
            bx = _unrotate_box(box, rot, w, h)
            if (bx[2] - bx[0]) > (bx[3] - bx[1]):
                found.append((bx, c))
    return found


def _nms_people(found: list[tuple[list[float], float]], iou_max: float = 0.5) -> list[Person]:
    found = sorted(found, key=lambda t: t[1], reverse=True)
    out: list[Person] = []
    for box, c in found:
        if all(iou(box, p.box) < iou_max for p in out):
            out.append(Person(box=box, conf=c))
    return out


def _intersection(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _inside(a, b) -> float:
    """Share of box `a` that lies inside box `b`."""
    area = box_area(a)
    return _intersection(a, b) / area if area > 0 else 0.0


def _union_area(boxes) -> float:
    xs = sorted({v for b in boxes for v in (b[0], b[2])})
    ys = sorted({v for b in boxes for v in (b[1], b[3])})
    total = 0.0
    for x1, x2 in zip(xs, xs[1:]):
        for y1, y2 in zip(ys, ys[1:]):
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes):
                total += (x2 - x1) * (y2 - y1)
    return total


def suppress_group_boxes(people: list[Person], frame_w: float | None = None) -> list[Person]:
    """Drop boxes that span several people at once, plus duplicates, keeping one box per person.

    On a 90°-rotated frame two people side by side are stacked vertically and YOLO often returns
    one tall "person" for both, which maps back to a wide box over everyone. Removed:
      - any box holding 2+ smaller detections (each >= 70% inside) that fill >= half of it
        (for upright boxes only when it scores lower than its members: a parent holding children
        is one confident box);
      - rotated boxes wider than 85% of the frame that hold another detection;
      - rotated boxes lying >= 50% inside an upright detection (the upright pass is more reliable);
      - duplicates: IoU >= 0.5, or intersection over the smaller box >= 0.8 when a rotated box is
        involved; the higher score wins.
    """
    keep = []
    for p in people:
        members = [q for q in people if q is not p and box_area(q.box) < box_area(p.box)
                   and _inside(q.box, p.box) >= 0.7]
        if (len(members) >= 2 and _union_area([q.box for q in members]) >= 0.5 * box_area(p.box)
                and (p.rotated or p.conf < max(q.conf for q in members))):
            continue
        if p.rotated:
            if members and frame_w and p.box[2] - p.box[0] > 0.85 * frame_w:
                continue
            if any(not q.rotated and _inside(p.box, q.box) >= 0.5 for q in people):
                continue
        keep.append(p)
    keep.sort(key=lambda p: p.conf, reverse=True)
    out: list[Person] = []
    for p in keep:
        if any(iou(p.box, q.box) >= 0.5 or ((p.rotated or q.rotated) and _intersection(p.box, q.box)
               >= 0.8 * min(box_area(p.box), box_area(q.box))) for q in out):
            continue
        out.append(p)
    return out


def _rotated_people(model, img, conf: float, imgsz: int, augment: bool) -> list[Person]:
    people = _nms_people(_rotated_boxes(model, img, conf, imgsz, augment))
    for p in people:
        p.rotated = True
    return people


def detect_persons(model, img, conf: float = 0.35, imgsz: int = 960, augment: bool = False,
                   lying: bool = False) -> list[Person]:
    """COCO-trained detectors mostly learnt upright people. With `lying`, the frame is also
    run rotated 90° both ways so people lying in bed / on the floor look upright."""
    people = [Person(box=b, conf=c) for b, c in _predict_boxes(model, img, conf, imgsz, augment)]
    if lying:
        people += _rotated_people(model, img, conf, imgsz, augment)
    return suppress_group_boxes(people, img.shape[1])


def detect_lying(model, img, conf: float = 0.3, imgsz: int = 640, min_area: float = 0.03) -> list[Person]:
    """Only the people lying down (rotated passes). `model` must not be a tracking model:
    ByteTrack callbacks would swallow these rotated predictions."""
    h, w = img.shape[:2]
    return [p for p in suppress_group_boxes(_rotated_people(model, img, conf, imgsz, False), w)
            if box_area(p.box) >= min_area * w * h]


def _stash_raw_boxes(predictor) -> None:
    """Runs before Ultralytics' tracking callback, which replaces the detected boxes with
    Kalman-filtered ones that lag behind (and cut off parts of) a moving person."""
    predictor.raw_person_boxes = [
        r.boxes.data.cpu().numpy().copy() if r.boxes is not None else None for r in predictor.results
    ]


def _snap_to_detections(people: list[Person], raw) -> None:
    """Give each track the box of the detection it was matched to in this frame
    (ByteTrack keeps that detection's score, so score + overlap identifies it)."""
    if raw is None or len(raw) == 0:
        return
    for p in people:
        best, best_key = None, (False, 0.5)
        for row in raw:
            box = [float(v) for v in row[:4]]
            ov = iou(p.box, box)
            same_score = abs(float(row[4]) - p.conf) < 1e-4
            if (same_score and ov > 0.1) or ov >= 0.5:
                key = (same_score, ov)
                if key > best_key:
                    best, best_key = box, key
        if best is not None:
            p.box = best


def track_persons(model, img, imgsz: int = 640, tracker: str = "bytetrack.yaml") -> list[Person]:
    """One step of ByteTrack. Call with a fresh model per video so IDs start clean.
    IDs come from the tracker, boxes from this frame's detections."""
    cbs = model.callbacks.setdefault("on_predict_postprocess_end", [])
    if _stash_raw_boxes not in cbs:
        cbs.insert(0, _stash_raw_boxes)
    res = model.track(
        img,
        persist=True,
        classes=[0],
        conf=0.1,  # ByteTrack needs low-score boxes for its second association pass
        imgsz=effective_imgsz(img, imgsz),
        tracker=tracker,
        verbose=False,
    )
    if not res or res[0].boxes is None or res[0].boxes.id is None:
        return []
    r = res[0].boxes
    boxes = r.xyxy.cpu().numpy()
    ids = r.id.cpu().numpy().astype(int)
    confs = r.conf.cpu().numpy()
    people = [
        Person(box=[float(v) for v in bx], conf=float(c), tid=int(t))
        for bx, t, c in zip(boxes, ids, confs)
    ]
    raw = getattr(model.predictor, "raw_person_boxes", None)
    _snap_to_detections(people, raw[0] if raw else None)
    return people


# ----------------------------------------------------------------------------
# VIP selection
# ----------------------------------------------------------------------------

class VipSelector:
    def __init__(
        self,
        width: int,
        height: int,
        area_w: float = 0.9,
        center_w: float = 0.05,
        age_w: float = 0.05,
        smoothing: float = 0.5,
        hysteresis: float = 1.15,
        switch_frames: int = 2,
        age_frames: int = 300,
        lost_patience: int = 45,
        min_conf: float = 0.25,
    ):
        self.w, self.h = max(1, width), max(1, height)
        self.area_w, self.center_w, self.age_w = area_w, center_w, age_w
        self.smoothing = smoothing
        self.hysteresis = hysteresis  # box-area ratio a challenger needs over the current VIP
        self.switch_frames = switch_frames
        self.pending = 0
        self.age_frames = age_frames
        self.lost_patience = lost_patience
        self.min_conf = min_conf
        self.scores: dict[int, float] = defaultdict(float)
        self.age: dict[int, int] = defaultdict(int)
        self.last_seen: dict[int, int] = {}
        self.last_box: dict[int, list[float]] = {}
        self.alias: dict[int, int] = {}
        self.vip: int | None = None
        self.frame = 0

    def canonical(self, tid: int) -> int:
        while tid in self.alias:
            tid = self.alias[tid]
        return tid

    def update(self, people: list[Person], scene_cut: bool = False) -> int | None:
        """Feed one frame of tracked people. Returns the canonical VIP id if visible.
        After a scene cut the largest person of the new shot becomes the VIP."""
        self.frame += 1
        if scene_cut:
            self.vip = None
            self.pending = 0
        visible: dict[int, Person] = {}
        for p in people:
            if p.tid is None:
                continue
            cid = self.canonical(p.tid)
            # A tracker ID can only be "first seen" once; low-confidence boxes of an
            # existing track still count, but they never start a VIP candidacy.
            if cid not in self.age and p.conf < self.min_conf:
                continue
            visible[cid] = p

        # ByteTrack drops IDs after long occlusions; if a brand-new ID appears where the
        # missing VIP was last seen, treat it as the same person.
        if self.vip is not None and self.vip not in visible:
            vbox = self.last_box.get(self.vip)
            if vbox is not None:
                for cid, p in list(visible.items()):
                    if cid in self.age:
                        continue
                    if iou(vbox, p.box) >= 0.3:
                        self.alias[cid] = self.vip
                        visible[self.vip] = visible.pop(cid)
                        break

        if visible:
            max_area = max(box_area(p.box) for p in visible.values()) or 1.0
            half_diag = math.hypot(self.w, self.h) / 2
            for cid, p in visible.items():
                self.age[cid] += 1
                x1, y1, x2, y2 = p.box
                area = box_area(p.box) / max_area
                dist = math.hypot((x1 + x2) / 2 - self.w / 2, (y1 + y2) / 2 - self.h / 2)
                center = max(0.0, 1.0 - dist / half_diag)
                persistence = min(self.age[cid] / self.age_frames, 1.0)
                importance = self.area_w * area + self.center_w * center + self.age_w * persistence
                self.scores[cid] = self.smoothing * self.scores[cid] + (1 - self.smoothing) * importance
                self.last_seen[cid] = self.frame
                self.last_box[cid] = list(p.box)

            best = max(visible, key=lambda c: self.scores[c])
            if self.vip is None or best == self.vip:
                self.vip = best
                self.pending = 0
            elif self.vip in visible:
                larger = box_area(visible[best].box) >= self.hysteresis * box_area(visible[self.vip].box)
                self.pending = self.pending + 1 if larger else 0
                if self.pending >= self.switch_frames:
                    self.vip = best
                    self.pending = 0
            elif self.frame - self.last_seen.get(self.vip, -10**9) > self.lost_patience:
                self.vip = best
                self.pending = 0
            if self.vip not in visible:
                # VIP briefly hidden: the largest visible person stands in without taking over the VIP identity
                return best

        return self.vip if self.vip in visible else None

    def visible_people(self, people: list[Person]) -> list[tuple[int, Person]]:
        return [(self.canonical(p.tid), p) for p in people if p.tid is not None]


# ----------------------------------------------------------------------------
# People lying down (rotated-pass detections, which ByteTrack never sees)
# ----------------------------------------------------------------------------

LYING_TID_BASE = 1_000_000


def is_lying_tid(tid: int | None) -> bool:
    return tid is not None and tid >= LYING_TID_BASE


class LyingTracker:
    """Gives rotated-pass boxes stable IDs (>= LYING_TID_BASE) with greedy IoU / centroid matching."""

    def __init__(self, max_age: int = 30, iou_thresh: float = 0.3):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.tracks: dict[int, tuple[list[float], int]] = {}
        self.next_id = LYING_TID_BASE
        self.frame = 0

    def reset(self) -> None:
        self.tracks.clear()

    def _match_score(self, a, b) -> float:
        ov = iou(a, b)
        if ov >= self.iou_thresh:
            return 1.0 + ov
        aw, ah = a[2] - a[0], a[3] - a[1]
        ratio = box_area(b) / max(1.0, box_area(a))
        if centroid_dist(a, b) < 0.4 * max(aw, ah) and 0.5 <= ratio <= 2.0:
            return 0.5
        return 0.0

    def update(self, people: list[Person]) -> list[Person]:
        self.frame += 1
        self.tracks = {t: v for t, v in self.tracks.items() if self.frame - v[1] <= self.max_age}
        pairs = sorted(
            ((self._match_score(box, p.box), t, k) for t, (box, _) in self.tracks.items()
             for k, p in enumerate(people)),
            reverse=True,
        )
        used_t, used_p = set(), set()
        for score, t, k in pairs:
            if score <= 0 or t in used_t or k in used_p:
                continue
            used_t.add(t)
            used_p.add(k)
            people[k].tid = t
        for k, p in enumerate(people):
            if k not in used_p:
                p.tid = self.next_id
                self.next_id += 1
        for p in people:
            self.tracks[p.tid] = (list(p.box), self.frame)
        return people


def merge_lying(tracked: list[Person], lying: list[Person], iou_max: float = 0.3) -> list[Person]:
    """Rotated-pass boxes that ByteTrack did not already cover (and that are not group boxes)."""
    for p in lying:
        p.rotated = True
    kept = suppress_group_boxes(tracked + lying)
    return [p for p in lying if any(p is k for k in kept)
            and all(iou(p.box, q.box) < iou_max for q in tracked)]


# ----------------------------------------------------------------------------
# Following one chosen person through already-tracked frames
# ----------------------------------------------------------------------------

def reassociate(box, people: list[Person], iou_thresh: float = 0.3) -> Person | None:
    """The person most likely to be the one last seen at `box`: best overlap, else a close,
    similarly sized box (fast motion lowers IoU)."""
    best, best_score = None, 0.0
    for p in people:
        ov = iou(box, p.box)
        if ov >= iou_thresh and ov > best_score:
            best, best_score = p, ov
    if best is not None:
        return best
    bw, bh = box[2] - box[0], box[3] - box[1]
    limit = 0.5 * max(bw, bh)
    for p in people:
        ratio = box_area(p.box) / max(1.0, bw * bh)
        d = centroid_dist(box, p.box)
        if d < limit and 0.6 <= ratio <= 1.6:
            limit, best = d, p
    return best


def match_seed(people: list[Person], seed_box, min_iou: float = 0.3) -> Person | None:
    best = max(people, key=lambda p: iou(seed_box, p.box), default=None)
    return best if best is not None and iou(seed_box, best.box) >= min_iou else None


def follow_identity(frames: list[list[Person]], shot_of: list[int], seed_idx: int, seed: Person,
                    max_gap: int = 15, iou_thresh: float = 0.3) -> tuple[dict[int, Person], dict[str, int | str]]:
    """Follow the tracked person `seed` (on frame `seed_idx`) forwards and backwards within its shot.

    The tracker ID carries the person; when it is dropped, a box where they were last seen takes over,
    but never one belonging to somebody who was visible at the same time as them. Following stops at a
    scene cut or after `max_gap` frames without them. Returns {frame index: their box} and the span."""
    found = {seed_idx: seed}
    span: dict[str, int | str] = {}
    for step in (1, -1):
        tid, box, misses = seed.tid, list(seed.box), 0
        others = {p.tid for p in frames[seed_idx] if p is not seed}
        reason = "end"
        i = seed_idx + step
        while 0 <= i < len(frames):
            if shot_of[i] != shot_of[seed_idx]:
                reason = "scene_cut"
                break
            people = frames[i]
            match = next((p for p in people if p.tid == tid), None)
            if match is not None and misses == 0 and iou(box, match.box) < 0.05:
                match = None  # ID switched to someone far away
            if match is None:
                match = reassociate(box, [p for p in people if p.tid not in others], iou_thresh)
            if match is None:
                misses += 1
                if misses > max_gap:
                    reason = "lost"
                    break
            else:
                tid, box, misses = match.tid, list(match.box), 0
                found[i] = match
                others.update(p.tid for p in people if p is not match)
            i += step
        if step == 1:
            span["last"], span["stop_reason"] = max(found), reason
        else:
            span["first"], span["start_reason"] = min(found), reason
    return found, span


# ----------------------------------------------------------------------------
# Single-target tracking from a seed box
# ----------------------------------------------------------------------------

def make_visual_tracker():
    """Best available OpenCV single-object tracker (CSRT/KCF need opencv-contrib; MIL ships in core)."""
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        for ns in (cv2, getattr(cv2, "legacy", None)):
            factory = getattr(ns, name, None) if ns is not None else None
            if callable(factory):
                try:
                    return factory()
                except Exception:
                    continue
    return None


def frame_signature(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2  # type: ignore

    small = cv2.resize(img, (64, 36), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return gray, cv2.normalize(hist, hist).flatten()


def is_scene_cut(prev_sig, sig) -> bool:
    """Hard cut between consecutive frames: a big pixel change or a very different colour histogram.
    Ordinary motion changes 64x36 thumbnails by ~2-6 grey levels on average; cuts by 20+."""
    import cv2  # type: ignore

    if prev_sig is None or sig is None:
        return False
    diff = float(np.abs(prev_sig[0] - sig[0]).mean())
    corr = cv2.compareHist(prev_sig[1], sig[1], cv2.HISTCMP_CORREL)
    return diff > 22 or corr < 0.6


def reset_tracker(model) -> None:
    for t in getattr(getattr(model, "predictor", None), "trackers", None) or []:
        try:
            t.reset()
        except Exception:
            pass


def _clip_box(b, w, h) -> list[float]:
    x1 = min(max(0.0, b[0]), w - 1)
    y1 = min(max(0.0, b[1]), h - 1)
    x2 = min(max(x1 + 1, b[2]), w)
    y2 = min(max(y1 + 1, b[3]), h)
    return [float(x1), float(y1), float(x2), float(y2)]


class SingleTargetTracker:
    """Follow one person from a seed box.

    ByteTrack IDs keep the person across frames; when the ID is dropped we re-associate by
    overlap with the last known box, and when no detection matches (e.g. someone lying under
    a blanket) an OpenCV visual tracker bridges the gap for up to `max_gap` frames.
    """

    def __init__(
        self,
        model,
        seed_img: np.ndarray,
        seed_box,
        imgsz: int = 640,
        iou_thresh: float = 0.3,
        max_gap: int = 15,
        lying_model=None,
    ):
        self.model = model
        self.imgsz = imgsz
        self.iou_thresh = iou_thresh
        self.max_gap = max_gap
        self.h, self.w = seed_img.shape[:2]
        self.box = _clip_box([float(v) for v in seed_box], self.w, self.h)
        self.prev_img = seed_img
        self.prev_sig = frame_signature(seed_img)
        self.visual = None
        self.misses = 0

        people = track_persons(model, seed_img, imgsz=imgsz)
        best, best_iou = None, 0.0
        for p in people:
            i = iou(self.box, p.box)
            if i > best_iou:
                best, best_iou = p, i
        self.tid = best.tid if best is not None and best_iou >= 0.3 else None
        # A seed nobody detects (manual box on an undetectable person) relies on the visual tracker.
        self.detectable = self.tid is not None
        # Seed on a person lying down: the rotated passes can re-find them (3x cost, so only then).
        self.lying_model = None
        if not self.detectable and lying_model is not None:
            if any(iou(self.box, p.box) >= 0.3 for p in detect_lying(lying_model, seed_img, imgsz=imgsz)):
                self.lying_model = lying_model

    def _reassociate(self, people: list[Person]) -> Person | None:
        return reassociate(self.box, people, self.iou_thresh)

    def _visual_step(self, img: np.ndarray) -> list[float] | None:
        if self.visual is None:
            self.visual = make_visual_tracker()
            if self.visual is None:
                return None
            x1, y1, x2, y2 = self.box
            try:
                self.visual.init(self.prev_img, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            except Exception:
                self.visual = None
                return None
        try:
            ok, bb = self.visual.update(img)
        except Exception:
            return None
        if not ok:
            return None
        x, y, w, h = bb
        return _clip_box([x, y, x + w, y + h], self.w, self.h)

    def step(self, img: np.ndarray) -> tuple[list[float] | None, str]:
        sig = frame_signature(img)
        cut = is_scene_cut(self.prev_sig, sig)
        self.prev_sig = sig
        if cut:
            # A new shot: box overlap and appearance say nothing about identity across a cut
            return None, "scene_cut"
        people = track_persons(self.model, img, imgsz=self.imgsz)
        match = None
        if self.tid is not None:
            match = next((p for p in people if p.tid == self.tid), None)
            if match is not None and iou(self.box, match.box) < 0.05 and self.misses == 0:
                match = None  # ID switched to someone far away
        if match is None:
            match = self._reassociate(people)
            if match is not None and not self.detectable and iou(self.box, match.box) < 0.5:
                match = None  # stay on the visual tracker until a detection clearly overlaps

        if match is None and self.lying_model is not None:
            lying = merge_lying(people, detect_lying(self.lying_model, img, imgsz=self.imgsz))
            lm = self._reassociate(lying)
            if lm is not None:
                self.box = _clip_box(lm.box, self.w, self.h)
                self.misses = 0
                self.visual = None
                self.prev_img = img
                return self.box, "lying"

        if match is not None:
            self.tid = match.tid
            self.detectable = True
            self.box = _clip_box(match.box, self.w, self.h)
            self.misses = 0
            self.visual = None
            self.prev_img = img
            return self.box, "tracker"

        self.misses += 1
        est = self._visual_step(img)
        self.prev_img = img
        if est is not None and (not self.detectable or self.misses <= self.max_gap):
            if not self.detectable:
                self.misses = 0
            self.box = est
            return self.box, "visual"
        if self.misses <= self.max_gap:
            return self.box, "hold"
        return None, "lost"
