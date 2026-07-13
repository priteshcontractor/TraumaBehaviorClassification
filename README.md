# Trauma Behaviour Video Annotator

A local, browser-based tool for annotating videos that are stored as **folders of
ordered frames**. Built with FastAPI (Python) + a vanilla-JS frontend. Runs on
your machine, no cloud.

Per **frame** you set a behaviour label and (optionally) a bounding box for the
person of interest. Per **video** you set a Trauma / No-Trauma label, a comment,
and a contextual scenario. YOLO11 can detect people and propagate a single
selected box forward across frames.

---

## 1. Data layout

One folder per video inside your data directory. Each folder holds ordered image
frames and an optional `context.txt`:

```
data/
├── video_001/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   ├── ...
│   └── context.txt          # background scenario (also: scenario.txt / background.txt)
├── video_002/
│   └── ...
```

Frames are sorted **naturally** (`frame_2` before `frame_10`). Supported image
types: jpg, jpeg, png, bmp, webp, tif. Annotations are written to
`annotations.json` inside each video folder, so your source frames are never
modified.

To split an `.mp4` into a frame folder:
```
ffmpeg -i video_001.mp4 data/video_001/frame_%04d.jpg
```

## 2. Install

```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The core app needs only **fastapi, uvicorn, pillow**. Detection and tracking need
**ultralytics + opencv-python** (already listed in requirements.txt; they pull in
PyTorch). If you skip them, everything works except the *Detect* and *Track
forward* buttons.

## 3. Run

```
python app.py --data ./data
```
Then open <http://127.0.0.1:8000/>. Options: `--host`, `--port`, `--data`.

A synthetic sample is included so you can try it immediately. Regenerate with:
```
python tools/make_sample.py --frames 40
```

## 4. Annotation workflow

1. Pick a video in the left sidebar.
2. Step through frames (arrow keys / scrubber / play).
3. For each frame set a **behaviour** (keys `1`–`5`).
4. Mark the **person of interest** box:
   - **Draw** it with the mouse, **or**
   - press **Detect** (`D`) to run YOLO11, then **click the correct person**.
     Other detections are discarded automatically.
   - press **Track forward** (`T`) to propagate that single box across the
     following frames. It follows only the one person; drawing/re-detecting on a
     later frame re-seeds the track from there.
5. Set the **video label** (Trauma / No-Trauma), add comments, and review the
   **scenario** — it is highlighted when you reach the last frame.
6. Everything autosaves. Use **Export** for `annotations.json` or a COCO
   bounding-box file.

### Keyboard shortcuts
| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `←` `→` | prev / next frame | `1`–`5` | set behaviour |
| `Home` `End` | first / last frame | `D` | detect people |
| `Space` | play / pause | `T` | track box forward |
| `Del` | clear box | | |

## 5. How tracking works

`Track forward` seeds a single-object tracker with your box, then for each later
frame runs YOLO11 and keeps the detection that best matches the running box (IoU,
with a centroid fallback). If detection is briefly lost it coasts with an OpenCV
CSRT visual tracker for a few frames, then stops so you can re-seed. Existing
boxes are respected if you set `overwrite=false` in the request.

Swap the model for higher accuracy in `app.py` (`get_yolo`): `yolo11n.pt` →
`yolo11m.pt` / `yolo11l.pt` / `yolo11x.pt`. Weights download automatically on
first use.

## 6. Output format

Two files are written into each video folder and kept in sync automatically on
every edit — no manual export step needed:

- **`annotations.json`** — the working record.
  ```json
  {
    "video_id": "video_001",
    "video_label": "trauma",
    "video_comment": "...",
    "context": "...",
    "frames": {
      "frame_0001.jpg": { "behaviour": "flashback", "comment": "", "bbox": [x1,y1,x2,y2] }
    }
  }
  ```
  Behaviours: `flashback`, `avoidance`, `negative_emotion`, `hyper_arousal`,
  `normal`. Boxes are `[x1,y1,x2,y2]` in image pixels.

- **`<video>_coco.json`** — standard COCO detection format, ready for training.
  Boxes become `xywh` under a single `person_of_interest` category. Behaviour and
  per-frame comment ride on each `image`; video label, comment and context sit in
  `info`.
  ```json
  {
    "info": { "video_id": "video_001", "video_label": "trauma", "context": "..." },
    "images": [ { "id": 1, "file_name": "frame_0001.jpg", "width": 640, "height": 360,
                  "behaviour": "flashback", "comment": "" } ],
    "annotations": [ { "id": 1, "image_id": 1, "category_id": 1,
                       "bbox": [x, y, w, h], "area": 0, "iscrowd": 0,
                       "behaviour": "flashback" } ],
    "categories": [ { "id": 1, "name": "person_of_interest" } ]
  }
  ```

The COCO file is (re)generated when a video is first opened, so pre-existing
annotation folders are backfilled the moment you view them. **Rebuild all COCO
files** (Export panel) refreshes every folder at once. The Download buttons still
give you a copy through the browser. Frame dimensions are read once and cached in
`annotations.json`, so re-saving large videos stays fast.

## 7. Assumptions & notes

- **One behaviour per frame** (single-select). To allow multiple simultaneous
  labels, change `behaviour` to a list in the backend and the buttons to toggles.
- **Comments** exist at both frame and video level.
- **Context** is stored in `annotations.json`; *Write to context.txt* pushes the
  edited version back to the sidecar file if you want the source updated.
- Single user, no authentication — intended for local use.
- The synthetic sample figure is not a real person, so YOLO won't detect it; use
  real footage to exercise Detect/Track.

## 8. Files
```
app.py                 FastAPI backend + YOLO detect/track
static/index.html      UI
static/style.css       styling
static/app.js          canvas editing, navigation, autosave, detect/track
tools/make_sample.py   synthetic sample generator
requirements.txt
