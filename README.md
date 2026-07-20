# Trauma Behaviour Video Annotator

Local browser tool for labelling trauma-related behaviours on video frames.
Built with **FastAPI** (Python) + a vanilla JS frontend. Runs on your machine — no cloud.

---

## What you can do

| Feature | Description |
|--------|-------------|
| **Load video** | Upload MP4/MOV/AVI/MKV; frames are extracted in the app (ffmpeg or OpenCV) |
| **Load frames** | Upload images or a ZIP of frames |
| **Import JSON** | Merge working `annotations.json` or COCO onto the open video |
| **Conflict handling** | If a video id already exists: Open / Re-extract keep / Wipe (with backup) / New folder |
| **Detect people** | YOLO person detection (≥ 35% confidence). Object list appears under the frame |
| **Object list** | Confirm (OK), edit label, set behaviours per person, Delete one row, Clear all |
| **POI (red)** | Exactly one Person of Interest — always drawn in **red**; other IDs use fixed colors |
| **Multi-select behaviours** | Flashback, Avoidance, Negative emotion, Hyperarousal, Normal (per object + per frame) |
| **Trauma / No Trauma** | Video-level label + comments + scenario/context |
| **Track forward** | Propagate POI box and/or labels across frames (with options + undo) |
| **Export** | `annotations.json` and COCO (`*_coco.json`) |

---

## Install

```bash
py -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
py -m pip install -r requirements.txt
```

Optional: put [ffmpeg](https://ffmpeg.org/) on your PATH for faster video extraction.
Otherwise OpenCV (in `requirements.txt`) is used.

---

## Run

```bash
py app.py --data ./data
```

Open **http://127.0.0.1:8000/**

Options: `--host`, `--port`, `--data`.

If port 8000 is busy:

```bash
py app.py --port 8001
```

After code updates, **restart the server** and hard-refresh the browser (`Ctrl+F5`).

---

## How to use (updated workflow)

### 1. Import media
1. Click **Load video…** (or **Load frames…**).
2. Optionally set **Extract FPS** (blank = all frames).
3. If that video id already exists, choose:
   - **Open existing** (safest)
   - **Re-extract, keep annotations**
   - **Re-extract & wipe** (type the video id to confirm; a backup is kept)
   - **Save as new folder**
4. A wait popup appears while frames are extracted.

### 2. Navigate frames
- Scrubber, ◀ ▶, **Space** (play/pause), **Home** / **End**
- Zoom: **+** / **−** / **Fit**, or `Ctrl` + mouse wheel
- Collapse the left sidebar with **«** for a larger frame

### 3. Detect people
1. Adjust **Min conf** (floor is **35%** — lower scores are ignored).
2. Prefer **Size 960** and **nano** model for speed; enable **Augment** only if lying/bed poses are missed.
3. Click **Detect** (`D`). Wait popup shows while YOLO runs.
4. The **object list** appears under the frame. A **smart POI** is chosen automatically (highest confidence + larger + more central person) and drawn in **red**.
5. Change the POI anytime: click **POI** on another row, or set label to *Person of interest*. Only one POI is allowed.

### 4. Object list (under the frame)
For each detected person:

| Column | Action |
|--------|--------|
| **OK** | Confirm this detection |
| **ID** | Color-coded id (1 green, 2 blue, 3 amber…) |
| **Role** | Shows **POI** if this is the person of interest |
| **Label** | Person of interest / Person / Other |
| **Behaviours** | Tick Flash / Avoid / Neg. / Hyper / Normal **for that person** |
| **POI** | Make this the only red Person of Interest |
| **Delete** | Remove **this** object only (others stay) |

Also:
- **Confirm all** — confirm every detection; smart POI if none set
- **Apply confirmed** — write the confirmed object list onto this frame, or onto the next **N** frames. **The same person keeps the same ID** across those frames.
- **Existing boxes** (dropdown):
  - **Replace them (default):** remove previous boxes on those frames, then write the confirmed list
  - **Keep them & add new:** leave existing boxes; add the confirmed ones (near-duplicates / same ID skipped)
- **Clear all objects** — wipe objects on this frame (asks to confirm)

**Track** also keeps one stable ID for the POI across all tracked frames.

**Smart POI (automatic)**  
After Detect, one person is marked POI (red) using a score from:
1. Detection confidence (~45%)
2. Bounding-box size (~35%) — larger person preferred
3. How central they are in the frame (~20%)

If you already had a POI, it is kept. Change anytime with the **POI** button (only one POI allowed).

**Why some people are missed**  
YOLO only returns the COCO *person* class, and scores below **35%** are dropped. Lying down, under blankets, far away, blurred, or heavily occluded people often score low or are missed — especially with the fast nano model. Raise Size / use small–medium model / turn Augment on, or draw the box manually.

### 5. Frame & video labels
- **Frame behaviour** chips (keys `1`–`5`) — whole-frame labels
- **Trauma / No Trauma** — video-level
- Comments (video + frame) and **Context / scenario**

### 6. Track forward
1. Ensure a red **POI** box exists.
2. Click **Track…** (`T`) and choose:
   - Propagate box / behaviours / comment
   - Range (to end or next N frames)
   - Overwrite options, stop-if-lost, IoU
3. **Preview** or **Track**. Wait popup appears for long runs.
4. **Undo** restores the previous snapshot of affected frames.
5. `Shift+T` repeats the last track settings.

### 7. Export
- Download **JSON** or **COCO** from the Export panel.
- **Rebuild all COCO** refreshes COCO files for every video folder.
- Working file is always `data/<video>/annotations.json`.
- COCO is also written on track/apply/export (lightweight edits do not rebuild COCO every click, for speed).

---

## Keyboard shortcuts

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| ← → | Previous / next frame | `1`–`5` | Toggle frame behaviours |
| Home / End | First / last frame | `D` | Detect people |
| Space | Play / pause | `T` | Track options |
| Del | Delete selected object | `Shift+T` | Track with last options |
| `+` / `-` / `0` | Zoom in / out / fit | `A` | Apply confirmed objects |

---

## Colors

| Role | Color |
|------|--------|
| **POI** (person of interest) | Always **red** |
| Object ID 1 | Green |
| Object ID 2 | Blue |
| Object ID 3 | Amber |
| ID 4+ | Purple, cyan, pink, … |

Pending (unconfirmed) detections are drawn with a **dashed** outline.

---

## Output format

Per video folder under `--data`:

```
data/
└── video_001/
    ├── frame_000001.jpg
    ├── …
    ├── annotations.json
    ├── video_001_coco.json
    └── annotations.backup.<timestamp>.json   # after wipe / re-extract
```

### `annotations.json` (working record)

```json
{
  "video_id": "video_001",
  "video_label": "trauma",
  "video_comment": "",
  "context": "",
  "frames": {
    "frame_000001.jpg": {
      "behaviours": ["flashback"],
      "comment": "",
      "bbox": [120, 40, 380, 460],
      "objects": [
        {
          "id": "obj_…",
          "bbox": [120, 40, 380, 460],
          "label": "person_of_interest",
          "behaviours": ["flashback", "hyper_arousal"],
          "confirmed": true,
          "is_poi": true,
          "conf": 0.81,
          "source": "detection"
        }
      ]
    }
  }
}
```

### COCO

- Category includes `person_of_interest`, `person`, `other`
- Boxes are COCO `xywh` in pixels
- Video label / context live in `info`; per-frame and per-object behaviours ride on images/annotations

---

## Tips for better detection (e.g. person on a bed)

YOLO often misses **lying / covered** people. Try:

1. Larger **Size** (1280+) and **small/medium** model  
2. Turn **Augment** on  
3. Or **draw** the POI box manually, then **Track**

Detections below **35%** confidence are always discarded.

---

## Project files

```
app.py              FastAPI backend (import, detect, track, annotate, export)
static/index.html   UI
static/style.css    Layout / theme
static/app.js       Canvas, object table, autosave, wait overlay
requirements.txt
README.md
```
