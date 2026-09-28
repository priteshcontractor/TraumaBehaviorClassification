# Trauma Behaviour Video Annotator

Local browser tool for labelling trauma-related behaviours and the person of interest (VIP / POI)
in videos. FastAPI backend + vanilla JS frontend, YOLO11 + ByteTrack for people. Runs fully on your
machine.

---

## Install (once)

```bash
py -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
py -m pip install -r requirements.txt
```

Optional: put [ffmpeg](https://ffmpeg.org/) on your PATH for faster frame extraction (OpenCV is used
otherwise). The default YOLO model is *small* (`yolo11s.pt`); *nano* (`yolo11n.pt`, fastest) and
*medium* (`yolo11m.pt`) can be picked in *Detection settings*. Weight files next to `app.py` are
used directly, missing ones download on first use.

---

## Run the annotator

```bash
py app.py --data ./data
```

Open **http://127.0.0.1:8000/** (options: `--port 8001`, `--host`, `--data <folder>`).
After updating the code, restart the server and hard-refresh the browser (`Ctrl+F5`).

### Workflow

1. **Open folder…** (left sidebar) — type or *Browse…* to a folder such as `D:\Trauma_Dataset_Input`.
   Every video in it **and all sub-folders** is listed immediately in the sidebar (grouped by
   folder, natural order) and extracted **one after another** in the background.
   - *Extract FPS*: blank = every frame; `5` is a good default for long recordings.
   - *Pre-fill Trauma / No trauma from folder names*: a path containing `no trauma`, `non trauma`
     or `without trauma` → **No trauma**, any other path containing `trauma` → **Trauma**.
     `P001` / `S001` in the path pre-fill Participant / Session ID.
   - *Detect + track people after extraction*: runs Detect + track (whole video, largest person =
     POI) on each video automatically.
   - Already-imported videos are skipped unless *Re-extract* is ticked.
2. Click a video in the sidebar (or `PgUp` / `PgDn`). The first ready video opens automatically.
3. **Video label** — Trauma / No trauma, participant, session, comments, context.
4. **Person of interest** (red box, exactly one per frame):
   - **Detect + track all frames** (`Shift+D`) — one click: detects every person in every frame
     (upright, plus people lying down when *Find people lying down* is on), tracks them with stable
     IDs (ByteTrack; fresh IDs after a scene cut) and marks the POI. Runs as a background job with
     progress and Cancel; nothing changes until it finishes, and **Undo** reverts it. Behaviours,
     hand-drawn boxes and POIs you chose yourself are always kept; earlier automatic boxes in the
     range are replaced. It uses the model, lying-down option and minimum confidence (for
     lying-down boxes) from *Detection settings*; tracking runs at 640 px.
     The line under the button shows the current options; **⚙** changes them (remembered):
     - *Frames*: **Whole video** (default) or **From this frame to the end**.
     - *Person of interest*: **Largest visible person** (default) — switches to someone else only
       when that person is clearly larger (≥ 1.15× the box area for 2 frames) or the POI is gone,
       and re-selects at scene cuts; while the POI is briefly hidden the largest visible person
       stands in. Or **Follow the current red POI** — the red box on the current frame is matched to
       a tracked person, who stays the POI before and after that frame for the whole shot (tracker
       ID, then re-matching by position, never switching to someone who was visible at the same
       time). Following stops at a scene cut or when the person is gone for 15 frames; from there
       the POI stays as it was (none on a fresh video), so nobody else is picked silently. The app
       jumps to that frame: double-click the person and run again with *From this frame*.
   - **Detect frame** (`D`) — people on this frame; the largest one becomes the POI, unless you
     chose the POI yourself.
   - **Draw** a box by dragging on the image (the first box becomes the POI); drag to move, drag
     corners to resize, **double-click** a box (or *Make POI* / `P`) to make it the POI.
     A POI you pick, draw or move is locked: Detect and Detect + track won't replace it.
   - **Track forward** (`T`) — follows the red box to the end / next N frames. Uses ByteTrack IDs,
     re-matches after misses and falls back to a visual tracker for people YOLO can't see
     (e.g. lying under a blanket). Stops at scene cuts or when the person is lost.
   - **Undo** (`Ctrl+Z`) reverts the last Detect + track / Track / Fill.
5. **Behaviour** — toggle `1`–`5` (Flashback, Avoidance, Negative emotion, Hyperarousal, Normal)
   on the frame, then **Fill until next label** (`F`) or *Next N → Apply* to spread it.
   The timeline under the image shows labelled ranges in colour.
6. **Export** — everything autosaves. *Export all videos* writes the dataset files below.

Background jobs (extraction, Detect + track, tracking) show in the sidebar with progress and a ✕ to cancel.

### Keyboard shortcuts (`?` in the app)

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| ← → (`Shift` ×10) | Previous / next frame | `1`–`5` | Toggle behaviour |
| Home / End | First / last frame | `F` | Fill until next label |
| Space | Play / pause | `D` | Detect people |
| PgUp / PgDn | Previous / next video | `Shift+D` | Detect + track all frames |
| | | `T` / `Shift+T` | Track / repeat last settings |
| `P` | Selected box → POI | Del | Delete selected box |
| `Ctrl+Z` | Undo | `+` `-` `0` | Zoom in / out / fit |

---

## Batch VIP script (no browser)

`trauma_vip_annotation.py` uses the same engine as the app and writes into the same data folder,
so you can pre-process a whole dataset and then review it in the annotator:

```bash
py trauma_vip_annotation.py --input "D:\Trauma_Dataset_Input" --data ./data --fps 5
```

| Option | Meaning |
|--------|---------|
| `--fps 5` | Extraction rate (default: every frame) |
| `--model yolo11n.pt` | YOLO weights: `yolo11n.pt` (fastest), `yolo11s.pt` (default), `yolo11m.pt` (most accurate) |
| `--imgsz 960` | Detection size (default 640) |
| `--preview` | Also render `<video>_vip_preview.mp4` with the VIP in red |
| `--skip-vip` | Only extract + pre-fill labels |
| `--reextract` | Re-extract videos already in the data folder |
| `--no-folder-labels` | Don't guess Trauma / No trauma from folder names |

---

## Output

```
data/
├── <video_id>/                      parent folder + file name, e.g. P001_S001_clip_a
│   ├── frame_000001.jpg …
│   ├── meta.json                    source path, fps, size, group
│   ├── annotations.json             working record (autosaved)
│   └── <video_id>_coco.json         COCO, kept up to date automatically
└── _exports/                        written by "Export all videos"
    ├── all_annotations.json         every video: id, filename, labels, per-frame bbox + behaviours
    ├── metadata.csv / metadata.xlsx one row per behaviour segment
    └── videos/<video_id>_annotations.json + _coco.json
```

`metadata.csv` columns:
`# | participant_id | session_id | filename | modality | start_time | end_time | behavioral | trauma_label | video_id | start_frame | end_frame`

Per-frame record in `all_annotations.json`:

```json
"frame_000042.jpg": {
  "index": 41, "time": 8.2,
  "behaviours": ["flashback"],
  "comment": "",
  "bbox": [120, 40, 380, 460],
  "objects": [
    {"id": "track_3", "bbox": [120, 40, 380, 460], "label": "person_of_interest",
     "is_poi": true, "poi_locked": false, "conf": 0.81, "source": "auto_vip", "behaviours": ["flashback"]}
  ]
}
```

`bbox` is always the POI box in pixels `[x1, y1, x2, y2]`; COCO files use `xywh`.

---

## Tips

- People lying in bed / on the floor: *Detection settings* → *Find people lying down* (on by default)
  also runs the detector on the frame rotated 90° both ways, so horizontal bodies are found.
- Other missed people (covered, far away, blurred): model *medium* and *Augment*; or just draw
  the box and **Track**. Image sizes above the video's own resolution don't help.
- Wrong POI after Detect + track: double-click the right person, then **⚙ → Follow the current
  red POI** (re-runs detection and keeps that person as POI through the shot) or **Track forward**.
- API: `POST /api/detect-track/<video_id>` with `{"start": 0, "poi": "largest" | "follow",
  "seed_frame": 12, "seed_box": [x1, y1, x2, y2], "model": "yolo11s.pt", "lying": true, "conf": 0.35}`
  queues the same job (`/api/auto-vip/<video_id>` = whole video, largest person).
- Any existing `data/<video>` folders from older versions still open (listed under *Uploads*).

---

## Project files

```
app.py                     FastAPI backend (library, import, jobs, detect, track, export)
vip.py                     Shared YOLO / ByteTrack / VIP selection / single-target tracker
trauma_vip_annotation.py   Batch CLI (import folder + Auto-VIP + preview video)
static/index.html          UI
static/app.js              Frontend logic
static/style.css           Theme
requirements.txt
```
