"""
Batch VIP annotation (command line).

Walks an input folder (and all sub-folders) in natural order, extracts frames for every video
into the annotator's data folder, runs YOLO + ByteTrack with VIP selection on each one, and
writes annotations.json + <video>_coco.json exactly like the web tool does. Open the web tool
afterwards to review boxes and add behaviours / Trauma labels.

Examples:
    py trauma_vip_annotation.py --input "D:\\Trauma_Dataset_Input"
    py trauma_vip_annotation.py --input "D:\\Trauma_Dataset_Input" --data .\\data --fps 5 --preview
"""

from __future__ import annotations

import argparse
from pathlib import Path

import app
import vip


def render_preview(vid: str, out_path: Path) -> None:
    """Write an MP4 with the POI in red and other people in green (the old script's output video)."""
    import cv2  # type: ignore

    d = app.video_dir(vid)
    data = app.load_annotations(vid)
    frames = data["_frame_order"]
    if not frames:
        return
    fps = app.read_meta(vid).get("fps") or 25
    first = vip.read_image(d / frames[0])
    h, w = first.shape[:2]
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for name in frames:
        img = vip.read_image(d / name)
        if img is None:
            continue
        for o in data["frames"][name]["objects"]:
            x1, y1, x2, y2 = (int(v) for v in o["bbox"])
            color = (0, 0, 255) if o.get("is_poi") else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            tag = ("VIP " if o.get("is_poi") else "") + str(o["id"]).replace("track_", "ID ").replace("lie_", "Lying ")
            cv2.putText(img, tag, (x1, max(12, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(img)
    writer.release()
    tmp.replace(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch YOLO tracking + VIP detection into the annotator data folder")
    ap.add_argument("--input", required=True, help="Folder with videos (sub-folders are included)")
    ap.add_argument("--data", default=str(app.APP_DIR / "data"), help="Annotator data folder (default: ./data)")
    ap.add_argument("--fps", type=float, default=None, help="Extract at this FPS (default: every frame)")
    ap.add_argument("--model", default=vip.DEFAULT_WEIGHTS, choices=vip.WEIGHT_CHOICES,
                    help=f"YOLO weights (default: {vip.DEFAULT_WEIGHTS}; yolo11n.pt is faster, yolo11m.pt more accurate)")
    ap.add_argument("--imgsz", type=int, default=640, help="YOLO inference size")
    ap.add_argument("--no-folder-labels", action="store_true",
                    help="Do not pre-fill Trauma / No trauma from folder names")
    ap.add_argument("--reextract", action="store_true", help="Re-extract videos that already have frames")
    ap.add_argument("--skip-vip", action="store_true", help="Only extract frames")
    ap.add_argument("--preview", action="store_true", help="Also write <video>_vip_preview.mp4 with boxes drawn")
    args = ap.parse_args()

    app.set_data_dir(args.data)
    root = Path(args.input).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input folder not found: {root}")

    print(f"Input : {root}")
    print(f"Data  : {app.DATA_DIR}")
    result = app.import_folder(
        None, root, args.fps, not args.no_folder_labels, auto_vip=False,
        reextract=args.reextract, log=print,
    )
    ready = result["imported"] + result["skipped"]
    if not args.skip_vip:
        for i, vid in enumerate(ready, 1):
            print(f"[VIP {i}/{len(ready)}] {vid}")
            res = app.run_auto_vip(None, vid, args.model, args.imgsz, log=print)
            print(f"    VIP found in {res['poi_frames']}/{res['frames']} frames · "
                  f"lying person in {res.get('lying_frames', 0)}")
            if args.preview:
                out = app.video_dir(vid) / f"{vid}_vip_preview.mp4"
                render_preview(vid, out)
                print(f"    preview: {out}")
    for f in result["failed"]:
        print(f"FAILED {f['video_id']}: {f['error']}")
    print(f"Finished: {len(ready)} video(s) ready, {len(result['failed'])} failed. "
          f"Start the annotator with:  py app.py --data \"{app.DATA_DIR}\"")


if __name__ == "__main__":
    main()
