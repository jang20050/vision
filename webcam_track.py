"""Real-time webcam multi-object tracking with Ultralytics YOLO26.

Usage:
    pip install ultralytics opencv-python
    python webcam_track.py --model yolo26n.pt --tracker bytetrack.yaml --source 0

Press 'q' or ESC to quit, 'c' to clear track trails.
"""

import argparse
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 webcam object tracking")
    parser.add_argument("--model", default="yolo26n.pt", help="model weights (yolo26n/s/m/l/x.pt, or *-seg.pt)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--source", type=int, default=0, help="webcam index")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", default=None, help="e.g. 'cpu', '0' (GPU)")
    parser.add_argument("--trail", type=int, default=30, help="number of past points to draw per track")
    return parser.parse_args()


def main():
    args = parse_args()

    # Weights are downloaded automatically on first run
    model = YOLO(args.model)

    # CAP_DSHOW opens Windows webcams faster and more reliably
    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam (index {args.source})")

    window = "YOLO26 Webcam Tracking"
    prev_time = time.time()
    # track_id -> recent box centers
    trails = defaultdict(lambda: deque(maxlen=args.trail))
    seen_ids = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam")
                break

            # persist=True keeps tracker state (IDs) across frames
            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()  # boxes + "id:N class conf" labels

            boxes = result.boxes
            active_ids = []
            if boxes.id is not None:
                xywh = boxes.xywh.cpu().numpy()
                ids = boxes.id.int().cpu().tolist()
                for (x, y, _, _), track_id in zip(xywh, ids):
                    trails[track_id].append((int(x), int(y)))
                    active_ids.append(track_id)
                    seen_ids.add(track_id)

            # Draw motion trails for currently visible tracks
            for track_id in active_ids:
                points = np.array(trails[track_id], dtype=np.int32).reshape(-1, 1, 2)
                if len(points) > 1:
                    cv2.polylines(annotated, [points], isClosed=False, color=(230, 230, 230), thickness=2)

            # Drop trails of tracks that are gone
            for track_id in list(trails):
                if track_id not in active_ids:
                    del trails[track_id]

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(annotated, f"FPS: {fps:.1f}  Tracks: {len(active_ids)}  Total IDs: {len(seen_ids)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                trails.clear()
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
