"""Real-time webcam object detection with Ultralytics YOLO26.

Usage:
    pip install ultralytics opencv-python
    python webcam_detect.py --model yolo26n.pt --source 0 --conf 0.25

Press 'q' or ESC to quit.
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 webcam object detection")
    parser.add_argument("--model", default="yolo26n.pt", help="model weights (yolo26n/s/m/l/x.pt)")
    parser.add_argument("--source", type=int, default=0, help="webcam index")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", default=None, help="e.g. 'cpu', '0' (GPU)")
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

    window = "YOLO26 Webcam Detection"
    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam")
                break

            results = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            cv2.putText(annotated, f"FPS: {fps:.1f}  Objects: {len(result.boxes)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
