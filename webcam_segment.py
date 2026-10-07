"""Real-time webcam instance segmentation with Ultralytics YOLO26.

Usage:
    pip install ultralytics opencv-python
    python webcam_segment.py --model yolo26n-seg.pt --source 0 --conf 0.25

Press 'q' or ESC to quit, 'm' to toggle mask-only view.
"""

import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 webcam instance segmentation")
    parser.add_argument("--model", default="yolo26n-seg.pt", help="model weights (yolo26n/s/m/l/x-seg.pt)")
    parser.add_argument("--source", type=int, default=0, help="webcam index")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", default=None, help="e.g. 'cpu', '0' (GPU)")
    return parser.parse_args()


def mask_only_view(frame, result):
    """Black out everything except the segmented objects."""
    if result.masks is None:
        return np.zeros_like(frame)
    # masks.data: (N, H', W') at inference resolution -> resize to frame size
    combined = result.masks.data.any(dim=0).cpu().numpy().astype(np.uint8)
    combined = cv2.resize(combined, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return cv2.bitwise_and(frame, frame, mask=combined)


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

    window = "YOLO26 Webcam Segmentation"
    prev_time = time.time()
    show_mask_only = False

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
                retina_masks=True,
                verbose=False,
            )
            result = results[0]

            if show_mask_only:
                annotated = mask_only_view(frame, result)
            else:
                annotated = result.plot()  # boxes + labels + colored masks

            # FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            num_objects = 0 if result.masks is None else len(result.masks)
            cv2.putText(annotated, f"FPS: {fps:.1f}  Objects: {num_objects}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mask_only = not show_mask_only
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
