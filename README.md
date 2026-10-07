# vision

Real-time webcam demos with [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26.

| Script | Task | Default model |
|---|---|---|
| `webcam_detect.py` | Object detection | `yolo26n.pt` |
| `webcam_segment.py` | Instance segmentation | `yolo26n-seg.pt` |
| `webcam_track.py` | Multi-object tracking (ByteTrack / BoT-SORT) | `yolo26n.pt` |

Model weights are downloaded automatically on first run.

## Setup

```bash
pip install -U ultralytics opencv-python
```

## Usage

```bash
python webcam_detect.py
python webcam_segment.py
python webcam_track.py --tracker botsort.yaml
```

Common options: `--model`, `--source` (webcam index), `--conf`, `--imgsz`, `--device` (`cpu` or `0` for GPU).

## Keys

| Key | Action |
|---|---|
| `q` / `ESC` | Quit |
| `m` | Toggle mask-only view (segmentation) |
| `c` | Clear track trails (tracking) |
