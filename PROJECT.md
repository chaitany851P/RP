# Project: Industrial Safety Detection System (RP)

An automated computer vision safety monitoring system designed for industrial environments to detect workplace hazards and safety violations in real-time from CCTV or webcam video feeds.

## Tech Stack
- **Language**: Python 3.12+
- **Computer Vision**: OpenCV (`cv2`), NumPy
- **Deep Learning / Object Detection**: Ultralytics YOLOv8 (`yolov8n.pt`, custom `best.pt`, `posture_best.pt`)
- **Pose Estimation**: MediaPipe Pose Landmarker (`pose_landmarker.task`)
- **Tracking**: ByteTrack (via Ultralytics YOLO track)

## Project Structure Overview
- `core/`: Master integration pipeline (`pipeline.py`) orchestrating all detection modules and rendering the unified UI/status HUD.
- `fall_detection/`: Detects fallen/unconscious workers via fine-tuned YOLOv8 (`best.pt`) or MediaPipe pose estimation fallback (aspect ratio + spine tilt). Includes Roboflow dataset (`train`, `valid`, `test`, `data.yaml`).
- `loitering_detection/`: Tracks persons inside restricted polygons using YOLOv8 + ByteTrack; flags stationary loitering (>10s) and suspicious roaming.
- `prolonged_exposure/`: Monitors accumulated duration spent by individuals near hazardous machinery zones with visual progress bars.
- `unauthorized_entry/`: Enforces strict exclusion zones with instant alerts and flashing warning boxes upon breach.
- `unsafe_posture/`: Detects safety violations:
  - Action/object violations (cigarette smoking, eating, drinking, mobile phone use) via `posture_best.pt`.
  - Ergonomic/posture violations (spine angle >90°, neck lean >120°, deep squat <100°, machine boundary proximity <80px) via MediaPipe.
- `utils/`: Shared utilities for polygon geometry (`zone.py`) and visual/console alert dispatching (`alert.py`).
- `zone_setup.py`: Interactive OpenCV GUI tool to calibrate zone polygons, permission rules (allowed/forbidden activities), and exposure thresholds into `config.json`.
- `test_all.py`: Verification script testing dependencies, module imports, detector initialization, and dummy frame inference.
- `config.json`: Persisted zone definitions and safety rules.
- `RESEARCH PAPER`: Reference link to documentation/academic paper.

## How to Run
- Run test suite: `py test_all.py`
- Setup custom zones: `py zone_setup.py <video_or_camera_id>`
- Run full integrated pipeline: `py core/pipeline.py [input_video] [output_video]`
- Run individual detector: `py <feature_folder>/<detector_file>.py [input_video]`

## Key Features Implemented
- Multi-hazard detection (falls, loitering/roaming, prolonged machine exposure, unauthorized entry, posture/action violations).
- Hybrid AI approach: combines fine-tuned YOLOv8 bounding box detection with MediaPipe skeletal keypoint kinematics.
- Multi-camera zone calibration utility with interactive polygon drawing and rule assignment.
- Visual HUD overlays: bounding boxes, skeleton wireframes, zone overlays, centroid trails, progress bars, and alert banners.

## Recent Changes
- 2026-09-21: Full remediation: resolved README.md merge conflict markers, added dynamic config.json zone loading across all detectors, implemented dual-engine pose fallback (MediaPipe + YOLOv8-Pose) to bypass Windows Application Control DLL blocks, and installed all dependencies (24/24 tests passing).
- 2026-09-21: Codebase audit & initial PROJECT.md specification created.
- 2026-09-21: Research paper reference added to project.
- 2026-09-21: Fall detection feature merged into main with trained YOLOv8 + MediaPipe pose fallback.
