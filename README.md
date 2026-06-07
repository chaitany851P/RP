
# RP
# Industrial Safety Detection System

## Branch Strategy

| Feature | Branch | File |
|---|---|---|
| Fall / Unconscious detection | `feature/fall-detection` | `fall_detection/fall_detector.py` |
| Loitering / Suspicious roaming | `feature/loitering-detection` | `loitering_detection/loitering_detector.py` |
| Prolonged exposure near machines | `feature/prolonged-exposure-detection` | `prolonged_exposure/prolonged_exposure_detector.py` |
| Unauthorized danger zone entry | `feature/unauthorized-entry-detection` | `unauthorized_entry/unauthorized_entry_detector.py` |
| Unsafe posture near machines | `feature/unsafe-posture-detection` | `unsafe_posture/unsafe_posture_detector.py` |
| Full integration pipeline | `feature/activity-detection-core` | `core/pipeline.py` |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Run Individual Features

```bash
# Feature 1 — Fall detection (webcam)
python fall_detection/fall_detector.py

# Feature 1 — Fall detection (video file)
python fall_detection/fall_detector.py path/to/video.mp4

# Feature 2 — Loitering detection
python loitering_detection/loitering_detector.py path/to/video.mp4

# Feature 3 — Prolonged exposure
python prolonged_exposure/prolonged_exposure_detector.py path/to/video.mp4

# Feature 4 — Unauthorized entry
python unauthorized_entry/unauthorized_entry_detector.py path/to/video.mp4

# Feature 5 — Unsafe posture
python unsafe_posture/unsafe_posture_detector.py path/to/video.mp4

# All features — Full pipeline (webcam)
python core/pipeline.py

# All features — Full pipeline with output video saved
python core/pipeline.py path/to/video.mp4 output.mp4
```

---

## Git Setup

```bash
git init
git add .
git commit -m "Initial project structure"

# Create and switch to each feature branch
git checkout -b feature/fall-detection
git checkout -b feature/loitering-detection
git checkout -b feature/prolonged-exposure-detection
git checkout -b feature/unauthorized-entry-detection
git checkout -b feature/unsafe-posture-detection
git checkout -b feature/activity-detection-core
```

---

## How Each Feature Works

### 1. Fall Detection (`feature/fall-detection`)
- Uses **MediaPipe Pose** to extract body keypoints
- Calculates body **aspect ratio** (width/height) — a fallen person is horizontal
- If `aspect_ratio > 1.4` AND `shoulder_Y > 0.65` for 8+ consecutive frames → **FALL ALERT**
- No GPU required, runs on CPU in real-time

### 2. Loitering Detection (`feature/loitering-detection`)
- Uses **YOLOv8n** for person detection + **ByteTrack** for cross-frame tracking
- Each person gets a unique Track ID
- If person stays inside a restricted zone polygon for `> 10 seconds` → **LOITERING ALERT**
- Also detects roaming (moving within zone without leaving)
- Shows trail of movement history

### 3. Prolonged Exposure (`feature/prolonged-exposure-detection`)
- Similar to loitering but zones are **machine proximity areas**
- Tracks **accumulated time** per person per machine (even across brief departures)
- Shows progress bar per person counting toward time limit
- Warning at 8s, alert at 15s (configurable per machine)

### 4. Unauthorized Entry (`feature/unauthorized-entry-detection`)
- Alert fires **immediately on zone entry** (3-frame grace period to avoid flicker)
- Zones are hard **DANGER ZONES** — no presence allowed
- Flashing red bounding box while person is inside zone
- Tracks duration inside zone

### 5. Unsafe Posture (`feature/unsafe-posture-detection`)
- Uses **MediaPipe Pose** angle calculations:
  - **Spine bend angle** > 40° → unsafe forward lean
  - **Neck forward lean** > 35° → unsafe head position
  - **Knee angle** < 100° → deep unsafe squat
  - **Head/hand proximity** < 80px to machine zone edge → too close
- HUD shows live angle readings with color coding

---

## Customizing Zones

Edit the zone polygons in each detector file:

```python
MACHINE_ZONES = {
    "MACHINE_1": {
        "polygon": [(x1,y1), (x2,y2), (x3,y3), (x4,y4)],  # clockwise points
        "color": (255, 100, 0),
    },
}
```

Use a tool like [labelme](https://github.com/labelmeai/labelme) or print coordinates by clicking on a reference frame.
>>>>>>> 977890a (Initial commit)
