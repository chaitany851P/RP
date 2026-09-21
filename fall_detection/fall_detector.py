"""
BRANCH: feature/fall-detection
Detects: Person falling or lying unconscious
Fixes:
  - Stricter aspect ratio to avoid false positives on slanted persons
  - Min bbox size filter to reject objects detected as persons
  - Added spine angle check for stronger confirmation
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import numpy as np
import math, urllib.request, sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.alert import draw_alert, log_alert

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Landmark indices ──────────────────────────────────────────────────────────
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28
LEFT_KNEE      = 25
RIGHT_KNEE     = 26

# ── Thresholds ────────────────────────────────────────────────────────────────
YOLO_CONF_THRESHOLD    = 0.70   # raised — reduce object false positives
ASPECT_RATIO_THRESHOLD = 1.8    # raised — slanted person ~1.2, fallen person >1.8
VERTICAL_THRESHOLD     = 0.55   # shoulders must be in lower 45% of frame
SPINE_ANGLE_THRESHOLD  = 60     # spine angle from vertical > 60° = horizontal body
FALL_CONFIRM_FRAMES    = 6      # needs more frames for confirmation
MIN_BBOX_AREA          = 4000   # ignore tiny detections (objects/noise)
MIN_BBOX_WIDTH         = 50     # ignore very narrow detections
MIN_BBOX_HEIGHT        = 40     # ignore very short detections

YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
POSE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
POSE_MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def download_pose_model():
    if not os.path.exists(POSE_MODEL_PATH):
        print("[FallDetector] Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("[FallDetector] Done.")


def spine_angle_from_vertical(ls, rs, lh, rh):
    """Calculate how tilted the spine is from vertical (0=upright, 90=horizontal)."""
    shoulder_mid = np.array([(ls.x + rs.x) / 2, (ls.y + rs.y) / 2])
    hip_mid      = np.array([(lh.x + rh.x) / 2, (lh.y + rh.y) / 2])
    spine_vec    = shoulder_mid - hip_mid
    vertical     = np.array([0, -1])  # pointing up
    cos_a = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-5)
    angle = math.degrees(math.acos(np.clip(cos_a, -1, 1)))
    return angle


def is_valid_bbox(x1, y1, x2, y2):
    """Filter out tiny/invalid bounding boxes that are likely objects, not people."""
    w = x2 - x1
    h = y2 - y1
    area = w * h
    return area >= MIN_BBOX_AREA and w >= MIN_BBOX_WIDTH and h >= MIN_BBOX_HEIGHT


class FallDetector:
    def __init__(self):
        if YOLO_AVAILABLE and os.path.exists(YOLO_MODEL_PATH):
            self.yolo = YOLO(YOLO_MODEL_PATH)
            self.use_yolo = True
            print("[FallDetector] Using trained best.pt ✅")
        else:
            self.yolo = None
            self.use_yolo = False
            print("[FallDetector] best.pt not found — using pose fallback")

        self.landmarker = None
        self.yolo_pose  = None
        try:
            download_pose_model()
            options = PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_poses=6,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            print("[FallDetector] MediaPipe Pose Landmarker ready ✅")
        except Exception as e:
            print(f"[FallDetector] MediaPipe unavailable ({type(e).__name__}). Using YOLOv8-Pose fallback.")
            if YOLO_AVAILABLE:
                self.yolo_pose = YOLO("yolov8n-pose.pt")
                print("[FallDetector] YOLOv8-Pose ready ✅")

        self.fall_counters = {}

    def _yolo_detections(self, frame):
        """Returns list of (bbox, conf) — only valid-sized fall detections."""
        results = self.yolo(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = self.yolo.names[cls_id].lower()
                if any(k in name for k in ['fall', 'lying', 'unconscious', 'down']):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # ← Size filter: reject tiny object detections
                    if is_valid_bbox(x1, y1, x2, y2):
                        detections.append(((x1, y1, x2, y2), conf))
        return detections

    def _draw_skeleton(self, frame, lm, w, h):
        if isinstance(lm, np.ndarray):
            for a, b in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]:
                cv2.line(frame, (int(lm[a][0]), int(lm[a][1])), (int(lm[b][0]), int(lm[b][1])), (0,200,255), 2)
        elif lm is not None:
            for a, b in [(11,12),(11,13),(13,15),(12,14),(14,16),
                         (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)]:
                cv2.line(frame,
                         (int(lm[a].x*w), int(lm[a].y*h)),
                         (int(lm[b].x*w), int(lm[b].y*h)),
                         (0,200,255), 2)

    def _pose_persons(self, frame):
        """Returns list of (bbox, aspect_ratio, spine_angle, is_fallen, lm)."""
        h, w = frame.shape[:2]
        persons = []

        if self.landmarker is not None:
            try:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect(mp_img)

                if result.pose_landmarks:
                    for lm in result.pose_landmarks:
                        ls = lm[LEFT_SHOULDER]
                        rs = lm[RIGHT_SHOULDER]
                        lh = lm[LEFT_HIP]
                        rh = lm[RIGHT_HIP]
                        la = lm[LEFT_ANKLE]
                        ra = lm[RIGHT_ANKLE]

                        xs = [p.x * w for p in lm]
                        ys = [p.y * h for p in lm]
                        x1, y1 = int(min(xs)), int(min(ys))
                        x2, y2 = int(max(xs)), int(max(ys))
                        bbox = (x1, y1, x2, y2)

                        if not is_valid_bbox(x1, y1, x2, y2):
                            continue

                        shoulder_midy = (ls.y + rs.y) / 2
                        ankle_midy    = (la.y + ra.y) / 2
                        body_height   = abs(ankle_midy - shoulder_midy) + 1e-5
                        body_width    = abs(ls.x - rs.x) + abs(lh.x - rh.x)
                        aspect_ratio  = body_width / body_height
                        spine_angle   = spine_angle_from_vertical(ls, rs, lh, rh)

                        fallen = (
                            aspect_ratio  > ASPECT_RATIO_THRESHOLD and
                            shoulder_midy > VERTICAL_THRESHOLD     and
                            spine_angle   > SPINE_ANGLE_THRESHOLD
                        )
                        persons.append((bbox, aspect_ratio, spine_angle, fallen, lm))
                    return persons
            except Exception as e:
                print(f"[FallDetector] MediaPipe detection error: {e}")

        if self.yolo_pose is not None:
            results = self.yolo_pose(frame, verbose=False)
            if results and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                kpts  = results[0].keypoints.xy.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    if not is_valid_bbox(x1, y1, x2, y2):
                        continue
                    pts = kpts[i]
                    ls, rs = pts[5], pts[6]
                    lh, rh = pts[11], pts[12]
                    la, ra = pts[15], pts[16]

                    shoulder_mid = (ls + rs) / 2
                    hip_mid      = (lh + rh) / 2
                    shoulder_midy = (shoulder_mid[1] / h) if h > 0 else 0
                    ankle_midy    = (((la[1] + ra[1]) / 2) / h) if h > 0 else 0

                    body_height   = abs(ankle_midy - shoulder_midy) * h + 1e-5
                    body_width    = abs(ls[0] - rs[0]) + abs(lh[0] - rh[0])
                    aspect_ratio  = body_width / body_height

                    spine_vec     = shoulder_mid - hip_mid
                    vertical      = np.array([0, -1])
                    cos_a         = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-5)
                    spine_angle   = math.degrees(math.acos(np.clip(cos_a, -1, 1)))

                    fallen = (
                        aspect_ratio  > ASPECT_RATIO_THRESHOLD and
                        shoulder_midy > VERTICAL_THRESHOLD     and
                        spine_angle   > SPINE_ANGLE_THRESHOLD
                    )
                    persons.append(((x1, y1, x2, y2), aspect_ratio, spine_angle, fallen, pts))

        return persons

    def detect(self, frame):
        h, w       = frame.shape[:2]
        any_fallen = False

        # ── YOLO path ─────────────────────────────────────────────────────────
        if self.use_yolo:
            detections = self._yolo_detections(frame)
            active_pids = set()

            for (bbox, conf) in detections:
                x1, y1, x2, y2 = bbox
                pid = f"{x1//60}_{y1//60}"
                active_pids.add(pid)
                self.fall_counters[pid] = self.fall_counters.get(pid, 0) + 1
                confirmed = self.fall_counters[pid] >= FALL_CONFIRM_FRAMES

                if confirmed:
                    any_fallen = True
                    log_alert("FALL_DETECTED", extra=f"conf={conf:.2f}")
                    frame = draw_alert(frame, "PERSON FALLEN", bbox, "FALL",
                                       f"{conf:.0%}")
                else:
                    frames_left = FALL_CONFIRM_FRAMES - self.fall_counters[pid]
                    cv2.rectangle(frame, (x1,y1),(x2,y2),(0,165,255),2)
                    cv2.putText(frame, f"Possible fall {conf:.0%}",
                                (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0,165,255), 2)

            # Decay absent persons
            for pid in list(self.fall_counters):
                if pid not in active_pids:
                    self.fall_counters[pid] = max(0, self.fall_counters[pid] - 1)
                    if self.fall_counters[pid] == 0:
                        del self.fall_counters[pid]

            # Skeleton via MediaPipe / YOLOv8-Pose
            persons = self._pose_persons(frame)
            for (_, _, _, _, lm) in persons:
                self._draw_skeleton(frame, lm, w, h)

            cv2.putText(frame,
                        f"YOLO best.pt | conf>{YOLO_CONF_THRESHOLD} | tracked:{len(self.fall_counters)}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
            if not detections:
                cv2.putText(frame, "No falls detected", (10,55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            return frame, any_fallen, None

        # ── MediaPipe fallback ────────────────────────────────────────────────
        persons = self._pose_persons(frame)

        for i, (bbox, aspect_ratio, spine_angle, fallen, lm) in enumerate(persons):
            x1, y1, x2, y2 = bbox
            pid = f"p{i}_{x1//60}_{y1//60}"

            if fallen:
                self.fall_counters[pid] = self.fall_counters.get(pid, 0) + 1
            else:
                self.fall_counters[pid] = max(0, self.fall_counters.get(pid, 0) - 1)

            confirmed = self.fall_counters.get(pid, 0) >= FALL_CONFIRM_FRAMES

            self._draw_skeleton(frame, lm, w, h)

            if confirmed:
                any_fallen = True
                log_alert("FALL_DETECTED", extra=f"AR={aspect_ratio:.2f} spine={spine_angle:.0f}deg")
                frame = draw_alert(frame, "PERSON FALLEN", bbox, "FALL",
                                   f"AR={aspect_ratio:.1f}")
            elif fallen:
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,165,255),2)
                cv2.putText(frame,
                            f"Possible fall AR={aspect_ratio:.1f} spine={spine_angle:.0f}deg",
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,165,255), 1)
            else:
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,
                            f"Safe | AR={aspect_ratio:.1f} spine={spine_angle:.0f}deg",
                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,255,0), 1)

        cv2.putText(frame, f"MediaPipe | Persons: {len(persons)}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        return frame, any_fallen, None


def run(source=0):
    cap      = cv2.VideoCapture(source)
    detector = FallDetector()
    print("[Fall Detection] Press Q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, fallen, _ = detector.detect(frame)
        cv2.imshow("Fall Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    run(src)
