"""
BRANCH: feature/fall-detection
Detects: Person falling or lying unconscious
Method:  MediaPipe Pose (new Tasks API, 0.10.30+) keypoint aspect ratio heuristic
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import numpy as np
import urllib.request
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.alert import draw_alert, log_alert

# ── Landmark indices (MediaPipe Pose 33 keypoints) ────────────────────────────
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_HIP       = 23
RIGHT_HIP      = 24
LEFT_ANKLE     = 27
RIGHT_ANKLE    = 28

# ── Thresholds ────────────────────────────────────────────────────────────────
ASPECT_RATIO_THRESHOLD = 1.4
VERTICAL_THRESHOLD     = 0.65
FALL_CONFIRM_FRAMES    = 8

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[FallDetector] Downloading MediaPipe pose model (~5MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[FallDetector] Model downloaded.")


class FallDetector:
    def __init__(self):
        download_model()
        options = PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=4,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker      = PoseLandmarker.create_from_options(options)
        self.fall_frame_count = 0
        self.is_fallen        = False

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        if not result.pose_landmarks:
            self.fall_frame_count = max(0, self.fall_frame_count - 1)
            self.is_fallen = False
            return frame, False, None

        # Use first detected person
        lm = result.pose_landmarks[0]

        ls = lm[LEFT_SHOULDER]
        rs = lm[RIGHT_SHOULDER]
        lh = lm[LEFT_HIP]
        rh = lm[RIGHT_HIP]
        la = lm[LEFT_ANKLE]
        ra = lm[RIGHT_ANKLE]

        # Bounding box
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        x1, y1 = int(min(xs)), int(min(ys))
        x2, y2 = int(max(xs)), int(max(ys))
        bbox   = (x1, y1, x2, y2)

        shoulder_midy = (ls.y + rs.y) / 2
        ankle_midy    = (la.y + ra.y) / 2
        body_height   = abs(ankle_midy - shoulder_midy) + 1e-5
        body_width    = abs(ls.x - rs.x) + abs(lh.x - rh.x)
        aspect_ratio  = body_width / body_height

        fallen_now = (
            aspect_ratio > ASPECT_RATIO_THRESHOLD and
            shoulder_midy > VERTICAL_THRESHOLD
        )

        if fallen_now:
            self.fall_frame_count += 1
        else:
            self.fall_frame_count = max(0, self.fall_frame_count - 1)

        self.is_fallen = self.fall_frame_count >= FALL_CONFIRM_FRAMES

        # Draw skeleton manually
        connections = [
            (11,12),(11,13),(13,15),(12,14),(14,16),
            (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)
        ]
        for a, b in connections:
            ax, ay = int(lm[a].x * w), int(lm[a].y * h)
            bx, by = int(lm[b].x * w), int(lm[b].y * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 200, 255), 2)
        for p in lm:
            cx, cy = int(p.x * w), int(p.y * h)
            cv2.circle(frame, (cx, cy), 4, (255, 255, 0), -1)

        # HUD
        cv2.putText(frame, f"Aspect Ratio: {aspect_ratio:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Shoulder Y:   {shoulder_midy:.2f}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        if self.is_fallen:
            log_alert("FALL_DETECTED", extra=f"aspect={aspect_ratio:.2f}")
            frame = draw_alert(frame, "PERSON FALLEN / UNCONSCIOUS",
                               bbox, "FALL", f"AR={aspect_ratio:.1f}")
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Safe", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return frame, self.is_fallen, bbox


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
