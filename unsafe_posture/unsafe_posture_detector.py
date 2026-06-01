"""
BRANCH: feature/unsafe-posture-detection
Detects: Unsafe postures near machines — bending, leaning too close, improper alignment
Method:  MediaPipe Pose (new Tasks API, 0.10.30+) + angle calculation
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import numpy as np
import math, sys, os, time, urllib.request

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.alert import draw_alert, log_alert
from utils.zone import point_in_polygon, draw_zones

# ── Landmark indices ──────────────────────────────────────────────────────────
NOSE=0; L_SHOULDER=11; R_SHOULDER=12; L_HIP=23; R_HIP=24
L_KNEE=25; R_KNEE=26; L_ANKLE=27; R_ANKLE=28
L_WRIST=15; R_WRIST=16

# ── Thresholds ────────────────────────────────────────────────────────────────
SPINE_BEND_THRESHOLD   = 40
NECK_FORWARD_THRESHOLD = 35
KNEE_SQUAT_THRESHOLD   = 100
HEAD_MACHINE_PROX_PX   = 80
CONFIRM_FRAMES         = 5

MACHINE_ZONES = {
    "MACHINE_1": {
        "polygon": [(50, 150), (280, 150), (280, 420), (50, 420)],
        "color": (255, 180, 0),
    },
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "fall_detection", "pose_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[PostureDetector] Downloading MediaPipe pose model (~5MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[PostureDetector] Model downloaded.")


def angle_between(a, b, c):
    ba = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
    bc = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-5)
    return math.degrees(math.acos(np.clip(cos_a, -1, 1)))


def point_dist_to_polygon_edge(point, polygon):
    min_dist = float("inf")
    pts = np.array(polygon, dtype=np.float32)
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i+1) % n]
        ab = b - a
        t  = np.dot(np.array(point, dtype=np.float32) - a, ab) / (np.dot(ab, ab) + 1e-5)
        t  = np.clip(t, 0, 1)
        proj = a + t * ab
        dist = np.linalg.norm(np.array(point, dtype=np.float32) - proj)
        min_dist = min(min_dist, dist)
    return min_dist


class UnsafePostureDetector:
    def __init__(self, machine_zones=None):
        self.zones = machine_zones or MACHINE_ZONES
        download_model()
        options = PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_poses=4,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker         = PoseLandmarker.create_from_options(options)
        self.unsafe_frame_count = {}

    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)
        alerts = []

        frame = draw_zones(frame, self.zones)

        if not result.pose_landmarks:
            return frame, []

        lm = result.pose_landmarks[0]

        def px(idx):
            return (int(lm[idx].x * w), int(lm[idx].y * h))

        nose       = px(NOSE)
        l_shoulder = px(L_SHOULDER)
        r_shoulder = px(R_SHOULDER)
        l_hip      = px(L_HIP)
        r_hip      = px(R_HIP)
        l_knee     = px(L_KNEE)
        r_knee     = px(R_KNEE)
        l_ankle    = px(L_ANKLE)
        r_ankle    = px(R_ANKLE)
        l_wrist    = px(L_WRIST)
        r_wrist    = px(R_WRIST)

        mid_shoulder = ((l_shoulder[0]+r_shoulder[0])//2, (l_shoulder[1]+r_shoulder[1])//2)
        mid_hip      = ((l_hip[0]+r_hip[0])//2, (l_hip[1]+r_hip[1])//2)
        virtual_up   = (mid_hip[0], mid_hip[1] - 100)

        spine_angle    = angle_between(virtual_up, mid_hip, mid_shoulder)
        neck_angle     = angle_between(mid_shoulder, (mid_shoulder[0], mid_shoulder[1]-50), nose)
        l_knee_angle   = angle_between(l_hip, l_knee, l_ankle)
        r_knee_angle   = angle_between(r_hip, r_knee, r_ankle)
        min_knee_angle = min(l_knee_angle, r_knee_angle)

        near_machine = False
        for zname, zinfo in self.zones.items():
            for pt_name, pt in [("HEAD", nose), ("L_HAND", l_wrist), ("R_HAND", r_wrist)]:
                dist = point_dist_to_polygon_edge(pt, zinfo["polygon"])
                if dist < HEAD_MACHINE_PROX_PX:
                    near_machine = True
                    cv2.circle(frame, pt, 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"{pt_name} {dist:.0f}px",
                                (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

        issues = []
        if spine_angle > SPINE_BEND_THRESHOLD:
            issues.append(f"SPINE BEND {spine_angle:.0f}deg")
        if neck_angle > NECK_FORWARD_THRESHOLD:
            issues.append(f"NECK LEAN {neck_angle:.0f}deg")
        if min_knee_angle < KNEE_SQUAT_THRESHOLD:
            issues.append(f"DEEP SQUAT {min_knee_angle:.0f}deg")
        if near_machine:
            issues.append("TOO CLOSE TO MACHINE")

        # Draw skeleton
        connections = [
            (11,12),(11,13),(13,15),(12,14),(14,16),
            (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)
        ]
        for a, b in connections:
            ax, ay = int(lm[a].x * w), int(lm[a].y * h)
            bx, by = int(lm[b].x * w), int(lm[b].y * h)
            cv2.line(frame, (ax, ay), (bx, by), (0, 200, 255), 2)

        # HUD angles
        cv2.putText(frame, f"Spine: {spine_angle:.1f}deg", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,0,255) if spine_angle > SPINE_BEND_THRESHOLD else (0,255,0), 2)
        cv2.putText(frame, f"Neck:  {neck_angle:.1f}deg", (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,0,255) if neck_angle > NECK_FORWARD_THRESHOLD else (0,255,0), 2)
        cv2.putText(frame, f"Knee:  {min_knee_angle:.1f}deg", (10, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,0,255) if min_knee_angle < KNEE_SQUAT_THRESHOLD else (0,255,0), 2)

        pid = f"{nose[0]//50}_{nose[1]//50}"
        self.unsafe_frame_count[pid] = self.unsafe_frame_count.get(pid, 0) + (1 if issues else -1)
        self.unsafe_frame_count[pid] = max(0, self.unsafe_frame_count[pid])

        xs   = [int(p.x*w) for p in lm]
        ys   = [int(p.y*h) for p in lm]
        bbox = (min(xs), min(ys), max(xs), max(ys))

        if issues and self.unsafe_frame_count.get(pid, 0) >= CONFIRM_FRAMES:
            log_alert("UNSAFE_POSTURE", extra=" | ".join(issues))
            frame = draw_alert(frame, "UNSAFE POSTURE", bbox, "UNSAFE_POSTURE", issues[0])
            alerts.append({"type": "UNSAFE_POSTURE", "issues": issues})
        else:
            cv2.rectangle(frame, (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,255,0), 2)
            cv2.putText(frame, "SAFE POSTURE", (bbox[0], bbox[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        return frame, alerts


def run(source=0):
    cap      = cv2.VideoCapture(source)
    detector = UnsafePostureDetector()
    print("[Unsafe Posture Detection] Press Q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = detector.detect(frame)
        cv2.imshow("Unsafe Posture Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    run(src)
