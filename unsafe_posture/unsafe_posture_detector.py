"""
BRANCH: feature/unsafe-posture-detection
Detects: Unsafe activities near machines — smoking, eating, drinking, mobile phone use
         + pose-based unsafe postures (spine bend, deep squat, proximity to machine)
Method:  YOLOv8 trained model (posture_best.pt) for activity detection
         + MediaPipe Pose for keypoint-based posture analysis
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
import numpy as np
import math, sys, os, time, urllib.request

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.alert import draw_alert, log_alert
from utils.zone import point_in_polygon, draw_zones, load_zones

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Landmark indices ──────────────────────────────────────────────────────────
NOSE=0; L_SHOULDER=11; R_SHOULDER=12; L_HIP=23; R_HIP=24
L_KNEE=25; R_KNEE=26; L_ANKLE=27; R_ANKLE=28
L_WRIST=15; R_WRIST=16

# ── Thresholds ────────────────────────────────────────────────────────────────
SPINE_BEND_THRESHOLD   = 90
NECK_FORWARD_THRESHOLD = 120
KNEE_SQUAT_THRESHOLD   = 100
HEAD_MACHINE_PROX_PX   = 80
CONFIRM_FRAMES         = 5
YOLO_CONF_THRESHOLD    = 0.50

# ── Unsafe activity classes from trained model ────────────────────────────────
UNSAFE_CLASSES = ['cigarette', 'drinking', 'eating', 'mobile']

MACHINE_ZONES = {}

# ── Model paths ───────────────────────────────────────────────────────────────
POSTURE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "posture_best.pt")
POSE_MODEL_PATH    = os.path.join(os.path.dirname(__file__), "..", "fall_detection", "pose_landmarker.task")
POSE_MODEL_URL     = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"


def download_pose_model():
    if not os.path.exists(POSE_MODEL_PATH):
        print("[PostureDetector] Downloading MediaPipe pose model (~5MB)...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
        print("[PostureDetector] Pose model downloaded.")


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
        if machine_zones is not None:
            self.zones = machine_zones
        else:
            loaded = load_zones()
            self.zones = loaded if loaded else MACHINE_ZONES

        # ── Load trained YOLOv8 activity detection model ──────────────────────
        if YOLO_AVAILABLE and os.path.exists(POSTURE_MODEL_PATH):
            self.yolo_posture     = YOLO(POSTURE_MODEL_PATH)
            self.use_yolo_posture = True
            print("[PostureDetector] Using trained posture_best.pt ✅")
            print(f"[PostureDetector] Classes: {self.yolo_posture.names}")
        else:
            self.yolo_posture     = None
            self.use_yolo_posture = False
            if not os.path.exists(POSTURE_MODEL_PATH):
                print("[PostureDetector] WARNING: posture_best.pt not found — using MediaPipe only")
            else:
                print("[PostureDetector] WARNING: ultralytics not installed")

        # ── Load MediaPipe Pose (with YOLO-Pose fallback) ─────────────────────
        self.landmarker = None
        self.yolo_pose  = None
        try:
            download_pose_model()
            options = PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=RunningMode.IMAGE,
                num_poses=4,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = PoseLandmarker.create_from_options(options)
            print("[PostureDetector] MediaPipe Pose Landmarker ready ✅")
        except Exception as e:
            print(f"[PostureDetector] MediaPipe unavailable ({type(e).__name__}). Using YOLOv8-Pose fallback.")
            if YOLO_AVAILABLE:
                self.yolo_pose = YOLO("yolov8n-pose.pt")
                print("[PostureDetector] YOLOv8-Pose ready ✅")

        self.unsafe_frame_count = {}
        self.yolo_frame_count   = {}

    def _extract_pose_data(self, frame, h, w):
        """Extract skeletal joints, bounding box, and draw callback using MediaPipe or YOLO-Pose."""
        if self.landmarker is not None:
            try:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect(mp_img)
                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    def px(idx):
                        return (int(lm[idx].x * w), int(lm[idx].y * h))
                    joints = {
                        "nose": px(NOSE),
                        "l_shoulder": px(L_SHOULDER), "r_shoulder": px(R_SHOULDER),
                        "l_hip": px(L_HIP), "r_hip": px(R_HIP),
                        "l_knee": px(L_KNEE), "r_knee": px(R_KNEE),
                        "l_ankle": px(L_ANKLE), "r_ankle": px(R_ANKLE),
                        "l_wrist": px(L_WRIST), "r_wrist": px(R_WRIST),
                    }
                    xs = [int(p.x * w) for p in lm]
                    ys = [int(p.y * h) for p in lm]
                    bbox = (min(xs), min(ys), max(xs), max(ys))
                    def draw_skel(f):
                        for a, b in [(11,12),(11,13),(13,15),(12,14),(14,16),
                                     (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28)]:
                            cv2.line(f, (int(lm[a].x*w), int(lm[a].y*h)), (int(lm[b].x*w), int(lm[b].y*h)), (0, 200, 255), 2)
                    return joints, bbox, draw_skel
            except Exception as e:
                print(f"[PostureDetector] MediaPipe detection error: {e}")

        if self.yolo_pose is not None:
            results = self.yolo_pose(frame, verbose=False)
            if results and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                kpts = results[0].keypoints.xy.cpu().numpy()[0]
                box  = results[0].boxes.xyxy.cpu().numpy()[0]
                joints = {
                    "nose": (int(kpts[0][0]), int(kpts[0][1])),
                    "l_shoulder": (int(kpts[5][0]), int(kpts[5][1])),
                    "r_shoulder": (int(kpts[6][0]), int(kpts[6][1])),
                    "l_hip": (int(kpts[11][0]), int(kpts[11][1])),
                    "r_hip": (int(kpts[12][0]), int(kpts[12][1])),
                    "l_knee": (int(kpts[13][0]), int(kpts[13][1])),
                    "r_knee": (int(kpts[14][0]), int(kpts[14][1])),
                    "l_ankle": (int(kpts[15][0]), int(kpts[15][1])),
                    "r_ankle": (int(kpts[16][0]), int(kpts[16][1])),
                    "l_wrist": (int(kpts[9][0]), int(kpts[9][1])),
                    "r_wrist": (int(kpts[10][0]), int(kpts[10][1])),
                }
                bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
                def draw_skel(f):
                    for a, b in [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(12,14),(13,15),(14,16)]:
                        cv2.line(f, (int(kpts[a][0]), int(kpts[a][1])), (int(kpts[b][0]), int(kpts[b][1])), (0, 200, 255), 2)
                return joints, bbox, draw_skel

        return None, None, None

    def _detect_activities_yolo(self, frame):
        """Detect smoking, eating, drinking, mobile using trained YOLOv8 model."""
        if not self.use_yolo_posture:
            return frame, []

        alerts  = []
        results = self.yolo_posture(frame, conf=YOLO_CONF_THRESHOLD, verbose=False)

        for r in results:
            for box in r.boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                name    = self.yolo_posture.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Track confirmation frames per detection grid cell
                pid = f"{x1//60}_{y1//60}"
                self.yolo_frame_count[pid] = self.yolo_frame_count.get(pid, 0) + 1
                confirmed = self.yolo_frame_count[pid] >= CONFIRM_FRAMES

                # Draw detection box
                color = (0, 80, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if confirmed:
                    # Check if activity is forbidden in current zone
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    zone_name = "ZONE"
                    for zname, zinfo in self.zones.items():
                        if point_in_polygon((cx, cy), zinfo.get("polygon", [])):
                            forbidden = zinfo.get("forbidden", [])
                            allowed   = zinfo.get("allowed", [])
                            if name in allowed:
                                # Activity is allowed here — green box
                                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                                cv2.putText(frame, f"{name} ALLOWED {conf:.0%}",
                                            (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.55, (0,255,0), 2)
                                break
                            zone_name = zname

                    frame = draw_alert(frame, f"UNSAFE: {name.upper()}",
                                       (x1,y1,x2,y2), "UNSAFE_POSTURE",
                                       f"{conf:.0%} | {zone_name}")
                    log_alert("UNSAFE_ACTIVITY", extra=f"{name} conf={conf:.2f}")
                    alerts.append({"type": "UNSAFE_ACTIVITY", "issues": [name],
                                   "conf": conf})
                else:
                    cv2.putText(frame, f"{name} {conf:.0%} [{self.yolo_frame_count[pid]}/{CONFIRM_FRAMES}]",
                                (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, color, 2)

        # Decay absent detections
        active = set()
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                active.add(f"{x1//60}_{y1//60}")
        for pid in list(self.yolo_frame_count):
            if pid not in active:
                self.yolo_frame_count[pid] = max(0, self.yolo_frame_count[pid]-1)
                if self.yolo_frame_count[pid] == 0:
                    del self.yolo_frame_count[pid]

        return frame, alerts

    def detect(self, frame):
        h, w   = frame.shape[:2]
        alerts = []

        # Draw zones
        frame = draw_zones(frame, self.zones)

        # ── Step 1: YOLO activity detection (smoking, eating, drinking, mobile) ──
        frame, yolo_alerts = self._detect_activities_yolo(frame)
        alerts.extend(yolo_alerts)

        # ── Step 2: Pose analysis (MediaPipe or YOLOv8-Pose) ─────────────────
        joints, bbox, draw_skel = self._extract_pose_data(frame, h, w)

        if joints is None:
            if not self.use_yolo_posture:
                cv2.putText(frame, "No person detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
            return frame, alerts

        nose       = joints["nose"]
        l_shoulder = joints["l_shoulder"]
        r_shoulder = joints["r_shoulder"]
        l_hip      = joints["l_hip"]
        r_hip      = joints["r_hip"]
        l_knee     = joints["l_knee"]
        r_knee     = joints["r_knee"]
        l_ankle    = joints["l_ankle"]
        r_ankle    = joints["r_ankle"]
        l_wrist    = joints["l_wrist"]
        r_wrist    = joints["r_wrist"]

        mid_shoulder = ((l_shoulder[0]+r_shoulder[0])//2, (l_shoulder[1]+r_shoulder[1])//2)
        mid_hip      = ((l_hip[0]+r_hip[0])//2, (l_hip[1]+r_hip[1])//2)
        virtual_up   = (mid_hip[0], mid_hip[1] - 100)

        spine_angle    = angle_between(virtual_up, mid_hip, mid_shoulder)
        neck_angle     = angle_between(mid_shoulder, (mid_shoulder[0], mid_shoulder[1]-50), nose)
        l_knee_angle   = angle_between(l_hip, l_knee, l_ankle)
        r_knee_angle   = angle_between(r_hip, r_knee, r_ankle)
        min_knee_angle = min(l_knee_angle, r_knee_angle)

        # Machine proximity check
        near_machine = False
        for zname, zinfo in self.zones.items():
            if not zinfo.get("is_machine_zone", False):
                continue
            for pt_name, pt in [("HEAD", nose), ("L_HAND", l_wrist), ("R_HAND", r_wrist)]:
                dist = point_dist_to_polygon_edge(pt, zinfo.get("polygon", []))
                if dist < HEAD_MACHINE_PROX_PX:
                    near_machine = True
                    cv2.circle(frame, pt, 8, (0, 0, 255), -1)
                    cv2.putText(frame, f"{pt_name} {dist:.0f}px to machine",
                                (pt[0]+10, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)

        # Collect pose issues
        pose_issues = []
        if spine_angle > SPINE_BEND_THRESHOLD:
            pose_issues.append(f"SPINE BEND {spine_angle:.0f}deg")
        if neck_angle > NECK_FORWARD_THRESHOLD:
            pose_issues.append(f"NECK LEAN {neck_angle:.0f}deg")
        if min_knee_angle < KNEE_SQUAT_THRESHOLD:
            pose_issues.append(f"DEEP SQUAT {min_knee_angle:.0f}deg")
        if near_machine:
            pose_issues.append("TOO CLOSE TO MACHINE")

        # Draw skeleton
        if draw_skel:
            draw_skel(frame)

        # HUD — only show if no YOLO alerts to keep screen clean
        hud_y = 30
        backend_name = "MediaPipe" if self.landmarker is not None else ("YOLO-Pose" if self.yolo_pose is not None else "None")
        model_label = f"YOLO+{backend_name}" if self.use_yolo_posture else backend_name
        cv2.putText(frame, f"Model: {model_label}", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
        hud_y += 22
        cv2.putText(frame, f"Spine: {spine_angle:.1f}deg", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,255) if spine_angle > SPINE_BEND_THRESHOLD else (0,255,0), 1)
        hud_y += 22
        cv2.putText(frame, f"Knee:  {min_knee_angle:.1f}deg", (10, hud_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,0,255) if min_knee_angle < KNEE_SQUAT_THRESHOLD else (0,255,0), 1)

        # Pose-based alert
        pid = f"{nose[0]//50}_{nose[1]//50}"
        self.unsafe_frame_count[pid] = self.unsafe_frame_count.get(pid, 0) + (1 if pose_issues else -1)
        self.unsafe_frame_count[pid] = max(0, self.unsafe_frame_count[pid])

        if pose_issues and self.unsafe_frame_count.get(pid, 0) >= CONFIRM_FRAMES:
            log_alert("UNSAFE_POSTURE", extra=" | ".join(pose_issues))
            frame = draw_alert(frame, "UNSAFE POSTURE", bbox,
                               "UNSAFE_POSTURE", pose_issues[0])
            alerts.append({"type": "UNSAFE_POSTURE", "issues": pose_issues})
        elif not yolo_alerts:
            cv2.rectangle(frame, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,255,0),2)
            cv2.putText(frame, "SAFE", (bbox[0], bbox[1]-8),
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
        cv2.imshow("Unsafe Activity Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        src = int(src)
    except:
        pass
    run(src)