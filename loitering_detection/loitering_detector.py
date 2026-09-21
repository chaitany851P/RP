"""
BRANCH: feature/loitering-detection
Detects: Suspicious roaming or loitering near restricted/sensitive areas
Method:  YOLOv8 person detection + ByteTrack + movement pattern analysis
"""

import cv2
import numpy as np
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.alert import draw_alert, log_alert
from utils.zone import point_in_polygon, draw_zones, get_centroid, load_zones

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")

# ── Config ────────────────────────────────────────────────────────────────────
LOITER_TIME_THRESHOLD  = 10.0   # seconds before loitering alert
ROAM_DISTANCE_THRESHOLD = 150   # pixels — movement without leaving zone
CONF_THRESHOLD         = 0.4

# Default restricted zones (customize per camera)
DEFAULT_ZONES = {
}


class LoiteringDetector:
    def __init__(self, zones=None, fps=25):
        if zones is not None:
            self.zones = zones
        else:
            loaded = load_zones()
            self.zones = loaded if loaded else DEFAULT_ZONES
        self.fps   = fps
        self.model = YOLO("yolov8n.pt") if YOLO_AVAILABLE else None

        # Per track-ID state
        self.zone_entry_time = {}    # track_id → timestamp when entered zone
        self.track_history   = {}    # track_id → list of (cx, cy) positions
        self.alert_active    = {}    # track_id → bool

    def _detect_persons(self, frame):
        """Returns list of (x1,y1,x2,y2, track_id)."""
        if self.model is None:
            return []
        results = self.model.track(frame, persist=True, conf=CONF_THRESHOLD,
                                   classes=[0], verbose=False)
        persons = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id[0])
                persons.append((x1, y1, x2, y2, tid))
        return persons

    def _is_roaming(self, track_id):
        """True if person keeps moving within zone (not standing still)."""
        hist = self.track_history.get(track_id, [])
        if len(hist) < 10:
            return False
        total_dist = sum(
            np.hypot(hist[i][0]-hist[i-1][0], hist[i][1]-hist[i-1][1])
            for i in range(1, len(hist))
        )
        return total_dist > ROAM_DISTANCE_THRESHOLD

    def detect(self, frame):
        now = time.time()
        frame = draw_zones(frame, self.zones)
        persons = self._detect_persons(frame)
        alerts = []

        for (x1, y1, x2, y2, tid) in persons:
            cx, cy = get_centroid((x1, y1, x2, y2))
            bbox   = (x1, y1, x2, y2)

            # Track movement history (keep last 60 positions)
            hist = self.track_history.setdefault(tid, [])
            hist.append((cx, cy))
            if len(hist) > 60:
                hist.pop(0)

            in_zone = False
            zone_name = None
            for zname, zinfo in self.zones.items():
                if point_in_polygon((cx, cy), zinfo["polygon"]):
                    in_zone = True
                    zone_name = zname
                    break

            if in_zone:
                if tid not in self.zone_entry_time:
                    self.zone_entry_time[tid] = now

                elapsed = now - self.zone_entry_time[tid]
                roaming = self._is_roaming(tid)

                # Draw timer
                cv2.putText(frame, f"ID{tid}: {elapsed:.1f}s in {zone_name}",
                            (x1, y2 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 165, 255), 1)

                threshold = self.zones.get(zone_name, {}).get("alert_after_seconds", LOITER_TIME_THRESHOLD)
                if elapsed >= threshold:
                    self.alert_active[tid] = True
                    label = "ROAMING" if roaming else "LOITERING"
                    log_alert(label, tid, f"{elapsed:.1f}s in {zone_name}")
                    frame = draw_alert(frame, label, bbox, "LOITERING",
                                       f"{elapsed:.0f}s | {zone_name}")
                    alerts.append({"type": label, "track_id": tid,
                                   "elapsed": elapsed, "zone": zone_name})
                else:
                    # Yellow warning approaching threshold
                    warn_color = (0, 255, 255)
                    cv2.rectangle(frame, (x1,y1),(x2,y2), warn_color, 2)
                    cv2.putText(frame, f"ID:{tid} WARNING",
                                (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, warn_color, 2)
            else:
                # Person left zone — reset their timer
                self.zone_entry_time.pop(tid, None)
                self.alert_active.pop(tid, None)
                cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # Draw centroid trail
            for i in range(1, len(hist)):
                alpha = int(255 * i / len(hist))
                cv2.line(frame, hist[i-1], hist[i], (alpha, alpha, 0), 1)

        return frame, alerts


# ── Standalone runner ──────────────────────────────────────────────────────────
def run(source=0):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    detector = LoiteringDetector(fps=fps)
    print("[Loitering Detection] Press Q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, alerts = detector.detect(frame)
        cv2.imshow("Loitering Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    run(src)
