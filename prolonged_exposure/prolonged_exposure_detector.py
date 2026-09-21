"""
BRANCH: feature/prolonged-exposure-detection
Detects: Person standing near a machine for too long (unsafe prolonged exposure)
Method:  YOLOv8 tracking + machine proximity zones + per-person time counter
         Separate from loitering — this is about MACHINE PROXIMITY, not restricted zones
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

# ── Config ────────────────────────────────────────────────────────────────────
EXPOSURE_THRESHOLD_SEC  = 15.0   # seconds near machine before alert
WARNING_THRESHOLD_SEC   = 8.0    # seconds before pre-alert warning
CONF_THRESHOLD          = 0.4

# Machine proximity zones — define one polygon per machine
MACHINE_ZONES = {
}


class ProlongedExposureDetector:
    def __init__(self, machine_zones=None):
        if machine_zones is not None:
            self.zones = machine_zones
        else:
            loaded = load_zones()
            machine_only = {k: v for k, v in loaded.items() if v.get("is_machine_zone", False)}
            self.zones = machine_only if machine_only else (loaded if loaded else MACHINE_ZONES)
        self.model = YOLO("yolov8n.pt") if YOLO_AVAILABLE else None

        # track_id → {zone_name: entry_timestamp}
        self.exposure_timers = {}
        # track_id → {zone_name: total_accumulated_seconds}
        self.accumulated     = {}

    def _detect_persons(self, frame):
        if self.model is None:
            return []
        results = self.model.track(frame, persist=True,
                                   conf=CONF_THRESHOLD, classes=[0], verbose=False)
        persons = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id[0])
                persons.append((x1, y1, x2, y2, tid))
        return persons

    def detect(self, frame):
        now = time.time()
        alerts = []

        # Draw machine zones
        frame = draw_zones(frame, self.zones)
        persons = self._detect_persons(frame)

        for (x1, y1, x2, y2, tid) in persons:
            cx, cy = get_centroid((x1, y1, x2, y2))
            bbox   = (x1, y1, x2, y2)

            # Initialize state dicts for new track IDs
            if tid not in self.exposure_timers:
                self.exposure_timers[tid] = {}
            if tid not in self.accumulated:
                self.accumulated[tid] = {}

            alert_triggered = False

            for zname, zinfo in self.zones.items():
                max_safe = zinfo.get("max_safe_seconds", zinfo.get("alert_after_seconds", EXPOSURE_THRESHOLD_SEC))

                if point_in_polygon((cx, cy), zinfo["polygon"]):
                    # Start timer if not already started
                    if zname not in self.exposure_timers[tid]:
                        self.exposure_timers[tid][zname] = now

                    session_elapsed = now - self.exposure_timers[tid][zname]
                    prev_accum      = self.accumulated[tid].get(zname, 0)
                    total_exposure  = prev_accum + session_elapsed

                    # Progress bar
                    bar_w = x2 - x1
                    fill  = min(int(bar_w * total_exposure / max_safe), bar_w)
                    bar_color = (0, 255, 0) if total_exposure < WARNING_THRESHOLD_SEC else \
                                (0, 165, 255) if total_exposure < max_safe else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y2+4), (x2, y2+12), (50,50,50), -1)
                    cv2.rectangle(frame, (x1, y2+4), (x1+fill, y2+12), bar_color, -1)
                    cv2.putText(frame, f"ID{tid} {total_exposure:.0f}s / {max_safe}s",
                                (x1, y2+26), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, bar_color, 1)

                    if total_exposure >= max_safe:
                        alert_triggered = True
                        log_alert("PROLONGED_EXPOSURE", tid,
                                  f"{total_exposure:.1f}s near {zname}")
                        frame = draw_alert(frame,
                                           f"PROLONGED EXPOSURE near {zname}",
                                           bbox, "PROLONGED",
                                           f"{total_exposure:.0f}s")
                        alerts.append({"type": "PROLONGED_EXPOSURE",
                                       "track_id": tid, "zone": zname,
                                       "total_sec": total_exposure})
                    elif total_exposure >= WARNING_THRESHOLD_SEC:
                        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,165,255), 2)
                        cv2.putText(frame, f"ID:{tid} APPROACHING LIMIT",
                                    (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, (0,165,255), 2)
                    else:
                        cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)

                else:
                    # Person left zone → accumulate time, reset session timer
                    if zname in self.exposure_timers[tid]:
                        session = now - self.exposure_timers[tid].pop(zname)
                        self.accumulated[tid][zname] = \
                            self.accumulated[tid].get(zname, 0) + session

            if not alert_triggered:
                pass  # box already drawn in loop above

        return frame, alerts


# ── Standalone runner ──────────────────────────────────────────────────────────
def run(source=0):
    cap = cv2.VideoCapture(source)
    detector = ProlongedExposureDetector()
    print("[Prolonged Exposure Detection] Press Q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, _ = detector.detect(frame)
        cv2.imshow("Prolonged Exposure Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    run(src)
