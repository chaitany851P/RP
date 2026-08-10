"""
BRANCH: feature/unauthorized-entry-detection
Detects: Unauthorized entry into danger / work zones
Method:  YOLOv8 tracking + strict danger zone polygons + immediate entry alert
         Key difference from loitering: alert fires INSTANTLY on entry, not after a timer
"""

import cv2
import numpy as np
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.alert import draw_alert, log_alert
from utils.zone import point_in_polygon, draw_zones, get_centroid

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
CONF_THRESHOLD       = 0.4
GRACE_PERIOD_FRAMES  = 3    # frames inside zone before confirmed entry (avoid flicker)

# Danger zones — immediate entry forbidden
DANGER_ZONES = {
}


class UnauthorizedEntryDetector:
    def __init__(self, danger_zones=None):
        self.zones = danger_zones or DANGER_ZONES
        self.model = YOLO("yolov8n.pt") if YOLO_AVAILABLE else None

        self.inside_frames  = {}   # track_id → {zone_name: frame_count}
        self.entry_time     = {}   # track_id → {zone_name: timestamp}
        self.alerted        = {}   # track_id → set of zone names already alerted

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
        now     = time.time()
        alerts  = []

        # Draw danger zones with red fill
        frame = draw_zones(frame, self.zones)

        persons = self._detect_persons(frame)

        for (x1, y1, x2, y2, tid) in persons:
            cx, cy = get_centroid((x1, y1, x2, y2))
            bbox   = (x1, y1, x2, y2)

            self.inside_frames.setdefault(tid, {})
            self.entry_time.setdefault(tid, {})
            self.alerted.setdefault(tid, set())

            any_zone_alert = False

            for zname, zinfo in self.zones.items():
                if point_in_polygon((cx, cy), zinfo["polygon"]):
                    # Increment grace-period counter
                    self.inside_frames[tid][zname] = \
                        self.inside_frames[tid].get(zname, 0) + 1

                    # Record entry time
                    if zname not in self.entry_time[tid]:
                        self.entry_time[tid][zname] = now

                    elapsed = now - self.entry_time[tid][zname]

                    if self.inside_frames[tid][zname] >= GRACE_PERIOD_FRAMES:
                        any_zone_alert = True
                        log_alert("UNAUTHORIZED_ENTRY", tid,
                                  f"zone={zname} duration={elapsed:.1f}s")
                        frame = draw_alert(frame,
                                           f"UNAUTHORIZED ENTRY: {zname}",
                                           bbox, "UNAUTHORIZED",
                                           f"{elapsed:.0f}s inside")
                        alerts.append({
                            "type": "UNAUTHORIZED_ENTRY",
                            "track_id": tid,
                            "zone": zname,
                            "elapsed": elapsed
                        })

                    # Overlay flashing red border when in danger zone
                    flash = int(time.time() * 3) % 2
                    if flash:
                        cv2.rectangle(frame, (x1-3,y1-3),(x2+3,y2+3),
                                      (0,0,255), 4)
                else:
                    # Reset when person leaves zone
                    self.inside_frames[tid].pop(zname, None)
                    self.entry_time[tid].pop(zname, None)
                    self.alerted[tid].discard(zname)

            if not any_zone_alert:
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame, f"ID:{tid}", (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Draw zone labels prominently
        for zname, zinfo in self.zones.items():
            pts = np.array(zinfo["polygon"])
            cx  = int(pts[:,0].mean())
            cy  = int(pts[:,1].mean())
            cv2.putText(frame, f"⚠ {zname}", (cx-50, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        return frame, alerts


# ── Standalone runner ──────────────────────────────────────────────────────────
def run(source=0):
    cap = cv2.VideoCapture(source)
    detector = UnauthorizedEntryDetector()
    print("[Unauthorized Entry Detection] Press Q to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame, alerts = detector.detect(frame)
        cv2.imshow("Unauthorized Entry Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    run(src)
