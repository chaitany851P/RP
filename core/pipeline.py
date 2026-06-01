"""
BRANCH: feature/activity-detection-core
Master pipeline — runs all 5 detectors on a single video stream
Toggle individual detectors via CONFIG flags
"""

import cv2
import sys, os, time
import numpy as np

sys.path.append(os.path.dirname(__file__))

from fall_detection.fall_detector           import FallDetector
from loitering_detection.loitering_detector import LoiteringDetector
from prolonged_exposure.prolonged_exposure_detector import ProlongedExposureDetector
from unauthorized_entry.unauthorized_entry_detector import UnauthorizedEntryDetector
from unsafe_posture.unsafe_posture_detector import UnsafePostureDetector

# ── Toggle detectors ──────────────────────────────────────────────────────────
CONFIG = {
    "fall":          True,
    "loitering":     True,
    "prolonged":     True,
    "unauthorized":  True,
    "posture":       True,
}

ALERT_BANNER_DURATION = 3.0   # seconds to show top banner per alert


class IndustrialSafetyPipeline:
    def __init__(self, config=None):
        self.config = config or CONFIG
        print("[Pipeline] Initializing detectors...")

        self.detectors = {}
        if self.config["fall"]:
            self.detectors["fall"]         = FallDetector()
        if self.config["loitering"]:
            self.detectors["loitering"]    = LoiteringDetector()
        if self.config["prolonged"]:
            self.detectors["prolonged"]    = ProlongedExposureDetector()
        if self.config["unauthorized"]:
            self.detectors["unauthorized"] = UnauthorizedEntryDetector()
        if self.config["posture"]:
            self.detectors["posture"]      = UnsafePostureDetector()

        self.active_alerts = []   # list of (message, expire_time, color)
        print("[Pipeline] Ready.")

    def _add_alert(self, msg, color=(0, 0, 255)):
        expire = time.time() + ALERT_BANNER_DURATION
        self.active_alerts.append((msg, expire, color))
        # Keep only fresh alerts
        self.active_alerts = [(m, e, c) for m, e, c in self.active_alerts
                              if e > time.time()]

    def _draw_status_panel(self, frame):
        h, w = frame.shape[:2]

        # Top status bar
        cv2.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Industrial Safety Monitor | {ts}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Active alert banners
        now = time.time()
        self.active_alerts = [(m, e, c) for m, e, c in self.active_alerts if e > now]
        for i, (msg, expire, color) in enumerate(self.active_alerts[-4:]):
            y = 60 + i * 28
            cv2.rectangle(frame, (0, y-20), (w, y+6), color, -1)
            cv2.putText(frame, f"⚠ {msg}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Bottom legend
        legend_items = [
            ("FALL",        (0,   0,   255)),
            ("LOITERING",   (0,   165, 255)),
            ("PROLONGED",   (0,   255, 255)),
            ("UNAUTHORIZED",(180, 0,   255)),
            ("POSTURE",     (0,   80,  255)),
        ]
        lx = 8
        for label, color in legend_items:
            cv2.rectangle(frame, (lx, h-22), (lx+14, h-8), color, -1)
            cv2.putText(frame, label, (lx+18, h-9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            lx += len(label)*8 + 30

        return frame

    def process_frame(self, frame):
        # Run each detector and collect alerts
        all_alerts = []

        if "fall" in self.detectors:
            frame, fallen, _ = self.detectors["fall"].detect(frame)
            if fallen:
                self._add_alert("PERSON FALLEN / UNCONSCIOUS", (0, 0, 255))
                all_alerts.append("FALL")

        if "loitering" in self.detectors:
            frame, alerts = self.detectors["loitering"].detect(frame)
            for a in alerts:
                self._add_alert(f"{a['type']} ID:{a['track_id']} in {a['zone']}",
                                 (0, 165, 255))
                all_alerts.append(a["type"])

        if "prolonged" in self.detectors:
            frame, alerts = self.detectors["prolonged"].detect(frame)
            for a in alerts:
                self._add_alert(f"PROLONGED EXPOSURE ID:{a['track_id']} {a['total_sec']:.0f}s",
                                 (0, 255, 255))
                all_alerts.append("PROLONGED")

        if "unauthorized" in self.detectors:
            frame, alerts = self.detectors["unauthorized"].detect(frame)
            for a in alerts:
                self._add_alert(f"UNAUTHORIZED ENTRY ID:{a['track_id']} → {a['zone']}",
                                 (180, 0, 255))
                all_alerts.append("UNAUTHORIZED")

        if "posture" in self.detectors:
            frame, alerts = self.detectors["posture"].detect(frame)
            for a in alerts:
                self._add_alert(f"UNSAFE POSTURE: {', '.join(a['issues'][:2])}",
                                 (0, 80, 255))
                all_alerts.append("UNSAFE_POSTURE")

        frame = self._draw_status_panel(frame)
        return frame, all_alerts


# ── Standalone runner ──────────────────────────────────────────────────────────
def run(source=0, output_path=None):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    pipeline = IndustrialSafetyPipeline()
    print("[Pipeline] Running. Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, alerts = pipeline.process_frame(frame)

        cv2.imshow("Industrial Safety Monitor", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run(src, out)
