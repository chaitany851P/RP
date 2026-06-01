import cv2
import time
from datetime import datetime

ALERT_COLORS = {
    "FALL":        (0, 0, 255),      # Red
    "LOITERING":   (0, 165, 255),    # Orange
    "PROLONGED":   (0, 255, 255),    # Yellow
    "UNAUTHORIZED":(180, 0, 255),    # Purple
    "UNSAFE_POSTURE":(0, 80, 255),   # Deep orange-red
}

def draw_alert(frame, label, bbox, alert_type="FALL", extra_text=""):
    color = ALERT_COLORS.get(alert_type, (0, 0, 255))
    x1, y1, x2, y2 = bbox

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Label background
    text = f"[ALERT] {label}"
    if extra_text:
        text += f" | {extra_text}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return frame


def log_alert(alert_type, track_id=None, extra=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] ALERT: {alert_type}"
    if track_id is not None:
        msg += f" | Track ID: {track_id}"
    if extra:
        msg += f" | {extra}"
    print(msg)
