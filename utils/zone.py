import cv2
import numpy as np


def point_in_polygon(point, polygon):
    """Check if a (cx, cy) point is inside a polygon [(x,y), ...]."""
    return cv2.pointPolygonTest(
        np.array(polygon, dtype=np.int32), point, False
    ) >= 0


def draw_zones(frame, zones: dict):
    """
    zones = {
        "MACHINE_A": {"polygon": [(x,y),...], "color": (0,255,0)},
        ...
    }
    """
    overlay = frame.copy()

    # Step 1: fill polygons on overlay
    for name, info in zones.items():
        pts   = np.array(info["polygon"], dtype=np.int32)
        color = info.get("color", (0, 255, 0))
        cv2.fillPoly(overlay, [pts], color)

    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

    # Step 2: draw borders + labels (after blend so labels stay sharp)
    used_label_rects = []   # track placed label boxes to avoid overlap

    for name, info in zones.items():
        pts   = np.array(info["polygon"], dtype=np.int32)
        color = info.get("color", (0, 255, 0))

        # Border
        cv2.polylines(frame, [pts], True, color, 2)

        # Label position — top-left corner of polygon + small offset
        xs = [p[0] for p in info["polygon"]]
        ys = [p[1] for p in info["polygon"]]
        lx = int(min(xs)) + 6
        ly = int(min(ys)) + 20

        # Nudge down if overlapping a previously placed label
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness  = 1
        (tw, th), _ = cv2.getTextSize(name, font, font_scale, thickness)
        pad = 4

        for _ in range(10):   # try up to 10 nudges
            label_rect = (lx - pad, ly - th - pad, lx + tw + pad, ly + pad)
            overlap = any(
                not (label_rect[2] < r[0] or label_rect[0] > r[2] or
                     label_rect[3] < r[1] or label_rect[1] > r[3])
                for r in used_label_rects
            )
            if not overlap:
                break
            ly += th + pad * 3   # nudge downward

        used_label_rects.append((lx - pad, ly - th - pad,
                                  lx + tw + pad, ly + pad))

        # Dark background pill for readability
        cv2.rectangle(frame,
                      (lx - pad,     ly - th - pad),
                      (lx + tw + pad, ly + pad),
                      (20, 20, 20), -1)
        cv2.rectangle(frame,
                      (lx - pad,     ly - th - pad),
                      (lx + tw + pad, ly + pad),
                      color, 1)
        cv2.putText(frame, name, (lx, ly), font, font_scale,
                    color, thickness, cv2.LINE_AA)

    return frame


def get_centroid(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))
