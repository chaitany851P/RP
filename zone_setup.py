"""
Zone Setup Tool — Run this ONCE per camera to define zones and rules.
Instructions:
  1. Run: python zone_setup.py video.mp4  (or 0 for webcam)
  2. Press N to start a new zone
  3. Click points on the frame to draw the zone polygon
  4. Press ENTER to finish the zone and set its rules
  5. Repeat for all zones
  6. Press S to save and exit
  7. config.json is generated automatically
"""

import cv2
import json
import os
import sys
import numpy as np

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

AVAILABLE_ACTIVITIES = [
    "fall", "smoking", "eating", "drinking",
    "phone", "sleeping", "sitting", "loitering",
    "unauthorized_entry", "unsafe_posture"
]

ZONE_COLORS = [
    (0, 0, 255),    # Red
    (0, 165, 255),  # Orange
    (0, 255, 255),  # Yellow
    (255, 0, 0),    # Blue
    (180, 0, 255),  # Purple
    (0, 255, 0),    # Green
]

class ZoneSetupTool:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        ret, self.base_frame = self.cap.read()
        if not ret:
            print("[ERROR] Could not read video source.")
            sys.exit(1)

        self.zones = {}
        self.current_points = []
        self.drawing = False
        self.color_idx = 0
        self.frame = self.base_frame.copy()

    def _draw_instructions(self, frame):
        instructions = [
            "N = New zone",
            "Click = Add point",
            "ENTER = Finish zone",
            "U = Undo last point",
            "S = Save & exit",
            "Q = Quit without saving",
        ]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (220, 20 + len(instructions)*22), (20,20,20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        for i, text in enumerate(instructions):
            cv2.putText(frame, text, (8, 18 + i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,200,200), 1)
        return frame

    def _draw_zones(self, frame):
        for i, (name, info) in enumerate(self.zones.items()):
            pts = np.array(info["polygon"], dtype=np.int32)
            color = ZONE_COLORS[i % len(ZONE_COLORS)]
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [pts], True, color, 2)
            cx = int(np.mean([p[0] for p in info["polygon"]]))
            cy = int(np.mean([p[1] for p in info["polygon"]]))
            cv2.putText(frame, name, (cx-20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def _draw_current(self, frame):
        if not self.current_points:
            return frame
        color = ZONE_COLORS[self.color_idx % len(ZONE_COLORS)]
        for pt in self.current_points:
            cv2.circle(frame, pt, 5, color, -1)
        for i in range(1, len(self.current_points)):
            cv2.line(frame, self.current_points[i-1], self.current_points[i], color, 2)
        if len(self.current_points) > 2:
            cv2.line(frame, self.current_points[-1], self.current_points[0], color, 1)
        pts_text = f"Points: {len(self.current_points)} | Press ENTER to finish"
        cv2.putText(frame, pts_text, (10, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.drawing:
            self.current_points.append((x, y))

    def _get_zone_rules(self, zone_name):
        print(f"\n{'='*50}")
        print(f"Setting rules for zone: {zone_name}")
        print(f"{'='*50}")
        print("\nAvailable activities:")
        for i, act in enumerate(AVAILABLE_ACTIVITIES):
            print(f"  {i+1}. {act}")

        print("\nEnter FORBIDDEN activity numbers (comma separated)")
        print("Example: 1,3,5  OR press ENTER for all forbidden")
        forbidden_input = input("Forbidden: ").strip()

        if forbidden_input == "":
            forbidden = AVAILABLE_ACTIVITIES.copy()
        else:
            try:
                indices = [int(x.strip())-1 for x in forbidden_input.split(",")]
                forbidden = [AVAILABLE_ACTIVITIES[i] for i in indices if 0 <= i < len(AVAILABLE_ACTIVITIES)]
            except:
                forbidden = AVAILABLE_ACTIVITIES.copy()

        print(f"\nEnter ALLOWED activity numbers (comma separated)")
        print("Example: 2,4  OR press ENTER for none")
        allowed_input = input("Allowed: ").strip()

        if allowed_input == "":
            allowed = []
        else:
            try:
                indices = [int(x.strip())-1 for x in allowed_input.split(",")]
                allowed = [AVAILABLE_ACTIVITIES[i] for i in indices if 0 <= i < len(AVAILABLE_ACTIVITIES)]
                # Remove allowed from forbidden
                forbidden = [f for f in forbidden if f not in allowed]
            except:
                allowed = []

        print("\nAlert after how many seconds? (0 = immediate)")
        print("Examples: 0 for unauthorized entry, 10 for loitering, 15 for prolonged exposure")
        try:
            alert_seconds = int(input("Seconds: ").strip())
        except:
            alert_seconds = 0

        print("\nIs this a MACHINE zone? (y/n)")
        is_machine = input("Machine zone: ").strip().lower() == "y"

        print(f"\nZone '{zone_name}' configured:")
        print(f"  Forbidden: {forbidden}")
        print(f"  Allowed:   {allowed}")
        print(f"  Alert after: {alert_seconds}s")
        print(f"  Machine zone: {is_machine}")

        return {
            "polygon": self.current_points,
            "forbidden": forbidden,
            "allowed": allowed,
            "alert_after_seconds": alert_seconds,
            "is_machine_zone": is_machine,
            "color": ZONE_COLORS[self.color_idx % len(ZONE_COLORS)]
        }

    def run(self):
        cv2.namedWindow("Zone Setup")
        cv2.setMouseCallback("Zone Setup", self._mouse_callback)

        print("\n" + "="*50)
        print("  ZONE SETUP TOOL")
        print("="*50)
        print("Press N to start drawing a new zone")
        print("Press S to save all zones and exit")
        print("="*50 + "\n")

        while True:
            self.frame = self.base_frame.copy()
            self.frame = self._draw_zones(self.frame)
            self.frame = self._draw_current(self.frame)
            self.frame = self._draw_instructions(self.frame)

            # Zone count
            cv2.putText(self.frame, f"Zones defined: {len(self.zones)}",
                        (self.frame.shape[1]-180, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

            cv2.imshow("Zone Setup", self.frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("n") or key == ord("N"):
                if self.drawing:
                    print("Already drawing — press ENTER to finish current zone first")
                else:
                    zone_name = input("\nEnter zone name (e.g. MACHINE_1, OUTSIDE_GATE, FACTORY_FLOOR): ").strip().upper()
                    if not zone_name:
                        zone_name = f"ZONE_{len(self.zones)+1}"
                    self.drawing = True
                    self.current_points = []
                    print(f"Drawing zone '{zone_name}' — click points on the frame")
                    print("Press ENTER when done")
                    # Store name for when enter is pressed
                    self._pending_name = zone_name

            elif key == 13:  # ENTER
                if self.drawing and len(self.current_points) >= 3:
                    self.drawing = False
                    zone_info = self._get_zone_rules(self._pending_name)
                    self.zones[self._pending_name] = zone_info
                    self.color_idx += 1
                    self.current_points = []
                    print(f"\nZone '{self._pending_name}' saved! Press N for next zone or S to save all.")
                elif self.drawing:
                    print("Need at least 3 points to define a zone!")

            elif key == ord("u") or key == ord("U"):
                if self.current_points:
                    self.current_points.pop()
                    print(f"Undone. Points remaining: {len(self.current_points)}")

            elif key == ord("s") or key == ord("S"):
                if self.zones:
                    self._save_config()
                    print(f"\nSaved {len(self.zones)} zones to config.json")
                    break
                else:
                    print("No zones defined yet!")

            elif key == ord("q") or key == ord("Q"):
                print("Exiting without saving.")
                break

        cv2.destroyAllWindows()
        self.cap.release()

    def _save_config(self):
        # Convert colors to list for JSON serialization
        config = {"zones": {}}
        for name, info in self.zones.items():
            config["zones"][name] = {
                "polygon": info["polygon"],
                "forbidden": info["forbidden"],
                "allowed": info["allowed"],
                "alert_after_seconds": info["alert_after_seconds"],
                "is_machine_zone": info["is_machine_zone"],
                "color": list(info["color"])
            }
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\nConfig saved to: {CONFIG_PATH}")
        print("\nGenerated config:")
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else 0
    try:
        src = int(src)
    except:
        pass
    tool = ZoneSetupTool(src)
    tool.run()
