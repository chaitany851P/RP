from ultralytics import YOLO
import os

from fall_detection.fall_detector import YOLO_AVAILABLE
from unsafe_posture.unsafe_posture_detector import POSTURE_MODEL_PATH

class UnsafePostureDetector:
    def __init__(self):

        if YOLO_AVAILABLE and os.path.exists(POSTURE_MODEL_PATH):
            self.yolo_posture = YOLO(POSTURE_MODEL_PATH)
            self.use_yolo_posture = True
            print("[PostureDetector] Using trained posture_best.pt ✅")
        else:
            self.yolo_posture = None
            self.use_yolo_posture = False
            print("[PostureDetector] Trained model not found ❌")