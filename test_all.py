"""
Test runner — checks all 5 detectors can import and initialize correctly.
Run: python test_all.py
"""

import sys
import os
import traceback

sys.path.append(os.path.dirname(__file__))

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"

results = []

def test(name, fn):
    try:
        fn()
        print(f"{PASS} {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL} {name}")
        print(f"       {type(e).__name__}: {e}")
        results.append((name, False, str(e)))

# ── Dependency checks ─────────────────────────────────────────────────────────
print("\n--- Checking Dependencies ---")

def check_cv2():
    import cv2
    print(f"       opencv version: {cv2.__version__}")

def check_mediapipe():
    import mediapipe as mp
    print(f"       mediapipe version: {mp.__version__}")
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

def check_ultralytics():
    from ultralytics import YOLO
    print(f"       ultralytics OK")

def check_numpy():
    import numpy as np
    print(f"       numpy version: {np.__version__}")

test("OpenCV",       check_cv2)
test("MediaPipe",    check_mediapipe)
test("Ultralytics",  check_ultralytics)
test("NumPy",        check_numpy)

# ── Module import checks ──────────────────────────────────────────────────────
print("\n--- Checking Module Imports ---")

def check_utils_alert():
    from utils.alert import draw_alert, log_alert

def check_utils_zone():
    from utils.zone import point_in_polygon, draw_zones, get_centroid

def check_fall():
    from fall_detection.fall_detector import FallDetector

def check_loitering():
    from loitering_detection.loitering_detector import LoiteringDetector

def check_prolonged():
    from prolonged_exposure.prolonged_exposure_detector import ProlongedExposureDetector

def check_unauthorized():
    from unauthorized_entry.unauthorized_entry_detector import UnauthorizedEntryDetector

def check_posture():
    from unsafe_posture.unsafe_posture_detector import UnsafePostureDetector

def check_pipeline():
    from core.pipeline import IndustrialSafetyPipeline

test("utils/alert.py",            check_utils_alert)
test("utils/zone.py",             check_utils_zone)
test("fall_detector.py",          check_fall)
test("loitering_detector.py",     check_loitering)
test("prolonged_exposure.py",     check_prolonged)
test("unauthorized_entry.py",     check_unauthorized)
test("unsafe_posture.py",         check_posture)
test("core/pipeline.py",          check_pipeline)

# ── Detector initialization checks ───────────────────────────────────────────
print("\n--- Checking Detector Initialization ---")

def init_fall():
    from fall_detection.fall_detector import FallDetector
    d = FallDetector()
    print(f"       FallDetector initialized OK")

def init_loitering():
    from loitering_detection.loitering_detector import LoiteringDetector
    d = LoiteringDetector()
    print(f"       LoiteringDetector initialized OK")

def init_prolonged():
    from prolonged_exposure.prolonged_exposure_detector import ProlongedExposureDetector
    d = ProlongedExposureDetector()
    print(f"       ProlongedExposureDetector initialized OK")

def init_unauthorized():
    from unauthorized_entry.unauthorized_entry_detector import UnauthorizedEntryDetector
    d = UnauthorizedEntryDetector()
    print(f"       UnauthorizedEntryDetector initialized OK")

def init_posture():
    from unsafe_posture.unsafe_posture_detector import UnsafePostureDetector
    d = UnsafePostureDetector()
    print(f"       UnsafePostureDetector initialized OK")

def init_pipeline():
    from core.pipeline import IndustrialSafetyPipeline
    p = IndustrialSafetyPipeline()
    print(f"       Full Pipeline initialized OK")

test("Init FallDetector",              init_fall)
test("Init LoiteringDetector",         init_loitering)
test("Init ProlongedExposureDetector", init_prolonged)
test("Init UnauthorizedEntryDetector", init_unauthorized)
test("Init UnsafePostureDetector",     init_posture)
test("Init Full Pipeline",             init_pipeline)

# ── Dummy frame inference check ───────────────────────────────────────────────
print("\n--- Checking Inference on Dummy Frame ---")
import numpy as np
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

def infer_fall():
    from fall_detection.fall_detector import FallDetector
    d = FallDetector()
    frame, fallen, bbox = d.detect(dummy_frame.copy())
    print(f"       fallen={fallen}, bbox={bbox}")

def infer_loitering():
    from loitering_detection.loitering_detector import LoiteringDetector
    d = LoiteringDetector()
    frame, alerts = d.detect(dummy_frame.copy())
    print(f"       alerts={alerts}")

def infer_prolonged():
    from prolonged_exposure.prolonged_exposure_detector import ProlongedExposureDetector
    d = ProlongedExposureDetector()
    frame, alerts = d.detect(dummy_frame.copy())
    print(f"       alerts={alerts}")

def infer_unauthorized():
    from unauthorized_entry.unauthorized_entry_detector import UnauthorizedEntryDetector
    d = UnauthorizedEntryDetector()
    frame, alerts = d.detect(dummy_frame.copy())
    print(f"       alerts={alerts}")

def infer_posture():
    from unsafe_posture.unsafe_posture_detector import UnsafePostureDetector
    d = UnsafePostureDetector()
    frame, alerts = d.detect(dummy_frame.copy())
    print(f"       alerts={alerts}")

def infer_pipeline():
    from core.pipeline import IndustrialSafetyPipeline
    p = IndustrialSafetyPipeline()
    frame, alerts = p.process_frame(dummy_frame.copy())
    print(f"       pipeline alerts={alerts}")

test("Inference: FallDetector",              infer_fall)
test("Inference: LoiteringDetector",         infer_loitering)
test("Inference: ProlongedExposureDetector", infer_prolonged)
test("Inference: UnauthorizedEntryDetector", infer_unauthorized)
test("Inference: UnsafePostureDetector",     infer_posture)
test("Inference: Full Pipeline",             infer_pipeline)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*50)
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"  TOTAL: {passed} passed, {failed} failed out of {len(results)} tests")

if failed:
    print(f"\n  Failed tests:")
    for name, ok, err in results:
        if not ok:
            print(f"    - {name}: {err}")
print("="*50 + "\n")
