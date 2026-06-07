"""
Fall Detection - YOLOv8 Training Script
Dataset: Roboflow Fall Detection (fall-detection-ca3o8)
Run: python train.py
"""

from ultralytics import YOLO
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_YAML   = os.path.join(os.path.dirname(__file__), "data.yaml")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "runs")

# ── Training Config ───────────────────────────────────────────────────────────
EPOCHS      = 50       # increase to 100 for better accuracy
IMG_SIZE    = 640
BATCH_SIZE  = 8        # lower to 4 if you get out-of-memory error
MODEL       = "yolov8n.pt"  # nano = fastest; use yolov8s.pt for better accuracy

def train():
    print("\n====================================")
    print("  Fall Detection - YOLOv8 Training  ")
    print("====================================\n")
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Epochs  : {EPOCHS}")
    print(f"  ImgSize : {IMG_SIZE}")
    print(f"  Batch   : {BATCH_SIZE}")
    print(f"  Model   : {MODEL}")
    print("\n  Training started... (this takes 20-40 mins)\n")

    # Load pretrained YOLOv8 model
    model = YOLO(MODEL)

    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        name="fall_detection_v1",
        project=OUTPUT_DIR,
        patience=15,          # stop early if no improvement for 15 epochs
        save=True,
        plots=True,           # saves training graphs
        verbose=True,
    )

    print("\n====================================")
    print("  Training Complete!")
    print(f"  Best model saved at:")
    print(f"  {OUTPUT_DIR}/fall_detection_v1/weights/best.pt")
    print("====================================\n")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("  Running evaluation on test set...\n")
    metrics = model.val(split="test")
    print(f"\n  Precision : {metrics.box.mp:.3f}")
    print(f"  Recall    : {metrics.box.mr:.3f}")
    print(f"  mAP@50    : {metrics.box.map50:.3f}")
    print(f"  mAP@50-95 : {metrics.box.map:.3f}")
    print("\n  Done! Use best.pt in your pipeline.\n")


if __name__ == "__main__":
    train()