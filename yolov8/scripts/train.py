"""Unified training script for YOLOv8 baseline and attention variants on BCCD."""

import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO

ROOT = Path(r"D:\Project\基于YOLO5细胞检测实战")
CFG_DIR = ROOT / "yolov8" / "cfg"
DATA_YAML = ROOT / "yolov8" / "bccd.yaml"
RUNS_DIR = ROOT / "yolov8" / "runs"


def train(variant="baseline", scale="s", epochs=150, batch=None, imgsz=416):
    """Train a YOLOv8 variant on the BCCD dataset.

    Args:
        variant (str): One of 'baseline', 'cbam', 'se', 'eca'.
        scale (str): One of 'n', 's', 'm'.
        epochs (int): Number of training epochs.
        batch (int | None): Batch size; auto-selected if None.
        imgsz (int): Input image size.
    """
    if batch is None:
        batch = 16 if scale in ("n", "s") else 8

    suffix = "" if variant == "baseline" else f"{variant}-"
    cfg_name = f"yolov8{scale}-{suffix}bccd.yaml"
    pretrained = f"yolov8{scale}.pt"

    cfg_path = CFG_DIR / cfg_name
    model = YOLO(str(cfg_path))
    model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=0,
        workers=4,
        pretrained=pretrained,
        project=str(RUNS_DIR),
        name=f"yolov8{scale}_{variant}",
        patience=50,
        save=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 variants on BCCD")
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "cbam", "se", "eca"],
        help="Attention variant to train",
    )
    parser.add_argument(
        "--scale", default="s", choices=["n", "s", "m"], help="YOLOv8 model scale"
    )
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size")
    args = parser.parse_args()

    train(args.variant, args.scale, args.epochs, args.batch, args.imgsz)
