"""Evaluate all trained YOLOv8 variants on the BCCD test set and export a comparison table."""

from pathlib import Path

import pandas as pd

from ultralytics import YOLO

ROOT = Path(r"D:\Project\基于YOLO5细胞检测实战")
DATA_YAML = ROOT / "yolov8" / "bccd.yaml"
RUNS_DIR = ROOT / "yolov8" / "runs"


def evaluate(variants=None, scale="s"):
    """Evaluate each variant and save metrics to CSV.

    Args:
        variants (list[str] | None): Variant names to evaluate.
        scale (str): Model scale used during training.
    """
    variants = variants or ["baseline", "cbam", "se", "eca"]
    records = []

    for variant in variants:
        weights = RUNS_DIR / f"yolov8{scale}_{variant}" / "weights" / "best.pt"
        if not weights.exists():
            print(f"[SKIP] weights not found: {weights}")
            continue

        model = YOLO(str(weights))
        metrics = model.val(data=str(DATA_YAML), split="test")

        p, r = metrics.box.mp, metrics.box.mr
        f1 = 2 * p * r / (p + r + 1e-6)

        records.append(
            {
                "variant": variant,
                "mAP50": round(metrics.box.map50, 4),
                "mAP50-95": round(metrics.box.map, 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "F1": round(f1, 4),
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        print(df.to_string(index=False))
        out_path = RUNS_DIR / "comparison.csv"
        df.to_csv(out_path, index=False)
        print(f"\nComparison saved to: {out_path}")
    else:
        print("No models were evaluated.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 variants on BCCD test set")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["baseline", "cbam", "se", "eca"],
        help="Variants to evaluate",
    )
    parser.add_argument("--scale", default="s", help="Model scale")
    args = parser.parse_args()

    evaluate(args.variants, args.scale)
