"""
predict_hand.py – Identify mahjong tiles in an image using a trained YOLO model.

Given a photo of a mahjong hand (or any image containing mahjong tiles), this
script detects every tile, prints the hand composition, and saves an annotated
copy of the image.

Usage
-----
    python predict_hand.py <image_path> [options]

Examples
--------
    # Basic usage (auto-finds the latest trained weights):
    python predict_hand.py hand.jpg

    # Specify weights explicitly:
    python predict_hand.py hand.jpg --model runs/mahjong/train/weights/best.pt

    # Adjust confidence threshold and save annotated image to a custom path:
    python predict_hand.py hand.jpg --conf 0.4 --output result.jpg

    # Skip saving the annotated image:
    python predict_hand.py hand.jpg --no-save
"""

import argparse
import glob
import os
import sys
from collections import Counter


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Class names in 0-indexed order (must match prepare_yolo_dataset.py)
CLASSES = [
    "dots-1", "dots-2", "dots-3", "dots-4", "dots-5",
    "dots-6", "dots-7", "dots-8", "dots-9",
    "bamboo-1", "bamboo-2", "bamboo-3", "bamboo-4", "bamboo-5",
    "bamboo-6", "bamboo-7", "bamboo-8", "bamboo-9",
    "characters-1", "characters-2", "characters-3", "characters-4",
    "characters-5", "characters-6", "characters-7", "characters-8",
    "characters-9",
    "honors-east", "honors-south", "honors-west", "honors-north",
    "honors-red", "honors-green", "honors-white",
    "bonus-spring", "bonus-summer", "bonus-autumn", "bonus-winter",
    "bonus-plum", "bonus-orchid", "bonus-chrysanthemum", "bonus-bamboo",
]

# Human-friendly display names (Chinese + English)
DISPLAY_NAMES = {
    "dots-1": "1筒 (dots-1)",
    "dots-2": "2筒 (dots-2)",
    "dots-3": "3筒 (dots-3)",
    "dots-4": "4筒 (dots-4)",
    "dots-5": "5筒 (dots-5)",
    "dots-6": "6筒 (dots-6)",
    "dots-7": "7筒 (dots-7)",
    "dots-8": "8筒 (dots-8)",
    "dots-9": "9筒 (dots-9)",
    "bamboo-1": "1条 (bamboo-1)",
    "bamboo-2": "2条 (bamboo-2)",
    "bamboo-3": "3条 (bamboo-3)",
    "bamboo-4": "4条 (bamboo-4)",
    "bamboo-5": "5条 (bamboo-5)",
    "bamboo-6": "6条 (bamboo-6)",
    "bamboo-7": "7条 (bamboo-7)",
    "bamboo-8": "8条 (bamboo-8)",
    "bamboo-9": "9条 (bamboo-9)",
    "characters-1": "1万 (characters-1)",
    "characters-2": "2万 (characters-2)",
    "characters-3": "3万 (characters-3)",
    "characters-4": "4万 (characters-4)",
    "characters-5": "5万 (characters-5)",
    "characters-6": "6万 (characters-6)",
    "characters-7": "7万 (characters-7)",
    "characters-8": "8万 (characters-8)",
    "characters-9": "9万 (characters-9)",
    "honors-east": "东风 (honors-east)",
    "honors-south": "南风 (honors-south)",
    "honors-west": "西风 (honors-west)",
    "honors-north": "北风 (honors-north)",
    "honors-red": "中 (honors-red)",
    "honors-green": "发 (honors-green)",
    "honors-white": "白板 (honors-white)",
    "bonus-spring": "春 (bonus-spring)",
    "bonus-summer": "夏 (bonus-summer)",
    "bonus-autumn": "秋 (bonus-autumn)",
    "bonus-winter": "冬 (bonus-winter)",
    "bonus-plum": "梅 (bonus-plum)",
    "bonus-orchid": "兰 (bonus-orchid)",
    "bonus-chrysanthemum": "菊 (bonus-chrysanthemum)",
    "bonus-bamboo": "竹 (bonus-bamboo)",
}


def find_latest_weights():
    """Search common training output directories for the most recent best.pt."""
    patterns = [
        os.path.join(REPO_ROOT, "runs", "mahjong", "*", "weights", "best.pt"),
        os.path.join(REPO_ROOT, "runs", "detect", "*", "weights", "best.pt"),
        os.path.join(REPO_ROOT, "runs", "*", "weights", "best.pt"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        return None
    # Return the most recently modified
    return max(candidates, key=os.path.getmtime)


def parse_args():
    p = argparse.ArgumentParser(
        description="Identify mahjong tiles in an image using a trained YOLO model"
    )
    p.add_argument("image", help="Path to the input image")
    p.add_argument("--model", default=None,
                   help="Path to trained YOLO weights (.pt file). "
                        "If omitted, the script searches runs/ for the latest best.pt.")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Minimum confidence threshold (default: 0.25)")
    p.add_argument("--iou", type=float, default=0.45,
                   help="IoU threshold for NMS (default: 0.45)")
    p.add_argument("--output", default=None,
                   help="Path to save the annotated image (default: <image>_result.jpg)")
    p.add_argument("--no-save", action="store_true",
                   help="Do not save the annotated image")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Inference image size – should match training imgsz (default: 320)")
    return p.parse_args()


def print_hand(detections):
    """Print a formatted hand summary from the list of (class_name, confidence) tuples."""
    if not detections:
        print("\nNo tiles detected.  Try lowering --conf or using a higher-quality image.")
        return

    print(f"\n{'=' * 50}")
    print(f"  Detected {len(detections)} tile(s) in the image:")
    print(f"{'=' * 50}")

    # Sort detections left-to-right (by x-center) if positions are available,
    # otherwise just sort by class name for a clean display.
    counts = Counter(name for name, _ in detections)
    sorted_tiles = sorted(detections, key=lambda d: d[0])

    for i, (cls_name, conf) in enumerate(sorted_tiles, 1):
        display = DISPLAY_NAMES.get(cls_name, cls_name)
        print(f"  {i:2d}. {display:<32s}  (conf: {conf:.2f})")

    print(f"\n  Hand summary (sorted):")
    for cls_name, count in sorted(counts.items()):
        display = DISPLAY_NAMES.get(cls_name, cls_name)
        print(f"       {display} × {count}")
    print("=" * 50)


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("Ultralytics is not installed.  Run:  pip install ultralytics")

    # Resolve model weights
    weights = args.model
    if weights is None:
        weights = find_latest_weights()
        if weights is None:
            sys.exit(
                "No trained weights found.  Run train.py first, or pass --model <path>."
            )
        print(f"Using weights: {weights}")

    if not os.path.isfile(weights):
        sys.exit(f"Weights file not found: {weights}")

    model = YOLO(weights)

    print(f"Running inference on: {args.image}")
    results = model.predict(
        source=args.image,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
    )

    # Collect detections
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for cls_idx, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            cls_name = CLASSES[int(cls_idx)]
            detections.append((cls_name, float(conf)))

    print_hand(detections)

    # Save annotated image
    if not args.no_save:
        out_path = args.output
        if out_path is None:
            base, _ = os.path.splitext(args.image)
            out_path = base + "_result.jpg"
        for result in results:
            annotated = result.plot()
            try:
                import cv2
                cv2.imwrite(out_path, annotated)
                print(f"\nAnnotated image saved to: {out_path}")
            except Exception as exc:
                print(f"Warning: could not save annotated image – {exc}")


if __name__ == "__main__":
    main()
