"""
train.py – End-to-end script to prepare the YOLO dataset and train a model.

Steps performed:
  1. Runs prepare_yolo_dataset.py to create the YOLO-format dataset.
  2. Trains a YOLOv8 (or YOLOv5/v9/v10/v11) model using Ultralytics.
  3. Prints the path to the best trained weights upon completion.

Usage
-----
    python train.py [options]

Examples
--------
    # Quick start – train for 100 epochs with YOLOv8n:
    python train.py

    # Custom settings:
    python train.py --epochs 200 --model yolov8s.pt --imgsz 640 --batch 32

    # Resume an interrupted training run:
    python train.py --resume runs/mahjong/weights/last.pt
"""

import argparse
import os
import subprocess
import sys


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET_DIR = os.path.join(REPO_ROOT, "yolo_dataset")
DEFAULT_DATASET_YAML = os.path.join(DEFAULT_DATASET_DIR, "dataset.yaml")


def parse_args():
    p = argparse.ArgumentParser(
        description="Train a YOLO model on the Mahjong tile dataset"
    )
    p.add_argument("--epochs", type=int, default=100,
                   help="Number of training epochs (default: 100)")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO base model to fine-tune (default: yolov8n.pt)")
    p.add_argument("--imgsz", type=int, default=320,
                   help="Training image size in pixels (default: 320)")
    p.add_argument("--batch", type=int, default=16,
                   help="Batch size (default: 16)")
    p.add_argument("--project", default=os.path.join(REPO_ROOT, "runs", "mahjong"),
                   help="Directory to save training runs (default: runs/mahjong)")
    p.add_argument("--name", default="train",
                   help="Run name inside --project (default: train)")
    p.add_argument("--val-split", type=float, default=0.2,
                   help="Validation fraction for dataset prep (default: 0.2)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for dataset split (default: 42)")
    p.add_argument("--skip-prep", action="store_true",
                   help="Skip dataset preparation if yolo_dataset/ already exists")
    p.add_argument("--resume", metavar="WEIGHTS",
                   help="Resume training from the given .pt weights file")
    p.add_argument("--device", default="",
                   help="Training device: '' (auto), 'cpu', '0' (GPU 0), '0,1' …")
    return p.parse_args()


def prepare_dataset(val_split, seed):
    """Run prepare_yolo_dataset.py to create the YOLO-format dataset."""
    script = os.path.join(REPO_ROOT, "prepare_yolo_dataset.py")
    cmd = [
        sys.executable, script,
        "--val-split", str(val_split),
        "--seed", str(seed),
        "--output", DEFAULT_DATASET_DIR,
    ]
    print("=" * 60)
    print("Step 1 – Preparing YOLO dataset …")
    print("=" * 60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"Dataset preparation failed (exit code {result.returncode}).", file=sys.stderr)
        sys.exit(1)


def run_training(args, yaml_path):
    """Launch Ultralytics YOLO training via the Python API."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "Ultralytics is not installed.  Run:  pip install ultralytics"
        )

    print()
    print("=" * 60)
    print("Step 2 – Training YOLO model …")
    print("=" * 60)

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        train_kwargs = dict(
            data=yaml_path,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
        )
        if args.device:
            train_kwargs["device"] = args.device
        results = model.train(**train_kwargs)

    # Locate the best weights
    best = None
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        candidate = os.path.join(str(save_dir), "weights", "best.pt")
        if os.path.isfile(candidate):
            best = candidate

    print()
    print("=" * 60)
    print("Training complete!")
    if best:
        print(f"  Best weights : {best}")
        print()
        print("To predict your mahjong hand, run:")
        print(f"  python predict_hand.py --model {best} <image_path>")
    print("=" * 60)
    return best


def main():
    args = parse_args()

    if args.resume:
        # Resume mode: skip dataset prep and go straight to training
        run_training(args, yaml_path=DEFAULT_DATASET_YAML)
        return

    # Prepare the dataset unless it already exists and --skip-prep is set
    if args.skip_prep and os.path.isfile(DEFAULT_DATASET_YAML):
        print(f"Skipping dataset prep – using existing {DEFAULT_DATASET_YAML}")
    else:
        prepare_dataset(args.val_split, args.seed)

    run_training(args, yaml_path=DEFAULT_DATASET_YAML)


if __name__ == "__main__":
    main()
