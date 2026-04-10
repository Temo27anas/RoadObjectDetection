from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO on a prepared dataset with augmentation settings."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("yolo_dataset/dataset.yaml"),
        help="Path to dataset.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov26m.pt",
        help="YOLO model checkpoint",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help='Device: "0", "0,1", or "cpu"')
    parser.add_argument("--workers", type=int, default=6, help="Dataloader workers")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project output dir")
    parser.add_argument("--name", type=str, default="kitti_aug", help="Run name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    
    # Augmentation controls
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV-H augmentation gain")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV-S augmentation gain")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV-V augmentation gain")
    parser.add_argument("--degrees", type=float, default=5.0, help="Image rotation (+/- degrees)")
    parser.add_argument("--translate", type=float, default=0.1, help="Image translation fraction")
    parser.add_argument("--scale", type=float, default=0.5, help="Image scale gain")
    parser.add_argument("--shear", type=float, default=2.0, help="Image shear (+/- degrees)")
    parser.add_argument("--perspective", type=float, default=0.0005, help="Perspective transform")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Left-right flip probability")
    parser.add_argument("--flipud", type=float, default=0.0, help="Up-down flip probability")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic probability")
    parser.add_argument("--mixup", type=float, default=0.1, help="MixUp probability")
    parser.add_argument("--copy-paste", type=float, default=0.0, help="Copy-paste probability")
    parser.add_argument(
        "--close-mosaic",
        type=int,
        default=10,
        help="Disable mosaic in final N epochs",
    )
    

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"dataset.yaml not found: {data_path}")

    model = YOLO(args.model)

    train_results = model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        pretrained=True,
        verbose=True,
       
        # Augmentations
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        fliplr=args.fliplr,
        flipud=args.flipud,
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        close_mosaic=args.close_mosaic,
    )

    print("Training finished.")
    print(f"Best checkpoint: {Path(train_results.save_dir) / 'weights' / 'best.pt'}")

    val_results = model.val(data=str(data_path), split="val")
    print("Validation complete.")
    print(val_results)


if __name__ == "__main__":
    main()
