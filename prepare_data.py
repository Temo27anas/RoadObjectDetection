from __future__ import annotations

import argparse
import random
import shutil
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare KITTI dataset in YOLO format with class filtering and data split."
    )
    parser.add_argument("--images", type=Path, required=True, help="Path to KITTI image_2 folder")
    parser.add_argument("--labels", type=Path, required=True, help="Path to KITTI label_2 folder")
    parser.add_argument("--output", type=Path, required=True, help="Output folder for YOLO dataset")
    parser.add_argument(
        "--classes",
        nargs="+",
        required=True,
        help="Classes to keep, e.g. Car Pedestrian Cyclist",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Testing split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument(
        "--difficulty",
        choices=("all", "easy", "moderate", "hard"),
        default="all",
        help="Keep only objects at the selected KITTI difficulty level.",
    )
    parser.add_argument(
        "--copy-empty-labels",
        action="store_true",
        help=(
            "Keep images that end up with no kept classes by writing empty YOLO label files. "
            "By default, such images are skipped."
        ),
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, val_ratio, test_ratio]
    if any(r < 0 for r in ratios):
        raise ValueError("Split ratios must be non-negative.")
    total = sum(ratios)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")


def read_png_size(path: Path) -> Optional[Tuple[int, int]]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def read_jpeg_size(path: Path) -> Optional[Tuple[int, int]]:
    with path.open("rb") as f:
        data = f.read()

    if len(data) < 2 or data[0] != 0xFF or data[1] != 0xD8:
        return None

    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2

        if marker in (0xD8, 0xD9):
            continue

        if i + 2 > len(data):
            break
        block_len = (data[i] << 8) + data[i + 1]
        if block_len < 2 or i + block_len > len(data):
            break

        # Start Of Frame markers that contain image size.
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            if block_len < 7:
                return None
            height = (data[i + 3] << 8) + data[i + 4]
            width = (data[i + 5] << 8) + data[i + 6]
            return int(width), int(height)

        i += block_len

    return None


def get_image_size(path: Path) -> Tuple[int, int]:
    suffix = path.suffix.lower()
    size: Optional[Tuple[int, int]] = None
    if suffix == ".png":
        size = read_png_size(path)
    elif suffix in {".jpg", ".jpeg"}:
        size = read_jpeg_size(path)

    if size is None:
        raise ValueError(
            f"Unsupported image format or failed to read image size for {path}. "
            "Use .png/.jpg/.jpeg images."
        )
    return size


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def get_kitti_difficulty(truncation: float, occlusion: int, bbox_height: float) -> Optional[str]:
    if bbox_height >= 40.0 and occlusion <= 0 and truncation <= 0.15:
        return "easy"
    if bbox_height >= 25.0 and occlusion <= 1 and truncation <= 0.30:
        return "moderate"
    if bbox_height >= 25.0 and occlusion <= 2 and truncation <= 0.50:
        return "hard"
    return None


def convert_kitti_line_to_yolo(
    line: str,
    class_to_id_lower: Dict[str, int],
    image_w: int,
    image_h: int,
    difficulty_filter: str,
) -> Optional[Tuple[str, str]]:
    parts = line.strip().split()
    if len(parts) < 8:
        return None

    class_name = parts[0].lower()
    if class_name not in class_to_id_lower:
        return None

    try:
        truncation = float(parts[1])
        occlusion = int(float(parts[2]))
        x1 = float(parts[4])
        y1 = float(parts[5])
        x2 = float(parts[6])
        y2 = float(parts[7])
    except ValueError:
        return None

    bbox_height = y2 - y1
    difficulty = get_kitti_difficulty(truncation, occlusion, bbox_height)
    if difficulty is None:
        return None
    if difficulty_filter != "all" and difficulty != difficulty_filter:
        return None

    x1 = clamp(x1, 0.0, float(image_w))
    y1 = clamp(y1, 0.0, float(image_h))
    x2 = clamp(x2, 0.0, float(image_w))
    y2 = clamp(y2, 0.0, float(image_h))

    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None

    x_center = (x1 + x2) / 2.0 / image_w
    y_center = (y1 + y2) / 2.0 / image_h
    w_norm = box_w / image_w
    h_norm = box_h / image_h

    class_id = class_to_id_lower[class_name]
    yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
    return yolo_line, difficulty


def discover_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    images_by_stem = {
        p.stem: p
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    }

    pairs: List[Tuple[Path, Path]] = []
    for label_path in labels_dir.glob("*.txt"):
        image_path = images_by_stem.get(label_path.stem)
        if image_path is not None:
            pairs.append((image_path, label_path))

    return sorted(pairs, key=lambda x: x[0].stem)


def split_items(
    items: Sequence[Tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    items = list(items)
    random.Random(seed).shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val : n_train + n_val + n_test]
    return train_items, val_items, test_items


def ensure_structure(output_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def process_split(
    split_name: str,
    items: Sequence[Tuple[Path, Path]],
    output_dir: Path,
    class_to_id_lower: Dict[str, int],
    difficulty_filter: str,
    copy_empty_labels: bool,
) -> Tuple[int, int, int, Dict[str, int]]:
    kept_images = 0
    skipped_images = 0
    written_boxes = 0
    difficulty_counts: Dict[str, int] = {"easy": 0, "moderate": 0, "hard": 0}

    for image_path, label_path in items:
        image_w, image_h = get_image_size(image_path)

        yolo_lines: List[str] = []
        with label_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                converted = convert_kitti_line_to_yolo(
                    raw_line,
                    class_to_id_lower,
                    image_w,
                    image_h,
                    difficulty_filter,
                )
                if converted is not None:
                    yolo_line, difficulty = converted
                    yolo_lines.append(yolo_line)
                    difficulty_counts[difficulty] += 1

        if not yolo_lines and not copy_empty_labels:
            skipped_images += 1
            continue

        out_image = output_dir / "images" / split_name / image_path.name
        out_label = output_dir / "labels" / split_name / f"{image_path.stem}.txt"

        shutil.copy2(image_path, out_image)
        with out_label.open("w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        kept_images += 1
        written_boxes += len(yolo_lines)

    return kept_images, skipped_images, written_boxes, difficulty_counts


def write_classes_file(output_dir: Path, classes: Sequence[str]) -> None:
    classes_path = output_dir / "classes.txt"
    with classes_path.open("w", encoding="utf-8") as f:
        for cls in classes:
            f.write(f"{cls}\n")


def write_dataset_yaml(output_dir: Path, classes: Sequence[str]) -> None:
    yaml_path = output_dir / "dataset.yaml"
    yaml_lines = [
        "# YOLO dataset config",
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(classes)}",
        "names:",
    ]
    yaml_lines.extend([f"  - {c}" for c in classes])

    with yaml_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(yaml_lines) + "\n")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    images_dir = args.images
    labels_dir = args.labels
    output_dir = args.output
    classes = list(args.classes)

    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    class_to_id_lower = {name.lower(): idx for idx, name in enumerate(classes)}

    pairs = discover_pairs(images_dir, labels_dir)
    if not pairs:
        raise RuntimeError(
            "No matching image/label pairs found. "
            "Check that file names match (e.g., 000001.png and 000001.txt)."
        )

    ensure_structure(output_dir)

    train_items, val_items, test_items = split_items(
        pairs, args.train_ratio, args.val_ratio, args.seed
    )

    t_kept, t_skipped, t_boxes, t_difficulties = process_split(
        "train",
        train_items,
        output_dir,
        class_to_id_lower,
        args.difficulty,
        args.copy_empty_labels,
    )
    v_kept, v_skipped, v_boxes, v_difficulties = process_split(
        "val",
        val_items,
        output_dir,
        class_to_id_lower,
        args.difficulty,
        args.copy_empty_labels,
    )
    te_kept, te_skipped, te_boxes, te_difficulties = process_split(
        "test",
        test_items,
        output_dir,
        class_to_id_lower,
        args.difficulty,
        args.copy_empty_labels,
    )

    write_classes_file(output_dir, classes)
    write_dataset_yaml(output_dir, classes)

    total_kept = t_kept + v_kept + te_kept
    total_skipped = t_skipped + v_skipped + te_skipped
    total_boxes = t_boxes + v_boxes + te_boxes
    total_difficulties = {
        "easy": t_difficulties["easy"] + v_difficulties["easy"] + te_difficulties["easy"],
        "moderate": t_difficulties["moderate"]
        + v_difficulties["moderate"]
        + te_difficulties["moderate"],
        "hard": t_difficulties["hard"] + v_difficulties["hard"] + te_difficulties["hard"],
    }

    print("Done. YOLO dataset created.")
    print(f"Selected classes: {classes}")
    print(f"Selected difficulty: {args.difficulty}")
    print(f"Matched pairs found: {len(pairs)}")
    print(f"Kept images: {total_kept}")
    print(f"Skipped images (no selected classes): {total_skipped}")
    print(f"Total YOLO boxes written: {total_boxes}")
    print("Boxes by KITTI difficulty:")
    print(f"  easy    : {total_difficulties['easy']}")
    print(f"  moderate: {total_difficulties['moderate']}")
    print(f"  hard    : {total_difficulties['hard']}")
    print("Per split:")
    print(f"  train: {t_kept} images")
    print(f"  val  : {v_kept} images")
    print(f"  test : {te_kept} images")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
