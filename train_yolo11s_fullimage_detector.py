"""Build a non-destructive box-only dataset and train a YOLO11s whole-image detector."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def build_box_dataset(source: Path, destination: Path) -> Path:
    for split in ("train", "val"):
        image_source = source / "images" / split
        image_destination = destination / "images" / split
        label_destination = destination / "labels" / split
        image_destination.mkdir(parents=True, exist_ok=True)
        label_destination.mkdir(parents=True, exist_ok=True)
        for image in image_source.iterdir():
            target = image_destination / image.name
            if not target.exists():
                try:
                    target.symlink_to(image.resolve())
                except OSError:
                    shutil.copy2(image, target)
        for source_label in (source / "labels" / split).glob("*.txt"):
            output = []
            for line in source_label.read_text(encoding="utf-8").splitlines():
                values = line.split()
                if len(values) >= 5:
                    output.append(" ".join(values[:5]))
            (label_destination / source_label.name).write_text(
                "\n".join(output) + ("\n" if output else ""), encoding="utf-8")
    yaml_path = destination / "detector.yaml"
    yaml_path.write_text(
        f"path: {destination.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n"
        "names:\n  0: brine_shrimp\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("brine_shrimp_yolo_pose_archive_15pt_trunkboxes"))
    parser.add_argument("--dataset", type=Path, default=Path("brine_shrimp_fullimage_detector_boxonly"))
    parser.add_argument("--model", default="yolo11s.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--name", default="brine_yolo11s_fullimage_detector_smoke")
    args = parser.parse_args()
    yaml_path = build_box_dataset(args.source, args.dataset)
    model = YOLO(args.model)
    model.train(
        data=str(yaml_path), epochs=args.epochs, imgsz=1280, batch=4, device=0,
        workers=4, amp=True, cache=False, optimizer="AdamW", lr0=8e-4,
        lrf=0.08, weight_decay=5e-4, warmup_epochs=1.0,
        degrees=5.0, translate=0.08, scale=0.25, fliplr=0.5,
        mosaic=0.35, close_mosaic=2, patience=max(args.epochs, 8),
        project="runs", name=args.name, exist_ok=True, plots=True,
    )


if __name__ == "__main__":
    main()
