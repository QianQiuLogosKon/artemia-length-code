"""End-to-end length evaluation: whole-image detector -> target-class crop Pose."""
from __future__ import annotations

import csv
import json
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "brine_shrimp_yolo_pose_archive_15pt_trunkboxes"
DETECTOR = ROOT / "runs" / "brine_shrimp_pose_trunkboxes_15pt" / "weights" / "best.pt"
DEFAULT_CROP_POSE = ROOT / "runs" / "brine_shrimp_pose_trunkboxes_crops_targetclass_15pt" / "weights" / "best.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "brine_shrimp_pose_targetclass_end_to_end_eval"
IOU_THRESHOLD = 0.5
DEFAULT_PADDING = 0.15
TARGET_CLASS = 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right-left) * max(0.0, bottom-top)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return float(inter / union) if union else 0.0


def polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def load_gt(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        v = np.asarray([float(x) for x in line.split()])
        cx, cy, bw, bh = v[1:5]
        boxes.append(np.asarray([(cx-bw/2)*width, (cy-bh/2)*height, (cx+bw/2)*width, (cy+bh/2)*height]))
    return boxes


def padded_crop(box: np.ndarray, width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2-x1)*padding, (y2-y1)*padding
    return (max(0, int(np.floor(x1-pad_x))), max(0, int(np.floor(y1-pad_y))),
            min(width, int(np.ceil(x2+pad_x))), min(height, int(np.ceil(y2+pad_y))))

def apply_visual_prompt(crop: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x1,y1,x2,y2=np.round(reference).astype(int); x1=max(0,x1); y1=max(0,y1); x2=min(crop.shape[1]-1,x2); y2=min(crop.shape[0]-1,y2)
    overlay=crop.copy(); cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,255),-1)
    out=cv2.addWeighted(overlay,.16,crop,.84,0); cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,255),3); return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-pose", type=Path, default=DEFAULT_CROP_POSE)
    parser.add_argument("--detector", type=Path, default=DETECTOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING)
    parser.add_argument("--match-iou", type=float, default=IOU_THRESHOLD)
    parser.add_argument("--detector-imgsz", type=int, default=1280)
    parser.add_argument("--detector-augment", action="store_true")
    parser.add_argument("--secondary-detector-imgsz", type=int)
    parser.add_argument("--secondary-union-iou", type=float, default=0.3)
    parser.add_argument("--secondary-detector-conf", type=float, default=0.001)
    args = parser.parse_args()
    crop_pose_path, output, split, padding = args.crop_pose, args.output, args.split, args.padding
    output.mkdir(parents=True, exist_ok=True)
    physical: dict[str, list[dict[str, float] | None]] = {}
    with (DATASET / "metadata" / "measurements.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["split"] == split:
                length_text, ruler_text = row["body_length_um"].strip(), row["ruler_length_px"].strip()
                measurement = ({"length_um": float(length_text), "ruler_px": float(ruler_text)}
                               if length_text and ruler_text else None)
                physical.setdefault(row["image"], []).append(measurement)
    detector, crop_pose = YOLO(str(args.detector)), YOLO(str(crop_pose_path))
    rows, unmatched_detector, unmatched_target = [], [], []
    total = detector_matches = secondary_added_total = detector_output_total = 0
    for image_path in sorted((DATASET / "images" / split).iterdir()):
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = image.shape[:2]
        gt_boxes = load_gt(DATASET / "labels" / split / f"{image_path.stem}.txt", w, h)
        total += len(gt_boxes)
        image_key = image_path.relative_to(DATASET).as_posix()
        detection = detector.predict(
            image,
            imgsz=args.detector_imgsz,
            conf=0.001,
            device=0,
            augment=args.detector_augment,
            verbose=False,
        )[0]
        det_boxes = detection.boxes.xyxy.cpu().numpy() if detection.boxes is not None else np.empty((0, 4))
        if args.secondary_detector_imgsz:
            secondary = detector.predict(
                image,
                imgsz=args.secondary_detector_imgsz,
                conf=args.secondary_detector_conf,
                device=0,
                verbose=False,
            )[0]
            secondary_boxes = (secondary.boxes.xyxy.cpu().numpy()
                               if secondary.boxes is not None else np.empty((0, 4)))
            secondary_confs = (secondary.boxes.conf.cpu().numpy()
                               if secondary.boxes is not None else np.empty(0))
            merged = [box for box in det_boxes]
            for index in np.argsort(-secondary_confs):
                candidate = secondary_boxes[index]
                max_overlap = max((iou(candidate, existing) for existing in merged), default=0.0)
                if max_overlap < args.secondary_union_iou:
                    merged.append(candidate)
                    secondary_added_total += 1
            det_boxes = np.asarray(merged, dtype=float).reshape(-1, 4)
        detector_output_total += len(det_boxes)
        candidates = sorted(((iou(det, gt), di, gi) for di, det in enumerate(det_boxes) for gi, gt in enumerate(gt_boxes)), reverse=True)
        used_dets, used_gt = set(), set()
        for det_iou, det_index, gt_index in candidates:
            if det_iou < args.match_iou or det_index in used_dets or gt_index in used_gt:
                continue
            used_dets.add(det_index); used_gt.add(gt_index); detector_matches += 1
            measurement = physical[image_key][gt_index]
            if measurement is None:
                continue
            left, top, right, bottom = padded_crop(det_boxes[det_index], w, h, padding)
            crop = image[top:bottom, left:right]
            if os.environ.get("VISUAL_PROMPT") == "1":
                crop = apply_visual_prompt(crop, det_boxes[det_index] - np.asarray([left, top, left, top]))
            pose = crop_pose.predict(crop, imgsz=1024, conf=0.001, device=0, verbose=False)[0]
            boxes = pose.boxes.xyxy.cpu().numpy() if pose.boxes is not None else np.empty((0, 4))
            classes = pose.boxes.cls.cpu().numpy().astype(int) if pose.boxes is not None else np.empty(0, dtype=int)
            confs = pose.boxes.conf.cpu().numpy() if pose.boxes is not None else np.empty(0)
            points = pose.keypoints.xy.cpu().numpy() if pose.keypoints is not None else np.empty((0, 15, 2))
            targets = np.flatnonzero(classes == TARGET_CLASS)
            if not len(targets):
                unmatched_target.append({"image": image_path.name, "instance_id": gt_index+1, "detector_iou": det_iou, "reason": "no_target_class_prediction"})
                continue
            # The detector box, transformed into crop coordinates, is available
            # at inference and identifies the requested target without true labels.
            reference = np.asarray([det_boxes[det_index][0]-left, det_boxes[det_index][1]-top,
                                    det_boxes[det_index][2]-left, det_boxes[det_index][3]-top])
            overlaps = np.asarray([iou(boxes[i], reference) for i in targets])
            best_candidates = targets[overlaps == overlaps.max()]
            pose_index = int(best_candidates[np.argmax(confs[best_candidates])])
            predicted_px = polyline_length(points[pose_index])
            true_um = measurement["length_um"]
            predicted_um = predicted_px * 400.0 / measurement["ruler_px"]
            ape = abs(predicted_um-true_um)/true_um*100
            rows.append({"image": image_path.name, "instance_id": gt_index+1, "detector_iou": det_iou, "target_confidence": float(confs[pose_index]),
                         "true_length_um": true_um, "predicted_length_um": predicted_um, "absolute_error_um": abs(predicted_um-true_um), "relative_error_percent": ape})
        for gt_index in set(range(len(gt_boxes))) - used_gt:
            unmatched_detector.append({"image": image_path.name, "instance_id": gt_index+1, "reason": "detector_iou_below_0.5"})
    if not rows:
        raise RuntimeError("No end-to-end predictions matched")
    ape = np.asarray([float(x["relative_error_percent"]) for x in rows]); ae = np.asarray([float(x["absolute_error_um"]) for x in rows])
    summary = {"split": split, "padding": padding, "ground_truth_instances": total, "detector_matched_instances": detector_matches, "detector_recall": detector_matches/total,
               "crop_pose_predicted_instances": len(rows), "end_to_end_recall": len(rows)/total, "length_mape_percent": float(ape.mean()),
               "length_median_ape_percent": float(np.median(ape)), "length_mae_um": float(ae.mean()), "ape_ge_10_percent": int((ape>=10).sum()), "ape_ge_20_percent": int((ape>=20).sum())}
    summary["match_iou"] = args.match_iou
    for filename, contents in (("length_errors.csv", rows), ("unmatched_detector.csv", unmatched_detector), ("unmatched_target_pose.csv", unmatched_target)):
        if contents:
            with (output / filename).open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=contents[0].keys()); writer.writeheader(); writer.writerows(contents)
    summary["crop_pose_weights"] = str(crop_pose_path)
    summary["detector_imgsz"] = args.detector_imgsz
    summary["detector_augment"] = args.detector_augment
    summary["secondary_detector_imgsz"] = args.secondary_detector_imgsz
    summary["secondary_union_iou"] = args.secondary_union_iou
    summary["secondary_detector_conf"] = args.secondary_detector_conf
    summary["secondary_added_detections"] = secondary_added_total
    summary["detector_output_total"] = detector_output_total
    summary["detector_unmatched_predictions"] = detector_output_total - detector_matches
    summary["detector_precision_at_match_iou"] = detector_matches / detector_output_total if detector_output_total else 0.0
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
