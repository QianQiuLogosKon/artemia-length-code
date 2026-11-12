"""Production inference: detector -> visual-prompt Pose -> locked endpoint refiner.

The refiner configuration is intentionally fixed to the validation-selected
epoch10 checkpoint, both endpoints, alpha=1.0, and a 3% arc-length movement cap.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from train_endpoint_residual_refiner import (ANCHOR, PATCH_SIZE, EndpointRefiner,
                                              endpoint_patch, read_image)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, right-left) * max(0.0, bottom-top)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return float(intersection / union) if union else 0.0


def padded_crop(box: np.ndarray, width: int, height: int, padding: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box; pad_x, pad_y = (x2-x1)*padding, (y2-y1)*padding
    return (max(0, int(np.floor(x1-pad_x))), max(0, int(np.floor(y1-pad_y))),
            min(width, int(np.ceil(x2+pad_x))), min(height, int(np.ceil(y2+pad_y))))


def apply_visual_prompt(crop: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x1,y1,x2,y2=np.round(reference).astype(int); x1=max(0,x1);y1=max(0,y1)
    x2=min(crop.shape[1]-1,x2);y2=min(crop.shape[0]-1,y2)
    overlay=crop.copy();cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,255),-1)
    output=cv2.addWeighted(overlay,.16,crop,.84,0);cv2.rectangle(output,(x1,y1),(x2,y2),(0,255,255),3)
    return output


def arc(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def endpoint_feature(patch: np.ndarray, endpoint_type: int) -> torch.Tensor:
    image = cv2.cvtColor(patch.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)
    kind = np.full((PATCH_SIZE, PATCH_SIZE, 1), float(endpoint_type), np.float32)
    yy, xx = np.mgrid[-1:1:complex(PATCH_SIZE), -1:1:complex(PATCH_SIZE)]
    value = np.concatenate([image, kind, xx[..., None].astype(np.float32),
                            yy[..., None].astype(np.float32)], axis=2)
    return torch.from_numpy(value.transpose(2, 0, 1).copy()).unsqueeze(0)


def refine_endpoints(image: np.ndarray, points: np.ndarray, model: EndpointRefiner,
                     device: torch.device, cap_fraction: float = .03) -> tuple[np.ndarray, list[float]]:
    refined = points.copy(); original_arc = arc(points); shifts = []
    for endpoint_type, endpoint_index, inward_index in ((0, 0, 2), (1, 14, 12)):
        extracted = endpoint_patch(image, points[endpoint_index], points[inward_index])
        if extracted is None:
            shifts.append(0.0); continue
        patch, affine = extracted
        with torch.no_grad():
            residual = model(endpoint_feature(patch, endpoint_type).to(device))[0].cpu().numpy()
        proposed_patch = ANCHOR + residual * 64.0
        proposed = cv2.transform(proposed_patch.reshape(1, 1, 2).astype(np.float32),
                                 cv2.invertAffineTransform(affine))[0, 0]
        delta = proposed - points[endpoint_index]
        magnitude = float(np.linalg.norm(delta)); cap = cap_fraction * original_arc
        if magnitude > cap:
            delta *= cap / magnitude; magnitude = cap
        refined[endpoint_index] += delta; shifts.append(magnitude)
    return refined, shifts


def scale_lookup(path: Path | None, ruler_um: float) -> dict[str, float]:
    if path is None:
        return {}
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("ruler_length_px", "").strip():
                result[Path(row["image"]).name] = ruler_um / float(row["ruler_length_px"])
    return result


def draw(canvas: np.ndarray, box: np.ndarray, raw: np.ndarray, refined: np.ndarray,
         index: int, label: str) -> None:
    cv2.rectangle(canvas, tuple(np.round(box[:2]).astype(int)), tuple(np.round(box[2:]).astype(int)),
                  (255, 210, 40), 2)
    cv2.polylines(canvas, [np.round(raw).astype(np.int32)], False, (0, 170, 255), 2)
    cv2.polylines(canvas, [np.round(refined).astype(np.int32)], False, (20, 230, 20), 3)
    cv2.circle(canvas, tuple(np.round(refined[0]).astype(int)), 6, (0, 0, 255), -1)
    cv2.circle(canvas, tuple(np.round(refined[-1]).astype(int)), 6, (255, 0, 0), -1)
    origin = tuple(np.round(box[:2]).astype(int) + np.asarray([0, -7]))
    cv2.putText(canvas, f"#{index} {label}", origin, cv2.FONT_HERSHEY_SIMPLEX, .62,
                (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, f"#{index} {label}", origin, cv2.FONT_HERSHEY_SIMPLEX, .62,
                (20, 20, 20), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--refiner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--detector-conf", type=float, default=.5,
                        help="Validation-selected practical default (217/221 TP, 5 FP). Use 0.02 for maximum-recall review mode.")
    parser.add_argument("--pose-conf", type=float, default=.001)
    parser.add_argument("--um-per-pixel", type=float)
    parser.add_argument("--measurements-csv", type=Path)
    parser.add_argument("--ruler-um", type=float, default=400.0)
    args = parser.parse_args()

    images = ([args.source] if args.source.is_file() else
              sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
    if not images:
        raise FileNotFoundError(f"No supported images under {args.source}")
    args.output.mkdir(parents=True, exist_ok=True); annotated_dir = args.output / "annotated"
    annotated_dir.mkdir(exist_ok=True)
    device = torch.device("cpu" if args.device == "cpu" or not torch.cuda.is_available() else f"cuda:{args.device}")
    detector, pose = YOLO(str(args.detector)), YOLO(str(args.pose))
    refiner = EndpointRefiner().to(device)
    payload = torch.load(args.refiner, map_location=device, weights_only=False)
    refiner.load_state_dict(payload["model"]); refiner.eval()
    scales = scale_lookup(args.measurements_csv, args.ruler_um)
    rows = []
    for image_path in images:
        image = read_image(image_path); height, width = image.shape[:2]; canvas = image.copy()
        detection = detector.predict(image, imgsz=1280, conf=args.detector_conf,
                                     device=args.device, verbose=False)[0]
        boxes = detection.boxes.xyxy.cpu().numpy() if detection.boxes is not None else np.empty((0, 4))
        det_conf = detection.boxes.conf.cpu().numpy() if detection.boxes is not None else np.empty(0)
        # Stable spatial numbering is easier to audit than confidence ordering.
        order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
        for number, detection_index in enumerate(order, 1):
            box = boxes[detection_index]
            left, top, right, bottom = padded_crop(box, width, height, .15)
            reference = box - np.asarray([left, top, left, top])
            crop = apply_visual_prompt(image[top:bottom, left:right], reference)
            prediction = pose.predict(crop, imgsz=1024, conf=args.pose_conf,
                                      device=args.device, verbose=False)[0]
            if prediction.boxes is None or prediction.keypoints is None: continue
            pose_boxes = prediction.boxes.xyxy.cpu().numpy()
            classes = prediction.boxes.cls.cpu().numpy().astype(int)
            confidence = prediction.boxes.conf.cpu().numpy(); candidates = np.flatnonzero(classes == 0)
            if not len(candidates): continue
            overlaps = np.asarray([iou(pose_boxes[i], reference) for i in candidates])
            tied = candidates[overlaps == overlaps.max()]; selected = int(tied[np.argmax(confidence[tied])])
            raw = prediction.keypoints.xy.cpu().numpy()[selected] + np.asarray([left, top])
            refined, shifts = refine_endpoints(image, raw, refiner, device)
            raw_px, refined_px = arc(raw), arc(refined)
            um_per_px = args.um_per_pixel if args.um_per_pixel is not None else scales.get(image_path.name)
            label = f"{refined_px:.1f}px" if um_per_px is None else f"{refined_px*um_per_px:.1f}um"
            draw(canvas, box, raw, refined, number, label)
            rows.append({"image": image_path.name, "shrimp_id": number,
                         "detector_confidence": float(det_conf[detection_index]),
                         "pose_confidence": float(confidence[selected]),
                         "raw_length_px": raw_px, "refined_length_px": refined_px,
                         "head_shift_px": shifts[0], "tail_shift_px": shifts[1],
                         "um_per_pixel": "" if um_per_px is None else um_per_px,
                         "raw_length_um": "" if um_per_px is None else raw_px * um_per_px,
                         "refined_length_um": "" if um_per_px is None else refined_px * um_per_px,
                         "keypoints_refined_json": json.dumps(refined.tolist(), separators=(",", ":"))})
        cv2.imencode(image_path.suffix if image_path.suffix.lower() in IMAGE_SUFFIXES else ".jpg", canvas)[1].tofile(
            str(annotated_dir / image_path.name))
    with (args.output / "length_predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = list(rows[0].keys()) if rows else ["image", "shrimp_id"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (args.output / "run_config.json").write_text(json.dumps({"source": str(args.source),
        "detector": str(args.detector), "pose": str(args.pose), "refiner": str(args.refiner),
        "detector_conf": args.detector_conf, "pose_conf": args.pose_conf,
        "endpoint_alpha": 1.0, "endpoint_cap_fraction": .03,
        "images": len(images), "predictions": len(rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(images), "predictions": len(rows), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
