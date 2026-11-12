"""Validation-only end-to-end sweep for local endpoint residual refiners."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from evaluate_targetclass_end_to_end_length import apply_visual_prompt, iou, padded_crop
from train_endpoint_residual_refiner import (ANCHOR, PATCH_SIZE, EndpointRefiner,
                                              endpoint_patch, load_points, read_image)


def instances(path: Path, width: int, height: int) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    points = load_points(path, width, height)
    for line, keypoints in zip(path.read_text(encoding="utf-8").splitlines(), points):
        value = np.asarray([float(x) for x in line.split()]); cx, cy, bw, bh = value[1:5]
        result.append((np.asarray([(cx-bw/2)*width, (cy-bh/2)*height,
                                  (cx+bw/2)*width, (cy+bh/2)*height]), keypoints))
    return result


def arc(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def feature(patch: np.ndarray, endpoint_type: int) -> np.ndarray:
    image = cv2.cvtColor(patch.astype(np.float32) / 255.0, cv2.COLOR_BGR2RGB)
    kind = np.full((PATCH_SIZE, PATCH_SIZE, 1), float(endpoint_type), np.float32)
    yy, xx = np.mgrid[-1:1:complex(PATCH_SIZE), -1:1:complex(PATCH_SIZE)]
    return np.concatenate([image, kind, xx[..., None].astype(np.float32),
                           yy[..., None].astype(np.float32)], axis=2).transpose(2, 0, 1)


def score(records: list[dict], prediction_key: str) -> dict:
    true = np.asarray([r["true_um"] for r in records]); pred = np.asarray([r[prediction_key] for r in records])
    ape = np.abs(pred - true) / true * 100
    return {"n": len(records), "mape_percent": float(ape.mean()),
            "median_ape_percent": float(np.median(ape)), "mae_um": float(np.abs(pred-true).mean()),
            "ape_ge_10": int((ape >= 10).sum()), "ape_ge_20": int((ape >= 20).sum()),
            "signed_bias_percent": float(np.mean((pred-true)/true*100)),
            "direction_correct": int(sum(r["direction_correct"] for r in records)),
            "direction_accuracy_percent": float(np.mean([r["direction_correct"] for r in records])*100)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reviewed-labels", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--refiner-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--endpoint-mode", choices=("both", "head", "tail"))
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--cap-fraction", type=float)
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    physical: dict[str, list[dict | None]] = {}
    with (args.dataset / "metadata/measurements.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != args.split: continue
            value = ({"true_um": float(row["body_length_um"]), "ruler_px": float(row["ruler_length_px"])}
                     if row["body_length_um"].strip() and row["ruler_length_px"].strip() else None)
            physical.setdefault(row["image"], []).append(value)

    detector, pose = YOLO(str(args.detector)), YOLO(str(args.pose))
    base, endpoint_features, inverse_affines = [], [], []
    total = detector_matched = 0
    for image_path in sorted((args.dataset / "images" / args.split).iterdir()):
        image = read_image(image_path); height, width = image.shape[:2]
        old = instances(args.dataset / "labels" / args.split / f"{image_path.stem}.txt", width, height)
        reviewed = instances(args.reviewed_labels / f"{image_path.stem}.txt", width, height)
        total += len(old)
        detection = detector.predict(image, imgsz=1280, conf=0.001, device=args.device, verbose=False)[0]
        det_boxes = detection.boxes.xyxy.cpu().numpy() if detection.boxes is not None else np.empty((0, 4))
        pairs = sorted(((iou(det, gt[0]), di, gi) for di, det in enumerate(det_boxes)
                        for gi, gt in enumerate(old)), reverse=True)
        used_det, used_gt = set(), set()
        for overlap, di, gi in pairs:
            if overlap < .5 or di in used_det or gi in used_gt: continue
            used_det.add(di); used_gt.add(gi); detector_matched += 1
            measure = physical[image_path.relative_to(args.dataset).as_posix()][gi]
            if measure is None: continue
            left, top, right, bottom = padded_crop(det_boxes[di], width, height, .15)
            reference = det_boxes[di] - np.asarray([left, top, left, top])
            crop = apply_visual_prompt(image[top:bottom, left:right], reference)
            result = pose.predict(crop, imgsz=1024, conf=.001, device=args.device, verbose=False)[0]
            if result.boxes is None or result.keypoints is None: continue
            boxes = result.boxes.xyxy.cpu().numpy(); classes = result.boxes.cls.cpu().numpy().astype(int)
            conf = result.boxes.conf.cpu().numpy(); candidates = np.flatnonzero(classes == 0)
            if not len(candidates): continue
            overlaps = np.asarray([iou(boxes[index], reference) for index in candidates])
            tied = candidates[overlaps == overlaps.max()]; chosen = int(tied[np.argmax(conf[tied])])
            points = result.keypoints.xy.cpu().numpy()[chosen] + np.asarray([left, top])
            gt = reviewed[gi][1]; scale = 400.0 / measure["ruler_px"]
            feature_indices = []
            ok = True
            for endpoint_type, endpoint_index, inward_index in ((0, 0, 2), (1, 14, 12)):
                extracted = endpoint_patch(image, points[endpoint_index], points[inward_index])
                if extracted is None: ok = False; break
                patch, affine = extracted; feature_indices.append(len(endpoint_features))
                endpoint_features.append(feature(patch, endpoint_type)); inverse_affines.append(cv2.invertAffineTransform(affine))
            if not ok: continue
            base.append({"image": image_path.name, "instance": gi+1, "points": points, "gt": gt,
                         "scale": scale, "true_um": measure["true_um"], "feature_indices": feature_indices})

    inputs = torch.from_numpy(np.asarray(endpoint_features)).float()
    checkpoints = ([args.checkpoint] if args.checkpoint else
                   sorted(args.refiner_dir.glob("epoch*.pt"), key=lambda p: int(p.stem[5:])))
    if not args.checkpoint and (args.refiner_dir / "best.pt").exists(): checkpoints.append(args.refiner_dir / "best.pt")
    results = []
    diagnostic_records: dict[str, list[dict]] = {}
    for checkpoint in checkpoints:
        model = EndpointRefiner().to(device); payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"]); model.eval(); outputs = []
        with torch.no_grad():
            for start in range(0, len(inputs), 128): outputs.append(model(inputs[start:start+128].to(device)).cpu().numpy())
        residuals = np.concatenate(outputs)
        proposed = []
        for inverse, residual in zip(inverse_affines, residuals):
            patch_point = ANCHOR + residual * 64.0
            proposed.append(cv2.transform(patch_point.reshape(1,1,2).astype(np.float32), inverse)[0,0])
        endpoint_modes = (args.endpoint_mode,) if args.endpoint_mode else ("both", "head", "tail")
        alphas = (args.alpha,) if args.alpha is not None else (.25, .5, .75, 1.0)
        caps = (args.cap_fraction,) if args.cap_fraction is not None else (.03, .05, .08, .12)
        for endpoint_mode in endpoint_modes:
            for alpha in alphas:
                for cap_fraction in caps:
                    records = []
                    for item in base:
                        points = item["points"].copy(); raw_arc = arc(points)
                        for local, endpoint_index in enumerate((0, 14)):
                            if endpoint_mode == "head" and endpoint_index == 14: continue
                            if endpoint_mode == "tail" and endpoint_index == 0: continue
                            target = proposed[item["feature_indices"][local]]
                            delta = (target - points[endpoint_index]) * alpha
                            cap = cap_fraction * raw_arc; magnitude = float(np.linalg.norm(delta))
                            if magnitude > cap: delta *= cap / magnitude
                            points[endpoint_index] += delta
                        gt = item["gt"]
                        direct = np.linalg.norm(points[0]-gt[0])+np.linalg.norm(points[-1]-gt[-1])
                        reverse = np.linalg.norm(points[0]-gt[-1])+np.linalg.norm(points[-1]-gt[0])
                        records.append({"image": item["image"], "instance": item["instance"],
                                        "true_um": item["true_um"],
                                        "raw_um": arc(item["points"])*item["scale"],
                                        "pred_um": arc(points)*item["scale"],
                                        "direction_correct": bool(direct <= reverse),
                                        "head_shift_fraction": float(np.linalg.norm(points[0]-item["points"][0]) / max(raw_arc, 1e-6)),
                                        "tail_shift_fraction": float(np.linalg.norm(points[-1]-item["points"][-1]) / max(raw_arc, 1e-6))})
                    metric = score(records, "pred_um")
                    metric.update({"checkpoint": checkpoint.name, "training_epoch": payload.get("epoch"),
                                   "holdout_loss": payload.get("holdout_loss"), "endpoint_mode": endpoint_mode,
                                   "alpha": alpha, "cap_fraction": cap_fraction})
                    results.append(metric)
                    key = f"{checkpoint.name}_{endpoint_mode}_a{alpha}_c{cap_fraction}"
                    if ((checkpoint.name == "epoch15.pt" and endpoint_mode == "both" and alpha == 1.0 and cap_fraction == .12) or
                        (checkpoint.name == "epoch10.pt" and endpoint_mode == "both" and alpha == 1.0 and cap_fraction == .03)):
                        diagnostic_records[key] = records
    raw_records = []
    for item in base:
        gt=item["gt"]; points=item["points"]
        direct=np.linalg.norm(points[0]-gt[0])+np.linalg.norm(points[-1]-gt[-1])
        reverse=np.linalg.norm(points[0]-gt[-1])+np.linalg.norm(points[-1]-gt[0])
        raw_records.append({"true_um":item["true_um"], "pred_um":arc(points)*item["scale"],
                            "direction_correct":bool(direct<=reverse)})
    ranked = sorted(results, key=lambda value: (value["mape_percent"], -value["direction_accuracy_percent"]))
    summary = {"protocol": ("locked validation-selected configuration; no test-set tuning" if args.split == "test"
                            else "validation-only checkpoint and safety-gate selection; fixed test untouched"),
               "split":args.split,
               "ground_truth_instances":total,"detector_matched_instances":detector_matched,
               "pose_predictions":len(base),"baseline":score(raw_records,"pred_um"),"top20":ranked[:20]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    with (args.output/"all_candidates.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=ranked[0].keys());writer.writeheader();writer.writerows(ranked)
    for key, records in diagnostic_records.items():
        with (args.output/f"per_instance_{key}.csv").open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
