"""Validation-only inference on the exact cached detector candidates used by HRNet.

No training, detector rerun, GT-guided candidate choice, or fixed-test access.
Every cached candidate is saved, including invalid / absent Pose outputs.
"""
from pathlib import Path
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / "output/baseline_hrnet_20260922_v1"
DATA = BASE / "data_v2"
ARCHIVE = BASE
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ARCHIVE / "src"))
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from common import sha256, read_image, crop_bounds, prompt, arc, save_json
from infer_brine_shrimp_endpoint_refined import padded_crop, apply_visual_prompt, iou
from train_endpoint_residual_refiner import ANCHOR, EndpointRefiner, endpoint_patch
from evaluate_endpoint_residual_refiner import feature


THRESHOLDS = [.001, .002, .005, .01, .02, .03, .05, .075, .1, .15, .2,
              .25, .3, .4, .5, .6, .7, .8, .9, .95]


def main():
    dest = OUT / "inference"
    dest.mkdir(exist_ok=False)
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert sha256(manifest_path) == "6631a3de2ee64f6600848a8593daff86f46e1d8274e961647d976dae53c40fa8"
    assert manifest["allowed_splits"] == ["train", "val"]
    images = [im for im in manifest["images"] if im["split"] == "val"]
    assert len(images) == 81
    assert sum(len(im["detections"]) for im in images) == 1498
    weights = ROOT / "deployment/endpoint_refined_v1/weights"
    expected = json.loads((weights / "SHA256SUMS.json").read_text(encoding="utf-8-sig"))
    hashes = {name: sha256(weights / name) for name in expected}
    assert all(hashes[name] == expected[name].lower() for name in expected)
    assert hashes["detector_best.pt"] == manifest["detector"]["weight_sha256"]
    protocol = dict(split="val", fixed_test_access=False, training=False,
        image_count=81, ground_truth_instances=221, cached_candidates=1498,
        data_manifest_sha256=sha256(manifest_path), weight_sha256=hashes,
        matching="offline global greedy IoU >= 0.5; descending (IoU, candidate index, GT index)",
        detector=manifest["detector"], detector_rerun=False,
        padding=.15, visual_prompt="yellow fill alpha .16, border 3 original-image pixels",
        pose=dict(imgsz=1024, conf=.001, iou=.7, max_det=300, half=False, augment=False,
                  batch=1, rect=True, choice="class 0, largest IoU with detector box; confidence tie-break"),
        endpoint_refiner=dict(epoch=10, endpoints=[1,15], alpha=1., cap_fraction=.03,
            batch_size=128, degenerate_patch="keep raw endpoint, matching existing deployment behavior"),
        threshold_grid=THRESHOLDS,
        threshold_analysis="post-NMS cached confidence filtering, with offline matching recomputed at each threshold; no change to deployment defaults",
        primary_threshold=.001,
        secondary_thresholds=[.02,.5],
        threshold_recommendation_rule="largest predeclared grid threshold retaining the same matched GT set as 0.001, based on detector coverage only",
        gt_usage="only after inference, for scoring and diagnostic figures; no GT enters inference",
        evidence_scope="unified validation evaluation of existing checkpoints, not equal-training-budget architecture isolation")
    save_json(OUT / "protocol.json", protocol)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    device = torch.device("cuda:0")
    pose = YOLO(str(weights / "pose_visual_prompt_best.pt"))
    model = EndpointRefiner().to(device)
    payload = torch.load(weights / "endpoint_refiner_epoch10.pt", map_location=device, weights_only=True)
    assert payload["epoch"] == 10
    model.load_state_dict(payload["model"])
    model.eval()
    save_json(dest / "yolo_checkpoint_metadata.json", dict(names=pose.names,
        train_args=pose.ckpt.get("train_args"), train_metrics=pose.ckpt.get("train_metrics"),
        checkpoint_version=pose.ckpt.get("version")))
    raw, refined, pending = {}, {}, []
    verified_inputs = {}
    prompt_checks = 0
    started = time.perf_counter()

    def flush():
        if not pending:
            return
        inputs = torch.from_numpy(np.stack([p[3] for p in pending])).to(device)
        with torch.inference_mode():
            residuals = model(inputs).float().cpu().numpy()
        for (cid, endpoint_type, inverse, _), residual in zip(pending, residuals):
            idx = 0 if endpoint_type == 0 else 14
            point = cv2.transform((ANCHOR + residual * 64).reshape(1,1,2).astype(np.float32), inverse)[0,0]
            points = np.asarray(raw[cid]["points"], np.float64)
            delta = point - points[idx]
            cap = .03 * arc(points)
            magnitude = float(np.linalg.norm(delta))
            if magnitude > cap:
                delta *= cap / magnitude
            refined[cid]["points"][idx] = (points[idx] + delta).tolist()
            refined[cid]["endpoint_shifts_px"][endpoint_type] = float(np.linalg.norm(delta))
            refined[cid]["cap_active"][endpoint_type] = bool(magnitude > cap)
        pending.clear()

    for ni, im in enumerate(images, 1):
        path = DATA / im["image"]
        assert path.resolve().is_relative_to((DATA / "images/val").resolve())
        assert sha256(path) == im["image_sha256"]
        verified_inputs[im["id"]] = im["image_sha256"]
        image = read_image(path)
        h,w = image.shape[:2]
        assert (w,h) == (im["width"],im["height"])
        # This loop deliberately does not read the image's GT annotations.
        for d in im["detections"]:
            cid, box = d["id"], np.asarray(d["box"], np.float64)
            bounds = crop_bounds(box,w,h,.15)
            assert tuple(bounds) == padded_crop(box,w,h,.15)
            left,top,right,bottom = bounds
            if right <= left or bottom <= top:
                raw[cid] = dict(status="empty_crop")
                refined[cid] = dict(status="empty_crop")
                continue
            reference = box - [left,top,left,top]
            crop = image[top:bottom,left:right]
            prompted = apply_visual_prompt(crop,reference)
            assert np.array_equal(prompted,prompt(crop,reference))
            prompt_checks += 1
            result = pose.predict(prompted, device="0", imgsz=1024, conf=.001,
                iou=.7,max_det=300,half=False,augment=False,rect=True,verbose=False,
                save=False,save_txt=False,save_crop=False)[0]
            value = dict(status="no_pose")
            if result.boxes is not None and result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidence = result.boxes.conf.cpu().numpy()
                candidates = np.flatnonzero(classes==0)
                value = dict(status="no_target_class_pose", pose_detection_count=len(classes))
                if len(candidates):
                    overlaps = np.asarray([iou(boxes[j],reference) for j in candidates])
                    tied = candidates[overlaps==overlaps.max()]
                    chosen = int(tied[np.argmax(confidence[tied])])
                    points = result.keypoints.xy.cpu().numpy()[chosen] + [left,top]
                    ok = points.shape==(15,2) and np.isfinite(points).all() and arc(points)>0
                    value = dict(status="ok" if ok else "invalid_keypoints",points=points.tolist(),
                        pose_confidence=float(confidence[chosen]),pose_reference_iou=float(overlaps.max()),
                        pose_detection_count=len(classes),pose_box=(boxes[chosen]+[left,top,left,top]).tolist())
            raw[cid] = value
            refined[cid] = json.loads(json.dumps(value))
            if value["status"] == "ok":
                refined[cid].update(endpoint_shifts_px=[0.,0.],cap_active=[False,False],degenerate_patch=[False,False])
                for kind,idx,inside in ((0,0,2),(1,14,12)):
                    extracted = endpoint_patch(image,points[idx],points[inside])
                    if extracted is None:
                        refined[cid]["degenerate_patch"][kind] = True
                        continue
                    patch,affine = extracted
                    pending.append((cid,kind,cv2.invertAffineTransform(affine),feature(patch,kind)))
                    if len(pending)==128:
                        flush()
        progress = dict(images=ni,total_images=len(images),candidates=len(raw),
            valid_outputs=sum(v["status"]=="ok" for v in raw.values()),seconds=round(time.perf_counter()-started,2))
        save_json(dest / "progress.json",progress)
        if ni%5==0 or ni==len(images):
            print(json.dumps(progress),flush=True)
    flush()
    assert len(raw)==len(refined)==1498
    for cid,v in raw.items():
        if v["status"]=="ok":
            assert np.array_equal(np.asarray(v["points"])[1:14],np.asarray(refined[cid]["points"])[1:14])
            assert np.isfinite(refined[cid]["points"]).all() and arc(refined[cid]["points"])>0
    save_json(dest / "yolo_raw_all_candidates.json",raw)
    save_json(dest / "yolo_refined_all_candidates.json",refined)
    save_json(dest / "inference_audit.json",dict(status="PASS",split="val",image_count=len(images),
        candidates=len(raw),prompt_and_bounds_identity_checks=prompt_checks,verified_image_hashes=verified_inputs,
        all_interior_points_unchanged=True,training_performed=False,fixed_test_access=False,
        report_files_modified=False,environment=dict(python=platform.python_version(),torch=torch.__version__,
        ultralytics=ultralytics.__version__,numpy=np.__version__,opencv=cv2.__version__,gpu=torch.cuda.get_device_name(0)),
        elapsed_seconds=time.perf_counter()-started,timing_scope="audit plus local YOLO/endpoint inference; not a fair model efficiency benchmark"))
    print(json.dumps(dict(status="PASS",candidates=len(raw),seconds=round(time.perf_counter()-started,2))),flush=True)


if __name__ == "__main__":
    main()
