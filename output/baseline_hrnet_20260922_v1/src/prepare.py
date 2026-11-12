"""Build an additive train/val snapshot and cache ALL detector outputs. No test IO."""
from pathlib import Path
import argparse
import csv
import json
import platform
import shutil
import time
import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO
from common import sha256, save_json, read_image, match_boxes, crop_bounds, arc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out, root = args.output, args.workspace
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    original = root / 'brine_shrimp_yolo_pose_archive_15pt_trunkboxes'
    reviewed = root / 'brine_shrimp_yolo_pose_archive_15pt_trunkboxes_headtail_fully_reviewed'
    weight = root / 'deployment/endpoint_refined_v1/weights/detector_best.pt'
    measurements = {}
    with (original / 'metadata/measurements.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['split'] in ('train', 'val'):
                key = (row['split'], Path(row['image']).name, int(row['instance_id']))
                assert key not in measurements
                measurements[key] = row
    detector = YOLO(str(weight))
    detector_args = dict(imgsz=1280, conf=.001, iou=.7, max_det=300, device='0', half=False, augment=False, verbose=False)
    result = dict(schema_version=1, allowed_splits=['train', 'val'], padding=.15,
                  detector=dict(weight_sha256=sha256(weight), **detector_args),
                  sources=dict(image_dataset=str(original), label_dataset=str(reviewed),
                               metadata_sha256=sha256(original/'metadata/measurements.csv')),
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   cv2=cv2.__version__, ultralytics=ultralytics.__version__,
                                   gpu=torch.cuda.get_device_name(0)), images=[])
    seen_hashes = {}
    split_counts = {}
    start = time.time()
    for split in ('train', 'val'):
        (out/'images'/split).mkdir(parents=True)
        (out/'labels'/split).mkdir(parents=True)
        paths = sorted((original/'images'/split).iterdir())
        for index, path in enumerate(paths):
            label_path = reviewed/'labels'/split/(path.stem+'.txt')
            image = read_image(path)
            height, width = image.shape[:2]
            image_id = f'{split}_{index:04d}'
            destination = f'images/{split}/{image_id}{path.suffix.lower()}'
            image_hash, label_hash = sha256(path), sha256(label_path)
            assert image_hash not in seen_hashes or seen_hashes[image_hash] == split, 'Exact-image split leakage'
            seen_hashes[image_hash] = split
            reviewed_image = reviewed/'images'/split/path.name
            assert sha256(reviewed_image) == image_hash
            shutil.copy2(path, out/destination)
            shutil.copy2(label_path, out/'labels'/split/(image_id+'.txt'))
            truth = []
            for gi, line in enumerate(label_path.read_text(encoding='utf-8').splitlines()):
                v = np.asarray([float(x) for x in line.split()], np.float64)
                assert v.size == 50 and int(v[0]) == 0
                cx, cy, bw, bh = v[1:5]
                points = v[5:].reshape(15,3)
                assert np.all(points[:,2] > 0) and np.all((points[:,:2]>=0)&(points[:,:2]<=1))
                points = points[:,:2] * [width, height]
                assert np.max(np.abs(points[1::2] - (points[:-2:2]+points[2::2])/2)) < .003
                physical = measurements[(split, path.name, gi+1)]
                length_um = float(physical['body_length_um']) if physical['body_length_um'].strip() else None
                ruler_px = float(physical['ruler_length_px']) if physical['ruler_length_px'].strip() else None
                if length_um is not None and ruler_px is not None:
                    assert length_um > 0 and ruler_px > 0
                    assert abs(arc(points)*400/ruler_px-length_um) < .01
                elif split == 'val':
                    raise ValueError(f'Missing validation calibration: {path.name}, {gi+1}')
                box = [(cx-bw/2)*width,(cy-bh/2)*height,(cx+bw/2)*width,(cy+bh/2)*height]
                truth.append(dict(id=f'{image_id}:g{gi+1}', instance=gi+1, box=box,
                                  points=points.tolist(), length_px=arc(points), length_um=length_um,
                                  um_per_pixel=400/ruler_px if ruler_px else None))
            tick = time.perf_counter()
            predicted = detector.predict(image, **{k:v for k,v in detector_args.items() if k != 'half'})[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter()-tick
            detections = [dict(id=f'{image_id}:d{di}', box=box.tolist(), confidence=float(conf), cls=int(cls))
                          for di,(box,conf,cls) in enumerate(zip(predicted.boxes.xyxy.cpu().numpy(),
                               predicted.boxes.conf.cpu().numpy(), predicted.boxes.cls.cpu().numpy()))]
            item = dict(id=image_id, split=split, image=destination, source_name=path.name,
                        width=width, height=height, image_sha256=image_hash, label_sha256=label_hash,
                        ground_truth=truth, detections=detections, detector_seconds=seconds)
            result['images'].append(item)
            if (index+1)%25 == 0 or index+1==len(paths):
                print(json.dumps(dict(stage='detector_cache', split=split, image=index+1, images=len(paths)), ensure_ascii=False), flush=True)
        items = [im for im in result['images'] if im['split']==split]
        split_counts[split] = dict(images=len(items), instances=sum(len(im['ground_truth']) for im in items),
                                  candidates=sum(len(im['detections']) for im in items),
                                  detector_matches=sum(len(match_boxes(im['detections'],im['ground_truth'])) for im in items))
    assert (split_counts['train']['images'],split_counts['train']['instances']) == (379,1011)
    assert (split_counts['val']['images'],split_counts['val']['instances']) == (81,221)
    result['counts'] = split_counts
    result['preparation_seconds'] = time.time()-start
    save_json(out/'manifest.json', result)
    (out/'dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/val\nnames:\n  0: shrimp\nkpt_shape: [15, 3]\nflip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\n', encoding='utf-8')
    save_json(out/'data_audit.json', dict(counts=split_counts, manifest_sha256=sha256(out/'manifest.json'),
              no_test_exported=True, midpoint_and_physical_length_checks='PASS', exact_image_split_leakage='none'))
    print(json.dumps(split_counts), flush=True)


if __name__ == '__main__':
    main()
