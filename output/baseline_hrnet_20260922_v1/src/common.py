"""Shared, train/validation-only geometry and full-denominator evaluation."""
from pathlib import Path
import copy
import hashlib
import json
import os
import cv2
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def read_image(path):
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Unreadable image: {path}')
    return image


def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return float(intersection / union) if union > 0 else 0.


def match_boxes(detections, truth, threshold=.5):
    # Preserve the archived global-greedy IoU ordering, including index tie breaks.
    pairs = sorted(((iou(d['box'], g['box']), di, gi) for di, d in enumerate(detections)
                    for gi, g in enumerate(truth)), reverse=True)
    used_d, used_g, matches = set(), set(), []
    for overlap, di, gi in pairs:
        if overlap >= threshold and di not in used_d and gi not in used_g:
            matches.append((di, gi, overlap))
            used_d.add(di)
            used_g.add(gi)
    return matches


def crop_bounds(box, width, height, padding=.15):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return [max(0, int(np.floor(x1 - dx))), max(0, int(np.floor(y1 - dy))),
            min(width, int(np.ceil(x2 + dx))), min(height, int(np.ceil(y2 + dy)))]


def prompt(image, reference):
    x1, y1, x2, y2 = np.round(reference).astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1] - 1, x2), min(image.shape[0] - 1, y2)
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
    result = cv2.addWeighted(overlay, .16, image, .84, 0)
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 3)
    return result


def apply_affine(points, affine):
    points = np.asarray(points, dtype=np.float64)
    return points @ np.asarray(affine)[:, :2].T + np.asarray(affine)[:, 2]


def make_input(image, box, size=512, padding=.15, visual_prompt=True, crop_box=None):
    """Pixel-center-aware letterbox. Matrix maps ORIGINAL pixels to input pixels."""
    height, width = image.shape[:2]
    left, top, right, bottom = crop_bounds(box if crop_box is None else crop_box, width, height, padding)
    if right <= left or bottom <= top:
        raise ValueError('empty_crop')
    crop = image[top:bottom, left:right]
    if visual_prompt:
        crop = prompt(crop, np.asarray(box) - [left, top, left, top])
    s = size / max(right - left, bottom - top)
    px = (size - (right - left) * s) / 2
    py = (size - (bottom - top) * s) / 2
    local = np.array([[s, 0, px + (s - 1) / 2], [0, s, py + (s - 1) / 2]], np.float64)
    canvas = cv2.warpAffine(crop, local, (size, size), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    original = local.copy()
    original[:, 2] -= original[:, :2] @ [left, top]
    return canvas, original


def apply_train_box_overrides(manifest, overrides):
    """Derived TRAINING adaptation only. Never mutate the cached source manifest."""
    result=copy.deepcopy(manifest)
    targets={g['id']:g for im in result['images'] if im['split']=='train' for g in im['ground_truth']}
    assert overrides['split']=='train'
    for entry in overrides['entries']:
        target=targets[entry['id']]
        assert target['box']==entry['original_box']
        target['box']=entry['training_box']
    return result


def training_crop_box(box, points, width, height, padding=.15):
    """Expose all training supervision if an annotated box clips its own curve."""
    points=np.asarray(points)
    l,t,r,b=crop_bounds(box,width,height,padding)
    if np.all((points >= [l,t]) & (points < [r,b])):
        return box
    return [min(box[0],float(points[:,0].min())),min(box[1],float(points[:,1].min())),
            max(box[2],float(points[:,0].max())+1),max(box[3],float(points[:,1].max())+1)]


def heatmaps(points_input, size=128, stride=4, sigma=2.):
    centers = (np.asarray(points_input) + .5) / stride - .5
    yy, xx = np.mgrid[:size, :size]
    target = np.exp(-((xx[None] - centers[:, 0, None, None]) ** 2
                      + (yy[None] - centers[:, 1, None, None]) ** 2) / (2 * sigma ** 2))
    # Input pixel 0 maps to heatmap -0.375, still inside the edge cell.
    # Retain these partially clipped Gaussian targets rather than dropping endpoints.
    valid = ((centers[:, 0] >= -.5) & (centers[:, 0] <= size - .5)
             & (centers[:, 1] >= -.5) & (centers[:, 1] <= size - .5))
    return target.astype(np.float32), valid.astype(np.float32)


def decode_heatmaps(maps, stride=4):
    """Argmax plus bounded local log-quadratic interpolation; no GT input."""
    maps = np.asarray(maps, dtype=np.float64)
    points, peaks = [], []
    for h in maps:
        y, x = np.unravel_index(np.argmax(h), h.shape)
        peak = float(h[y, x])
        offset = np.zeros(2)
        if 0 < x < h.shape[1] - 1 and 0 < y < h.shape[0] - 1 and peak > 1e-8:
            z = np.log(np.maximum(h[y-1:y+2, x-1:x+2], 1e-10))
            gradient = np.array([(z[1, 2] - z[1, 0]) / 2, (z[2, 1] - z[0, 1]) / 2])
            cross = (z[2, 2] - z[2, 0] - z[0, 2] + z[0, 0]) / 4
            hessian = np.array([[z[1, 2] - 2*z[1, 1] + z[1, 0], cross],
                                [cross, z[2, 1] - 2*z[1, 1] + z[0, 1]]])
            if np.linalg.det(hessian) > 1e-9 and np.trace(hessian) < 0:
                proposed = -np.linalg.solve(hessian, gradient)
                if np.isfinite(proposed).all() and np.max(np.abs(proposed)) <= 1:
                    offset = proposed
        points.append((np.array([x, y]) + offset + .5) * stride - .5)
        peaks.append(peak)
    return np.array(points), np.array(peaks)


def arc(points):
    return float(np.linalg.norm(np.diff(np.asarray(points, np.float64), axis=0), axis=1).sum())


def evaluate(manifest, predictions):
    """Match offline only; every GT and every post-NMS proposal remains accountable."""
    rows, false_positive_ids = [], []
    detector_count = detector_matches = valid_outputs = failed_candidates = 0
    expected_ids = {d['id'] for im in manifest['images'] if im['split']=='val' for d in im['detections']}
    if set(predictions)-expected_ids:
        raise ValueError('Unexpected validation candidate IDs')
    # Check unmatched outputs too, otherwise malformed false positives can look valid.
    for candidate_id,p in predictions.items():
        if p.get('status')=='ok':
            points=np.asarray(p.get('points'),dtype=np.float64)
            if points.shape!=(15,2) or not np.isfinite(points).all() or arc(points)<=0:
                raise ValueError(f'Invalid prediction marked ok: {candidate_id}')
    for image in manifest['images']:
        if image['split'] != 'val':
            continue
        detections, truth = image['detections'], image['ground_truth']
        pairs = match_boxes(detections, truth)
        by_gt = {gi: (di, overlap) for di, gi, overlap in pairs}
        matched_di = {di for di, _, _ in pairs}
        detector_count += len(detections)
        detector_matches += len(pairs)
        for di, detection in enumerate(detections):
            p = predictions.get(detection['id'])
            valid = p is not None and p.get('status') == 'ok'
            valid_outputs += int(valid)
            failed_candidates += int(not valid)
            if valid and di not in matched_di:
                false_positive_ids.append(detection['id'])
        for gi, gt in enumerate(truth):
            row = dict(id=gt['id'], image=image['source_name'], instance=gt['instance'], true_um=gt['length_um'])
            if gi not in by_gt:
                best = max((iou(d['box'], gt['box']) for d in detections), default=0.)
                row.update(status='detector_unmatched_at_iou_0.5', best_detector_iou=best)
            else:
                di, overlap = by_gt[gi]
                p = predictions.get(detections[di]['id'])
                row.update(candidate_id=detections[di]['id'], detector_iou=overlap)
                if p is None or p.get('status') != 'ok':
                    row.update(status=p.get('status', 'missing_prediction') if p else 'missing_prediction')
                else:
                    points = np.asarray(p['points'], np.float64)
                    expected = np.asarray(gt['points'], np.float64)
                    if points.shape != (15, 2) or not np.isfinite(points).all() or arc(points) <= 0:
                        raise ValueError('Invalid prediction marked ok')
                    pred_um = arc(points) * gt['um_per_pixel']
                    direct = np.linalg.norm(points[[0, 14]] - expected[[0, 14]], axis=1).sum()
                    reverse = np.linalg.norm(points[[0, 14]] - expected[[14, 0]], axis=1).sum()
                    row.update(status='ok', predicted_um=pred_um, ae_um=abs(pred_um-gt['length_um']),
                               ape_percent=abs(pred_um-gt['length_um']) / gt['length_um'] * 100,
                               direction_correct=bool(direct <= reverse))
            rows.append(row)
    good = [r for r in rows if r['status'] == 'ok']
    ape = np.array([r['ape_percent'] for r in good])
    summary = dict(split='val', ground_truth_instances=len(rows), detector_candidates=detector_count,
                   detector_matched_instances=detector_matches, predicted_instances=len(good),
                   prediction_rate_percent=100*len(good)/len(rows),
                   detector_false_positives=detector_count-detector_matches,
                   valid_candidate_outputs=valid_outputs, failed_candidate_outputs=failed_candidates,
                   valid_output_false_positives=len(false_positive_ids),
                   valid_output_precision=len(good)/valid_outputs if valid_outputs else 0.,
                   mape_percent=float(ape.mean()) if len(ape) else None,
                   median_ape_percent=float(np.median(ape)) if len(ape) else None,
                   mae_um=float(np.mean([r['ae_um'] for r in good])) if good else None,
                   ape_ge_10=int((ape >= 10).sum()), ape_ge_20=int((ape >= 20).sum()),
                   direction_correct=sum(r['direction_correct'] for r in good),
                   direction_accuracy_percent=100*sum(r['direction_correct'] for r in good)/len(good) if good else None)
    return summary, rows, false_positive_ids
