"""Fine-tune the reviewed YOLO11s hybrid model with an intrinsic curve continuity penalty."""
from pathlib import Path
import argparse, csv, yaml, torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8PoseLoss

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "brine_shrimp_yolo_pose_archive_15pt_headtail_fully_reviewed_crops_targetclass_jitter"
INIT = ROOT / "runs/brine_yolo11s_hybrid15_targetclass_weighted_full/weights/best.pt"
RUNS = ROOT / "runs"
MYAML = ROOT / "yolo11s-pose-15pt-continuity.yaml"
DYAML = DATA / "continuity_weighted.yaml"

def setup():
    model = YOLO(str(INIT))
    MYAML.write_text(yaml.safe_dump(dict(model.model.yaml), sort_keys=False), encoding="utf-8")
    rows = list(csv.DictReader((DATA / "metadata/targets.csv").open(encoding="utf-8-sig")))
    lines=[]
    for row in rows:
        if row["split"] == "train" and "_jit" not in Path(row["image"]).stem:
            lines.extend([str(DATA / row["image"])] * (3 if int(row["neighbour_instance_count"]) > 0 else 1))
    (DATA / "continuity_weighted_train.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    DYAML.write_text(f"path: {DATA.as_posix()}\ntrain: {(DATA/'continuity_weighted_train.txt').as_posix()}\nval: images/val\n\nnames:\n  0: target_shrimp\n  1: neighbour_shrimp\nkpt_shape: [15, 3]\nflip_idx: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]\n", encoding="utf-8")
    return DYAML

ORIGINAL = v8PoseLoss.calculate_keypoints_loss
def continuity_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts):
    loc, obj = ORIGINAL(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts)
    if masks.any():
        selected = self._select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks)
        selected[..., :2] /= stride_tensor.view(1, -1, 1, 1)
        gt = selected[masks]
        pred = pred_kpts[masks]
        visible = gt[..., 2] != 0 if gt.shape[-1] == 3 else torch.ones_like(gt[..., 0], dtype=torch.bool)
        pair_mask = visible[:, 1:] & visible[:, :-1]
        pred_seg = pred[:, 1:, :2] - pred[:, :-1, :2]
        gt_seg = gt[:, 1:, :2] - gt[:, :-1, :2]
        gt_lengths = gt_seg.norm(dim=-1)
        scale = gt_lengths.masked_fill(~pair_mask, float("nan")).nanmedian(dim=1).values.clamp_min(0.25)
        normalized_error = (pred_seg - gt_seg) / scale[:, None, None]
        if pair_mask.any():
            continuity = F.smooth_l1_loss(
                normalized_error[pair_mask], torch.zeros_like(normalized_error[pair_mask]), beta=0.5
            )
            loc = loc + 0.10 * continuity
    return loc, obj
v8PoseLoss.calculate_keypoints_loss = continuity_loss

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--smoke", action="store_true"); args=parser.parse_args()
    data=setup(); model=YOLO(str(INIT))
    model.train(data=str(data), epochs=3 if args.smoke else 60, imgsz=1024, batch=4, device=0, workers=4, amp=True,
        optimizer="AdamW", cos_lr=True, fliplr=0.5, flipud=0.0, degrees=5.0, translate=0.03,
        scale=0.08, mosaic=0.0, mixup=0.0, erasing=0.0, pose=15.0, kobj=1.0,
        lr0=0.00015, lrf=0.05, warmup_epochs=1 if args.smoke else 2, patience=20, save_period=5,
        seed=20260916, project=str(RUNS), name="brine_yolo11s_hybrid15_continuity_smoke" if args.smoke else "brine_yolo11s_hybrid15_continuity", exist_ok=False)

if __name__ == "__main__": main()
