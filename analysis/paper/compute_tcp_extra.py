import json, pathlib, statistics

R = pathlib.Path(__file__).resolve().parents[2]
M = json.loads((R / "output/baseline_hrnet_20260922_v1/data_v2/manifest.json").read_text(encoding="utf-8"))
ims = [x for x in M["images"] if x["split"] == "val"]


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter else 0


def match_boxes(dets, gts, thr=0.5):
    pairs = []
    for di, d in enumerate(dets):
        for gi, g in enumerate(gts):
            v = iou(d["box"], g["box"])
            if v >= thr:
                pairs.append((v, di, gi))
    pairs.sort(key=lambda z: (-z[0], -z[1], -z[2]))
    used_d, used_g, out = set(), set(), []
    for v, di, gi in pairs:
        if di in used_d or gi in used_g:
            continue
        used_d.add(di)
        used_g.add(gi)
        out.append((di, gi, v))
    return out


for name, path in {
    "TCP": R / "output/padding_ablation_20260927/raw_pad_0.15.json",
    "TCP-ER": R / "output/padding_ablation_20260927/refined_pad_0.15.json",
}.items():
    preds = json.loads(path.read_text(encoding="utf-8"))
    extras, vals, iou_low = [], [], 0
    for im in ims:
        pairs = match_boxes(im["detections"], im["ground_truth"])
        used = {di for di, _, _ in pairs}
        for di, d in enumerate(im["detections"]):
            if di in used:
                continue
            p = preds[d["id"]]
            if p.get("status") != "ok":
                continue
            extras.append(d["id"])
            pc = p.get("pose_confidence")
            pi = p.get("pose_reference_iou")
            if pc is not None:
                vals.append(pc)
            if pi is not None and pi < 0.5:
                iou_low += 1
    print(name, "extras", len(extras), "median pose", statistics.median(vals),
          "poseIoU<0.5", iou_low, "n", len(vals))
