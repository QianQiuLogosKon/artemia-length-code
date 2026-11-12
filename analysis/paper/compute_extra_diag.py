import json, pathlib, statistics

R = pathlib.Path(__file__).resolve().parents[2]
M = json.loads((R / "output/baseline_hrnet_20260922_v1/data_v2/manifest.json").read_text(encoding="utf-8"))
ims = {x["id"]: x for x in M["images"] if x["split"] == "val"}
det = {}
for im in ims.values():
    for d in im["detections"]:
        cid = im["id"] + ":d" + d["id"].split(":d")[-1]
        det[cid] = d

F = json.loads((R / "output/baseline_fairness_20261008/results.json").read_text(encoding="utf-8"))["conditions"]["0.001"]["methods"]
paths = {
    "UNet-MS": R / "output/unet_ms_sag_20260928/decoder_evaluation_v2/MS_all_candidates.json",
    "UNet-SAG": R / "output/unet_ms_sag_20260928/decoder_evaluation_v2/SAG_all_candidates.json",
    "FishKeypoints-HBB": R / "output/fishkeypoints_hbb_adaptation_20260929/validation_candidates.json",
    "TCP": R / "output/padding_ablation_20260927/raw_pad_0.15.json",
    "TCP-ER": R / "output/padding_ablation_20260927/refined_pad_0.15.json",
}
for name, p in paths.items():
    key = {"FishKeypoints-HBB": "Fish-HBB", "TCP": "Main raw", "TCP-ER": "Main refined"}.get(name, name)
    extras = F[key]["extras"]
    d = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(d, list):
        d = {x["id"]: x for x in d}
    vals, low, iou_low = [], 0, 0
    for cid in extras:
        x = d.get(cid)
        if not x:
            continue
        c = det.get(cid, {}).get("confidence")
        if c is not None and c < 0.02:
            low += 1
        pc = x.get("pose_confidence")
        pi = x.get("pose_reference_iou")
        if pc is not None:
            vals.append(pc)
        if pi is not None and pi < 0.5:
            iou_low += 1
    print(name, "extras", len(extras), "det<0.02", low,
          "median pose", statistics.median(vals) if vals else None,
          "poseIoU<0.5", iou_low, "validconf", len(vals))
