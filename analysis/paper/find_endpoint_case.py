import json, pathlib, math

R = pathlib.Path(__file__).resolve().parents[2]
M = json.loads((R / "output/baseline_hrnet_20260922_v1/data_v2/manifest.json").read_text(encoding="utf-8"))
ims = [x for x in M["images"] if x["split"] == "val"]
raw = json.loads((R / "output/padding_ablation_20260927/raw_pad_0.15.json").read_text(encoding="utf-8"))
ref = json.loads((R / "output/padding_ablation_20260927/refined_pad_0.15.json").read_text(encoding="utf-8"))


def arc(p):
    return sum(math.dist(p[i], p[i + 1]) for i in range(len(p) - 1))


rows = []
for im in ims:
    for d in im["detections"]:
        cid = d["id"]
        a, b = raw.get(cid, {}), ref.get(cid, {})
        if a.get("status") != "ok" or b.get("status") != "ok":
            continue
        shifts = a.get("endpoint_shifts_px") or [0, 0]
        rows.append((sum(shifts), cid, im["id"], d["id"], arc(a["points"]), arc(b["points"]), a.get("pose_confidence")))
rows.sort(reverse=True)
for r in rows[:15]:
    print(r)
