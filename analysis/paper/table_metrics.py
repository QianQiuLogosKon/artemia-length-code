"""Recompute the length-error summaries reported in the paper tables.

Reads per-individual evaluation records written by the evaluation scripts and
writes two JSON files used by make_review_figures.py:
  output/paper_revision_audit_20261013/rmse_recalculation.json
      MAPE/MAE/RMSE per method at detector thresholds 0.001 (Table 2) and 0.7 (Table 5)
  output/paper_revision_audit_20261013/component_metrics.json
      fixed-weight component analysis (Fig. 6)
Ground truth is used only for offline scoring.
"""
from pathlib import Path
import csv, hashlib, json, math

R = Path(__file__).resolve().parents[2]
O = R / "output/paper_revision_audit_20261013"
O.mkdir(parents=True, exist_ok=True)
read = lambda p: json.loads(p.read_text(encoding="utf-8"))


def summary(z, key_ape="ape_percent", key_ae="ae_um"):
    n = len(z)
    return {"n": n,
            "mape": sum(float(r[key_ape]) for r in z) / n,
            "mae": sum(float(r[key_ae]) for r in z) / n,
            "rmse": math.sqrt(sum(float(r[key_ae]) ** 2 for r in z) / n)}


# Tables 2 and 5: all methods at the main (0.001) and screening (0.7) thresholds.
fair_rows = read(R / "output/baseline_fairness_20261008/all_gt.json")
fish = read(R / "output/fishkeypoints_hbb_adaptation_20260929/validation_evaluation/all_221_gt_both_lengths.json")
metrics = {}
for t in (.001, .7):
    for method in sorted({r["method"] for r in fair_rows}):
        z = [r for r in fair_rows if r["threshold"] == t and r["method"] == method and r["success"]]
        metrics[f"{t}/{method}"] = summary(z)
metrics["0.001/Fish chord"] = summary([r for r in fish if r["method"] == "chord" and r["status"] == "ok"])
sources = [R / "output/baseline_fairness_20261008/all_gt.json",
           R / "output/fishkeypoints_hbb_adaptation_20260929/validation_evaluation/all_221_gt_both_lengths.json"]
(O / "rmse_recalculation.json").write_text(json.dumps({
    "metrics": metrics,
    "sources": {str(p.relative_to(R)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}},
    indent=2), encoding="utf-8")

# Fig. 6: fixed-weight component analysis on the 218 common successes.
rows = read(R / "output/baseline_structure_comparison_20260929/all_gt.json")
entries = []
for label, keys, extra in [("Complete input", ("yellow_raw", "yellow_refined"), 217),
                           ("Prompt removed", ("no_prompt_raw", "no_prompt_refined"), 112),
                           ("Endpoint chord", ("chord_raw", "chord_refined"), 217),
                           ("Padding 0", ("pad_raw", "pad_refined"), 145)]:
    for mode, key in zip(("TCP", "TCP-ER"), keys):
        if key.startswith("pad_"):
            name = ("raw" if mode == "TCP" else "refined") + "_pad_0.00_all_221_gt.csv"
            with (R / "output/padding_ablation_20260927/tables" / name).open(encoding="utf-8-sig") as f:
                z = [r for r in csv.DictReader(f) if r["status"] == "ok"]
        else:
            z = [r for r in rows if r["method"] == key and r["status"] == "ok"]
        assert len(z) == 218
        m = summary(z)
        entries.append(dict(setting=label, mode=mode, n=m["n"], mape=m["mape"], mae=m["mae"], rmse=m["rmse"], extra=extra))
(O / "component_metrics.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
print("wrote rmse_recalculation.json and component_metrics.json")
