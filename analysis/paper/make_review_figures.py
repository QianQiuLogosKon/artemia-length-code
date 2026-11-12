"""Generate revised manuscript figures from frozen, audited artifacts.

No new inference, no test access. Only layout/annotation changes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

R = Path(__file__).resolve().parents[2]
OUT = R / "figures"
PREV = R / "figures/previews"
OUT.mkdir(parents=True, exist_ok=True)
PREV.mkdir(parents=True, exist_ok=True)

for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    font_manager.fontManager.addfont(str(Path("C:/Windows/Fonts") / name))
assert font_manager.FontProperties(fname="C:/Windows/Fonts/times.ttf").get_name() == "Times New Roman"
plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 12,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "savefig.facecolor": "white",
    }
)

MANIFEST = R / "output/baseline_hrnet_20260922_v1/data_v2/manifest.json"
M = json.loads(MANIFEST.read_text(encoding="utf-8"))
IMS = {x["id"]: x for x in M["images"] if x["split"] == "val"}
TRAIN = [x for x in M["images"] if x["split"] == "train"]

RAW = json.loads((R / "output/padding_ablation_20260927/raw_pad_0.15.json").read_text(encoding="utf-8"))
REF = json.loads((R / "output/padding_ablation_20260927/refined_pad_0.15.json").read_text(encoding="utf-8"))
COMP = json.loads((R / "output/paper_revision_audit_20261013/component_metrics.json").read_text(encoding="utf-8"))
STRUCT = json.loads((R / "output/baseline_structure_comparison_20260929/analysis_summary.json").read_text(encoding="utf-8"))
BASE = json.loads((R / "output/baseline_structure_comparison_20260930_v2/analysis_summary.json").read_text(encoding="utf-8"))["methods"]
CS = json.loads((R / "output/baseline_strengthening_20260930_v1/mmpcf_validation_component_snap/analysis_summary.json").read_text(encoding="utf-8"))["methods"]["MMPCF_CS"]
PAD = json.loads((R / "output/padding_ablation_20260927/analysis_summary.json").read_text(encoding="utf-8"))
FAIR = json.loads((R / "output/baseline_fairness_20261008/results.json").read_text(encoding="utf-8"))
CASE = json.loads((R / "output/baseline_fairness_20261008/figure_index.json").read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("wrote", name)


def imread(im: dict) -> np.ndarray:
    p = MANIFEST.parent / im["image"]
    assert p.exists(), p
    return np.asarray(Image.open(p).convert("RGB"))


def arc(points) -> float:
    p = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter) if aa + bb - inter else 0.0


def crop_bounds(box, w, h, pad=0.15):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return (
        max(0, int(math.floor(x1 - dx))),
        max(0, int(math.floor(y1 - dy))),
        min(w, int(math.ceil(x2 + dx))),
        min(h, int(math.ceil(y2 + dy))),
    )


def draw_box(ax, box, color, ls="-", lw=1.8, label=None, off=(0, 0)):
    x1, y1, x2, y2 = box
    ax.add_patch(Rectangle((x1 + off[0], y1 + off[1]), x2 - x1, y2 - y1,
                           fill=False, edgecolor=color, linestyle=ls, linewidth=lw))
    if label:
        ax.text(x1 + off[0] + 4, y1 + off[1] + 4, label, color=color, fontsize=13,
                va="top", ha="left", bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))


# ---------------------------------------------------------------- Figure 1
counts = sorted({len(im["ground_truth"]) for im in TRAIN})
chosen = [next(im for im in TRAIN if len(im["ground_truth"]) == n) for n in (counts[0], counts[len(counts) // 2], counts[-1])]
fig, axs = plt.subplots(2, 3, figsize=(12.6, 7.0), layout="constrained")
for col, im in enumerate(chosen):
    arr = imread(im)
    for row in range(2):
        axs[row, col].imshow(arr)
        axs[row, col].axis("off")
    axs[0, col].set_title("Original image", fontsize=20, pad=8)
    n_gt = len(im["ground_truth"])
    axs[1, col].set_title(f"{n_gt} annotated individual" + ("" if n_gt == 1 else "s"), fontsize=20, pad=8)
    for g in im["ground_truth"]:
        q = np.asarray(g["points"])
        axs[1, col].plot(q[:, 0], q[:, 1], "o-", color="#00a574", markersize=2.2, linewidth=1.4)
save(fig, "Figure_1")

# ---------------------------------------------------------------- Figure 3a
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.set_xlim(0, 9); ax.set_ylim(0, 6.5); ax.axis("off")
box_style = dict(boxstyle="round,pad=0.06", facecolor="#edf3f6", edgecolor="#38546a", linewidth=1.0)
def flow_box(x, y, w, h, text, fs=12):
    ax.add_patch(FancyBboxPatch((x,y),w,h,**box_style))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,linespacing=1.2)
def arrow(x1,y1,x2,y2):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="->",mutation_scale=12,color="#38546a"))
ax.text(.1,6.12,"(a) Data preparation and measurement pipeline",fontsize=14,fontweight="bold")
ax.text(.1,5.68,"Offline preparation",fontsize=12.5,fontweight="bold")
for x,txt in zip([.12,3.16,6.20],["Recorded images\n+ per-individual boxes","Eight ordered manual\nmidline points","Insert seven midpoints\n15-point representation"]):
    flow_box(x,4.65,2.6,.78,txt)
arrow(2.77,5.04,3.08,5.04); arrow(5.81,5.04,6.12,5.04)
ax.text(.12,4.24,"Image-level splits; sealed test excluded. Interpolation adds no manual labels.",fontsize=11.5)
ax.text(.1,3.70,"Online measurement: repeat for every detector candidate",fontsize=12.5,fontweight="bold")
for x,txt in zip([.12,3.16,6.20],["Full image\nDetector candidate box","15% padded crop\nYellow target prompt","Stage 1: TCP\n15 ordered points"]):
    flow_box(x,2.60,2.6,.82,txt)
arrow(2.77,3.01,3.08,3.01);arrow(5.81,3.01,6.12,3.01)
for x,txt in zip([6.20,3.16,.12],["Stage 2: ER\nUpdate two endpoints","Corrected polyline length\n$\\times$ image scale","Physical length\nper valid candidate"]):
    flow_box(x,1.20,2.6,.82,txt)
arrow(7.5,2.54,7.5,2.08);arrow(6.14,1.61,5.82,1.61);arrow(3.10,1.61,2.78,1.61)
ax.text(.12,.60,"Offline only: candidate-reference matching and evaluation.",fontsize=11.5)
ax.text(.12,.22,"Ground-truth boxes and midlines are not inference inputs.",fontsize=11.5)
save(fig,"Figure_3a")

# ---------------------------------------------------------------- Figure 4
fig,ax=plt.subplots(figsize=(9,6.6));ax.set_xlim(0,9);ax.set_ylim(0,6.6);ax.axis("off")
ax.text(.1,6.25,"(a) Normalize a local endpoint patch",fontsize=14,fontweight="bold")
for x,txt in zip([.12,3.16,6.20],["Predicted endpoint\n+ inward guide\n$q_3$ (head), $q_{K-2}$ (tail)","Affine map $A_e$\nEndpoint at (32, 64)\nGuide at (96, 64)","128 x 128 patch\nSix channels:\nRGB, type, x/y"]):
    flow_box(x,4.90,2.6,1.02,txt,12)
arrow(2.77,5.41,3.08,5.41);arrow(5.81,5.41,6.12,5.41)
ax.text(.12,4.47,"Use the original RGB image; skip refinement if guide distance < 2 pixels.",fontsize=11)
ax.text(.1,3.92,"(b) Predict a residual and cap the image-space update",fontsize=14,fontweight="bold")
for x,txt in zip([.12,3.16,6.20],["Five convolution stages\n32, 48, 72, 112, 160\nFC 256 + SiLU + dropout","Two raw values $\\zeta_e$\n$r_e=1.5\\tanh(\\zeta_e)$\nPatch displacement $64r_e$","Inverse affine map\nImage displacement $\\delta_e$\nCap at $0.03L_{px}^{(0)}$"]):
    flow_box(x,2.38,2.6,1.06,txt,12)
arrow(2.77,2.91,3.08,2.91);arrow(5.81,2.91,6.12,2.91)
ax.text(.12,1.80,"$L_{px}^{(0)}$: pre-refinement polyline length, computed once for both endpoints.",fontsize=11.5)
flow_box(.12,.44,8.68,.96,"Output: corrected endpoints; the 13 internal points are unchanged.\nCompute physical length from the corrected polyline (Section 3.4).",12)
arrow(7.5,2.30,7.5,1.48)
save(fig,"Figure_4")

# ---------------------------------------------------------------- Figure 5
entries = [
    ("MMPCF", BASE["MMPCF"]),
    ("MMPCF-CS", CS),
    ("UNet-MS", BASE["MS"]),
    ("UNet-SAG", BASE["SAG"]),
    ("Fish-HBB\n(polyline)", BASE["Fish_HBB_polyline"]),
    ("Fish-HBB\n(chord)", BASE["Fish_HBB_chord"]),
    ("TCP-ER\n(proposed)", BASE["yellow_refined"]),
]
fig, axs = plt.subplots(1, 3, figsize=(9, 6.8), layout="constrained")
yy = np.arange(len(entries))
colors = ["#8899a4"] * 6 + ["#204c70"]
for ax, key, title in zip(
    axs,
    ["n_success", "mape_percent", "extra_valid_outputs"],
    ["Successful individuals\n(out of 221)", "MAPE on\nsuccesses (%)", "Extra outputs"],
):
    vals = [v[key] for _, v in entries]
    ax.barh(yy, vals, color=colors, height=0.62)
    ax.set_title(title, fontsize=19, pad=12)
    ax.set_yticks(yy)
    ax.set_yticklabels([n for n, _ in entries] if ax is axs[0] else [], fontsize=16)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.28)
    ax.tick_params(axis="x", labelsize=13)
    ax.spines[["top", "right"]].set_visible(False)
    for y, v in zip(yy, vals):
        ax.text(v + max(vals) * 0.025, y, f"{v:.3f}" if key == "mape_percent" else str(v),
                va="center", fontsize=14)
save(fig, "Figure_5")

# ---------------------------------------------------------------- Figure 5
order = ["Complete input", "Prompt removed", "Endpoint chord", "Padding 0"]
labels = ["Complete", "w/o prompt", "Chord", "w/o padding"]
colors = {"TCP":"#9aa9b4","TCP-ER":"#204c70"}
vals = {(r["setting"],r["mode"]):r for r in COMP}
fig=plt.figure(figsize=(9,8.8),layout="constrained")
gs=fig.add_gridspec(2,2,height_ratios=[1.05,1])
axs=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])]
for ax,key,title,unit in [(axs[0],"mape","(a) MAPE","%"),(axs[1],"mae","(b) MAE","$\\mu$m")]:
    y=np.arange(4); h=.34
    ax.axhspan(-.43,.43,color="#edf3f6",zorder=0)
    allv=[vals[(st,mo)][key] for st in order for mo in colors]
    mx=max(allv)
    for k,mode in enumerate(["TCP","TCP-ER"]):
        v=[vals[(st,mode)][key] for st in order]
        ax.barh(y+(k-.5)*h,v,height=h,color=colors[mode],label=mode)
        for j,(yy,vv) in enumerate(zip(y+(k-.5)*h,v)):
            ax.text(vv+mx*.025,yy,f"{vv:.3f}" if key=="mape" else f"{vv:.1f}",va="center",fontsize=12.5,fontweight="bold" if j==0 and mode=="TCP-ER" else "normal")
    ax.set_xlim(0,mx*1.42);ax.set_yticks(y);ax.set_yticklabels(labels,fontsize=12)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.invert_yaxis();ax.set_xlabel(f"{title.split(') ')[1]} ({unit})",fontsize=12)
    ax.set_title(title,fontsize=14);ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=12.5);ax.grid(axis="x",alpha=.15)
axs[0].legend(frameon=False,fontsize=12.5,loc="lower right")
endpoint=next(x for x in STRUCT["paired"] if x["a"]=="yellow_raw" and x["b"]=="yellow_refined")
prompt=next(x for x in STRUCT["paired"] if x["a"]=="no_prompt_refined" and x["b"]=="yellow_refined")
chord=next(x for x in STRUCT["paired"] if x["a"]=="chord_refined" and x["b"]=="yellow_refined")
rows=[("TCP (w/o ER)",endpoint["mean_a_minus_b_pp"],endpoint["ci95_image_cluster_pp"]),
      ("w/o prompt + ER",prompt["mean_a_minus_b_pp"],prompt["ci95_image_cluster_pp"]),
      ("Chord + ER",chord["mean_a_minus_b_pp"],chord["ci95_image_cluster_pp"]),
      ("w/o padding + ER",PAD["paired"]["refined"]["mean_ape_pad0_minus_pad015_pp"],PAD["paired"]["refined"]["image_cluster_bootstrap_95_ci_pp"])]
ax=axs[2]
for i,(name,v,ci) in enumerate(rows):
    ax.errorbar(v,3-i,xerr=[[v-ci[0]],[ci[1]-v]],fmt="o",color="#244b68",capsize=4)
    ax.text(v,3-i+.19,f"{v:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]",ha="center",fontsize=12.5)
ax.set_yticks(range(4));ax.set_yticklabels([r[0] for r in rows][::-1],fontsize=12)
ax.set_ylim(-.5,3.6);ax.set_xlim(-1,7.8)
ax.axvline(0,color="#666",ls="--");ax.set_xlabel("Variant APE minus complete TCP-ER APE (percentage points)",fontsize=12)
ax.set_title("(c) Paired differences and 95% image-cluster intervals",fontsize=14)
ax.tick_params(labelsize=12.5);ax.spines[["top","right"]].set_visible(False);ax.grid(axis="x",alpha=.15)
save(fig,"Figure_6")

# ---------------------------------------------------------------- Figures 6 and 7
method_specs = {
    "MMPCF": ((R / "output/mmpcf_adaptation_20260928/predictions_all_candidates.json"), "mmpcf"),
    "MMPCF-CS": ((R / "output/baseline_strengthening_20260930_v1/mmpcf_validation_component_snap/predictions_all_candidates.json"), "mmpcf"),
    "UNet-MS": ((R / "output/unet_ms_sag_20260928/decoder_evaluation_v2/MS_all_candidates.json"), "unet"),
    "UNet-SAG": ((R / "output/unet_ms_sag_20260928/decoder_evaluation_v2/SAG_all_candidates.json"), "unet"),
    "FishKeypoints-HBB": ((R / "output/fishkeypoints_hbb_adaptation_20260929/validation_candidates.json"), "fish"),
    "TCP": ((R / "output/padding_ablation_20260927/raw_pad_0.15.json"), "pose"),
    "TCP-ER": ((R / "output/padding_ablation_20260927/refined_pad_0.15.json"), "pose"),
}
predictions = {}
for name, (path, kind) in method_specs.items():
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        raw = {x["id"]: x for x in raw}
    preds = {}
    for cid, x in raw.items():
        pts = x.get("points", [])
        length = None
        if kind == "unet":
            length = x.get("length_px")
        elif x.get("status") == "ok":
            if kind == "mmpcf":
                length = x.get("native_length_px")
            elif kind == "fish":
                length = x.get("length_polyline_px")
            else:
                length = arc(pts) if len(pts) >= 2 else None
        preds[cid] = {"ok": length is not None and np.isfinite(length) and length > 0, "points": pts}
    predictions[name] = preds


def case_figure(case, name):
    im = IMS[case["image_id"]]
    d = next(x for x in im["detections"] if x["id"] == case["candidate_id"])
    arr = imread(im)
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = d["box"]
    # context crop is deliberately wider than the method crop
    l, t, r, b = crop_bounds(d["box"], w, h, 0.55)
    fig = plt.figure(figsize=(9, 7.7), layout="constrained")
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 0.9, 0.9])
    axc = fig.add_subplot(gs[0, 0:2])
    axc.imshow(arr[t:b, l:r])
    axc.set_xlim(0, r - l)
    axc.set_ylim(b - t, 0)
    axc.axis("off")
    axc.set_title("Matching context", fontsize=17, pad=7)
    # candidate and reference boxes
    draw_box(axc, d["box"], "#d98c00", "-", 2.0, "A", (-l, -t))
    if case["category"] == "duplicate":
        other = next(x for x in im["detections"] if x["id"] == "val_0077:d14")
        draw_box(axc, other["box"], "#1f77b4", "-.", 1.8, "B", (-l, -t))
    for g in im["ground_truth"]:
        draw_box(axc, g["box"], "#00a574", "--", 1.4, None, (-l, -t))
    axi = fig.add_subplot(gs[0, 2:4])
    axi.axis("off")
    if case["category"] == "duplicate":
        txt = ("A: IoU 0.717 with the reference\n"
               "B: IoU 0.886 with the reference\n"
               "One-to-one matching assigns B;\nA remains an unmatched output.\n"
               "Both boxes overlap a real animal.")
    else:
        txt = ("A: IoU 0.496 with the reference\n"
               "Below the 0.5 matching threshold;\nA remains unmatched although\nan animal is visibly present.\n"
               "Matching is offline only.")
    axi.text(0.02, 0.92, txt, va="top", fontsize=12.5, linespacing=1.45)
    # method panels
    l2, t2, r2, b2 = crop_bounds(d["box"], w, h, 0.15)
    axes = []
    for row in (1, 2):
        for col in range(4):
            axes.append(fig.add_subplot(gs[row, col]))
    target_g = max(im["ground_truth"], key=lambda g: iou(d["box"], g["box"]))
    names = ["Reference"] + list(method_specs)
    for ax, namex in zip(axes, names):
        ax.imshow(arr[t2:b2, l2:r2])
        ax.set_xlim(0, r2 - l2)
        ax.set_ylim(b2 - t2, 0)
        ax.axis("off")
        ax.add_patch(Rectangle((x1 - l2, y1 - t2), x2 - x1, y2 - y1, fill=False,
                               edgecolor="#d98c00", linewidth=1.2))
        if namex == "Reference":
            for g in im["ground_truth"]:
                q = np.asarray(g["points"])
                ax.plot(q[:, 0] - l2, q[:, 1] - t2, "--", color="#00ad50", linewidth=1.6)
            ax.set_title("Manual references", fontsize=15, pad=5)
        else:
            for g in im["ground_truth"]:
                q = np.asarray(g["points"])
                ax.plot(q[:, 0] - l2, q[:, 1] - t2, "--" if g is target_g else ":", color="#00ad50" if g is target_g else "#777777", linewidth=1.3)
            v = predictions[namex][d["id"]]
            if v["ok"]:
                q = np.asarray(v["points"])
                ax.plot(q[:, 0] - l2, q[:, 1] - t2, "-", color="#ec3273", linewidth=1.7)
                ax.set_title("Fish-HBB (polyline)" if namex=="FishKeypoints-HBB" else namex, fontsize=12, pad=5)
            else:
                ax.set_title(namex + "\n(no curve)", fontsize=12, pad=5)
    save(fig, name)


case_figure(next(c for c in CASE if c["category"] == "duplicate"), "Figure_7")
case_figure(next(c for c in CASE if c["category"] == "partial"), "Figure_8")

# ---------------------------------------------------------------- Figure 8
im = IMS["val_0003"]
arr = imread(im)
h, w = arr.shape[:2]
fig, axs = plt.subplots(1, 2, figsize=(9, 4.8), layout="constrained")
for ax in axs:
    ax.imshow(arr)
    ax.axis("off")
axs[0].set_title("Manual references", fontsize=18, pad=8)
axs[1].set_title("TCP-ER: all valid outputs", fontsize=14, pad=8)
for g in im["ground_truth"]:
    q = np.asarray(g["points"])
    axs[0].plot(q[:, 0], q[:, 1], "-", color="#00ad50", linewidth=1.8)
valid = []
# Greedy one-to-one matching, used only to identify the representative
# matched curve for each of the five individuals in this display.
pairs = []
for di, d in enumerate(im["detections"]):
    for gi, g in enumerate(im["ground_truth"]):
        v = iou(d["box"], g["box"])
        if v >= 0.5:
            pairs.append((v, di, gi))
pairs.sort(key=lambda z: (-z[0], -z[1], -z[2]))
used_d, used_g, matched = set(), set(), {}
for v, di, gi in pairs:
    if di in used_d or gi in used_g:
        continue
    used_d.add(di)
    used_g.add(gi)
    matched[im["detections"][di]["id"]] = gi
for d in im["detections"]:
    v = predictions["TCP-ER"][d["id"]]
    if not v["ok"]:
        continue
    valid.append(d["id"])
    q = np.asarray(v["points"])
    is_matched = d["id"] in matched
    axs[1].plot(q[:, 0], q[:, 1], "-", color="#ec3273",
                linewidth=1.9 if is_matched else 1.2, alpha=0.95 if is_matched else 0.45)
    if not is_matched:
        continue
    scale = im["ground_truth"][0]["um_per_pixel"]
    length_mm = arc(q) * scale / 1000.0
    k = int(len(q) * 0.58)
    axs[1].annotate(f"{length_mm:.2f} mm", (q[k, 0], q[k, 1]),
                    xytext=(5, 6), textcoords="offset points", fontsize=13, color="#8a1740",
                    bbox=dict(facecolor="white", alpha=0.86, edgecolor="none", pad=1.1))
axs[1].text(0.02, 0.98, f"{len(valid)} valid curves from {len(im['detections'])} fixed candidates\nLabels mark the matched measurement for each individual.",
            transform=axs[1].transAxes, va="top", fontsize=12.5, color="#222222",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=2))
save(fig, "Figure_9")

# ---------------------------------------------------------------- Figure 10
# Reproduce the selection from all matched validation candidates, not a hardcoded ID.
endpoint_cases = []
for image in IMS.values():
    pairs = sorted([(iou(det["box"],gt["box"]),di,gi)
                    for di,det in enumerate(image["detections"])
                    for gi,gt in enumerate(image["ground_truth"])
                    if iou(det["box"],gt["box"]) >= .5], reverse=True)
    ud, ug = set(), set()
    for ov, di, gi in pairs:
        if di in ud or gi in ug: continue
        ud.add(di); ug.add(gi)
        det = image["detections"][di]; cid=det["id"]
        if RAW[cid].get("status")!="ok" or REF[cid].get("status")!="ok": continue
        shift=float(np.linalg.norm(np.asarray(REF[cid]["points"])[[0,-1]]-np.asarray(RAW[cid]["points"])[[0,-1]],axis=1).sum())
        endpoint_cases.append((shift,cid,image,det,image["ground_truth"][gi]))
endpoint_cases.sort(key=lambda row:(-row[0],row[1]))
_, _, im, d, matched_reference = endpoint_cases[0]
assert len(endpoint_cases)==218
(OUT.parent/"endpoint_selection.json").write_text(json.dumps({"rule":"largest total endpoint displacement among valid matched candidates; candidate ID breaks ties", "n":len(endpoint_cases),"selected":d["id"],"shift_px":endpoint_cases[0][0],"second_shift_px":endpoint_cases[1][0]},indent=2),encoding="utf-8")
arr = imread(im)
h, w = arr.shape[:2]
l, t, r, b = crop_bounds(d["box"], w, h, 0.15)
qraw = np.asarray(RAW[d["id"]]["points"])
qref = np.asarray(REF[d["id"]]["points"])
scale = im["ground_truth"][0]["um_per_pixel"]
# Use the reference assigned by the same fixed matching.
g = matched_reference
qgt = np.asarray(g["points"])
fig = plt.figure(figsize=(9, 8.4), layout="constrained")
gs = fig.add_gridspec(2, 2)
ax = fig.add_subplot(gs[0, 0])
ax.imshow(arr[t:b, l:r])
ax.set_xlim(0, r - l)
ax.set_ylim(b - t, 0)
ax.axis("off")
ax.set_title("(a) Full crop", fontsize=17, pad=6)
ax.plot(qgt[:, 0] - l, qgt[:, 1] - t, "--", color="#00ad50", linewidth=1.8, label="Manual")
ax.plot(qraw[:, 0] - l, qraw[:, 1] - t, "-", color="#2b6cb0", linewidth=2.0, label="TCP")
ax.plot(qref[:, 0] - l, qref[:, 1] - t, "-", color="#e07b00", linewidth=2.0, label="TCP-ER")
ax.plot([qraw[0, 0] - l, qref[0, 0] - l], [qraw[0, 1] - t, qref[0, 1] - t], ":", color="#333333", linewidth=1.1)
ax.plot([qraw[-1, 0] - l, qref[-1, 0] - l], [qraw[-1, 1] - t, qref[-1, 1] - t], ":", color="#333333", linewidth=1.1)
ax.legend(frameon=False, fontsize=12.5, loc="lower right")
axi = fig.add_subplot(gs[0, 1])
axi.axis("off")
axi.text(0.02, 0.95,
         "Solid blue: TCP (no refinement)\n"
         "Solid orange: TCP-ER\n"
         "Dashed green: manual reference\n\n"
         f"TCP: {arc(qraw) * scale / 1000:.2f} mm\n"
         f"TCP-ER: {arc(qref) * scale / 1000:.2f} mm\n"
         f"Reference: {g['length_um'] / 1000:.2f} mm\n\n"
         "Only the two terminal points move;\nthe 13 internal points are unchanged.",
         va="top", fontsize=12.5, linespacing=1.5)
for idx, (axpos, label, endidx) in enumerate([
    (gs[1, 0], "(b) Head", 0),
    (gs[1, 1], "(c) Tail", -1),
]):
    axz = fig.add_subplot(axpos)
    e = qref[endidx]
    half = 300
    x0, x1 = max(0, int(e[0] - half)), min(w, int(e[0] + half))
    y0, y1 = max(0, int(e[1] - half)), min(h, int(e[1] + half))
    axz.imshow(arr[y0:y1, x0:x1])
    axz.set_xlim(0, x1 - x0)
    axz.set_ylim(y1 - y0, 0)
    axz.axis("off")
    axz.set_title(label, fontsize=16, pad=5)
    axz.plot(qgt[:, 0] - x0, qgt[:, 1] - y0, "--", color="#00ad50", linewidth=1.6)
    axz.plot(qraw[:, 0] - x0, qraw[:, 1] - y0, "-", color="#2b6cb0", linewidth=1.8)
    axz.plot(qref[:, 0] - x0, qref[:, 1] - y0, "-", color="#e07b00", linewidth=1.8)
    for q, col in ((qraw, "#2b6cb0"), (qref, "#e07b00")):
        axz.plot(q[endidx, 0] - x0, q[endidx, 1] - y0, "o", color=col, markersize=8)
    shift = float(np.linalg.norm(qref[endidx] - qraw[endidx]))
    axz.annotate("", xy=(qref[endidx, 0] - x0, qref[endidx, 1] - y0),
                 xytext=(qraw[endidx, 0] - x0, qraw[endidx, 1] - y0),
                 arrowprops=dict(arrowstyle="->", color="#333333", linewidth=1.1))
    axz.text(0.03, 0.04, f"shift {shift:.1f} px", transform=axz.transAxes, fontsize=12,
             bbox=dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.2))
save(fig, "Figure_10")

print("all figures done")

