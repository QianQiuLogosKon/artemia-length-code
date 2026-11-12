"""Evaluate completed padding inference; all GT use is offline in this file."""
from pathlib import Path
from collections import Counter
import csv, json, sys, shutil
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
BASE=ROOT/'output/baseline_hrnet_20260922_v1'
DATA=BASE/'data_v2'
OLD=ROOT/'output/baseline_comparison_20260922_v1'
ARCHIVE=BASE
sys.dont_write_bytecode=True
sys.path.insert(0,str(ARCHIVE/'src'))
from common import evaluate, match_boxes, crop_bounds, arc, sha256, iou

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj): Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def table(name,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'tables'/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

def paired(a,b):
    # a = padding 0, b = padding .15, positive delta favors .15.
    aa={r['id']:r for r in a if r['status']=='ok'}
    bb={r['id']:r for r in b if r['status']=='ok'}
    ids=sorted(aa.keys()&bb.keys())
    differences=np.array([aa[k]['ape_percent']-bb[k]['ape_percent'] for k in ids])
    clusters=sorted({r['image'] for r in a})
    counts=np.array([sum(aa[k]['image']==im for k in ids) for im in clusters])
    sums=np.array([sum(aa[k]['ape_percent']-bb[k]['ape_percent'] for k in ids if aa[k]['image']==im) for im in clusters])
    rng=np.random.default_rng(20260927); draw=rng.integers(0,len(clusters),(10000,len(clusters)))
    denominators=counts[draw].sum(1); valid=denominators>0
    means=sums[draw].sum(1)[valid]/denominators[valid]
    result=dict(n_common=len(ids),n_image_clusters=len(clusters),image_clusters_with_common_success=int((counts>0).sum()),
        mean_ape_pad0_minus_pad015_pp=float(differences.mean()),median_paired_ape_difference_pp=float(np.median(differences)),
        image_cluster_bootstrap_95_ci_pp=np.quantile(means,[.025,.975]).tolist(),bootstrap_replicates=10000,seed=20260927,
        pad015_better=int((differences>1e-9).sum()),pad0_better=int((differences< -1e-9).sum()),ties=int((abs(differences)<=1e-9).sum()),
        lost_at_pad015=sorted(aa.keys()-bb.keys()),gained_at_pad015=sorted(bb.keys()-aa.keys()),
        same_successful_gt_set=aa.keys()==bb.keys(),changed_successful_pairings=[k for k in ids if aa[k]['candidate_id']!=bb[k]['candidate_id']],
        interpretation='Descriptive development-validation interval; checkpoints and padding .15 were previously selected on validation; not independent test evidence.')
    for tag,rr in [('pad0',aa),('pad015',bb)]:
        result[tag+'_common_success']=dict(mape_percent=float(np.mean([rr[k]['ape_percent'] for k in ids])),
            median_ape_percent=float(np.median([rr[k]['ape_percent'] for k in ids])),
            mae_um=float(np.mean([rr[k]['ae_um'] for k in ids])))
    return result

def main():
    (OUT/'tables').mkdir(exist_ok=True)
    manifest=read(DATA/'manifest.json'); val={**manifest,'images':[im for im in manifest['images'] if im['split']=='val']}
    assert len(val['images'])==81 and sum(len(im['ground_truth']) for im in val['images'])==221
    old_protocol=read(OLD/'protocol.json')
    assert sha256(DATA/'manifest.json')==old_protocol['data_manifest_sha256']
    names=['raw_pad_0.00','refined_pad_0.00','raw_pad_0.15','refined_pad_0.15']
    pred={name:read(OUT/(name+'.json')) for name in names}
    candidate_ids=[d['id'] for im in val['images'] for d in im['detections']]
    assert len(set(candidate_ids))==1498
    assert all(list(p)==candidate_ids for p in pred.values())
    summaries={}; rows={}; extras={}; audit_checks={}
    for name,p in pred.items():
        summaries[name],rows[name],extras[name]=evaluate(val,p)
        assert len(rows[name])==221
        summaries[name]['status_counts']=dict(Counter(v['status'] for v in p.values()))
        assert all(v.get('pose_detection_count')==0 for v in p.values() if v['status']=='no_target_class_pose')
        table(name+'_all_221_gt.csv',rows[name])
        table(name+'_extra_outputs.csv',[dict(candidate_id=k) for k in extras[name]],['candidate_id'])
    for pad in ['0.00','0.15']:
        for cid in candidate_ids:
            raw=pred['raw_pad_'+pad][cid]; ref=pred['refined_pad_'+pad][cid]
            assert raw['status']==ref['status']
            if raw['status']=='ok':
                p=np.asarray(raw['points']); q=np.asarray(ref['points'])
                assert np.array_equal(p[1:14],q[1:14])
                movement=np.linalg.norm(q[[0,14]]-p[[0,14]],axis=1)
                assert np.all(movement<=.03*arc(p)+1e-8)
    pairing={im['ground_truth'][gi]['id']:im['detections'][di]['id'] for im in val['images'] for di,gi,_ in match_boxes(im['detections'],im['ground_truth'])}
    assert len(pairing)==218
    for rr in rows.values():
        assert all(r.get('candidate_id')==pairing.get(r['id']) for r in rr)
    comparison={kind:paired(rows[kind+'_pad_0.00'],rows[kind+'_pad_0.15']) for kind in ['raw','refined']}
    maps={n:{r['id']:r for r in rr} for n,rr in rows.items()}
    joined=[]
    for im in val['images']:
        for g in im['ground_truth']:
            row=dict(id=g['id'],image_id=im['id'],image=im['source_name'],instance=g['instance'],true_um=g['length_um'])
            for name in names:
                r=maps[name][g['id']]
                for k,v in r.items():
                    if k not in ['id','image','instance','true_um']: row[name+'_'+k]=v
            for kind in ['raw','refined']:
                a=maps[kind+'_pad_0.00'][g['id']]; b=maps[kind+'_pad_0.15'][g['id']]
                row[kind+'_success_change']='both' if a['status']==b['status']=='ok' else 'gained_at_pad015' if b['status']=='ok' else 'lost_at_pad015' if a['status']=='ok' else 'neither'
                if a['status']==b['status']=='ok':row[kind+'_ape_pad0_minus_pad015_pp']=a['ape_percent']-b['ape_percent']
            joined.append(row)
    table('paired_all_221_gt.csv',joined); save(OUT/'paired_all_221_gt.json',joined)
    table('coverage_changes.csv',[r for r in joined if r['raw_success_change'] not in ['both','neither']],list(joined[0]))
    table('all_failed_gt.csv',[r for r in joined if r['raw_success_change']!='both'],list(dict.fromkeys(k for r in joined for k in r)))
    candidate_rows=[]; inputs={str(DATA/'manifest.json'):sha256(DATA/'manifest.json')}
    for im in val['images']:
        p=DATA/im['image']; assert p.resolve().is_relative_to((DATA/'images/val').resolve()); assert sha256(p)==im['image_sha256']; inputs[str(p)]=sha256(p)
        matches={di:gi for di,gi,_ in match_boxes(im['detections'],im['ground_truth'])}
        for di,d in enumerate(im['detections']):
            max_iou=max(iou(d['box'],g['box']) for g in im['ground_truth'])
            cat='matched' if di in matches else 'duplicate_iou_ge_0.5' if max_iou>=.5 else 'partial_overlap_iou_0_to_0.5' if max_iou>0 else 'no_gt_box_overlap'
            r=dict(candidate_id=d['id'],image_id=im['id'],detector_confidence=d['confidence'],geometry_category=cat,
                matched_gt=im['ground_truth'][matches[di]]['id'] if di in matches else '',max_gt_iou=max_iou,
                detector_box=json.dumps(d['box']),crop_pad0=json.dumps(crop_bounds(d['box'],im['width'],im['height'],0)),crop_pad015=json.dumps(crop_bounds(d['box'],im['width'],im['height'],.15)))
            for name in names:r[name+'_status']=pred[name][d['id']]['status']
            candidate_rows.append(r)
    table('all_1498_candidates.csv',candidate_rows)
    extra_groups={n:dict(Counter(r['geometry_category'] for r in candidate_rows if r['candidate_id'] in set(extras[n]))) for n in names}
    summary_rows=[dict(method=n,**{k:v for k,v in s.items() if k!='status_counts'}) for n,s in summaries.items()]
    table('summary.csv',summary_rows)
    old_diff={}
    for kind in ['raw','refined']:
        previous=read(OLD/'inference'/f'yolo_{kind}_all_candidates.json'); current=pred[kind+'_pad_0.15']
        state_diff=[cid for cid in candidate_ids if previous[cid]['status']!=current[cid]['status']]
        changes=[float(np.abs(np.asarray(previous[cid]['points'])-np.asarray(current[cid]['points'])).max()) for cid in candidate_ids if current[cid]['status']==previous[cid]['status']=='ok']
        length_deltas=[abs(arc(previous[r['candidate_id']]['points'])-arc(current[r['candidate_id']]['points']))*g['um_per_pixel'] for im in val['images'] for g in im['ground_truth'] for r in [maps[kind+'_pad_0.15'][g['id']]] if r['status']=='ok' and previous[r['candidate_id']]['status']=='ok']
        old_diff[kind]=dict(status_change_ids=state_diff,maximum_coordinate_difference_px=max(changes),maximum_length_difference_um=max(length_deltas),mean_length_difference_um=float(np.mean(length_deltas)))
    protocol=read(OUT/'protocol.json')
    for n,expected in old_protocol['weight_sha256'].items():
        p=ROOT/'deployment/endpoint_refined_v1/weights'/n; inputs[str(p)]=sha256(p); assert inputs[str(p)]==expected
    protocol.update(data_manifest_sha256=sha256(DATA/'manifest.json'),detector=old_protocol['detector'],primary_threshold=.001,
        endpoint_refiner=dict(epoch=10,endpoints=[1,15],alpha=1.,cap_fraction=.03,batch_size=1,degenerate_patch='retain original endpoint'),
        paired_execution='Both paddings in one process, same loaded Pose and refiner, image/candidate order, FP32 and endpoint batch 1.',
        reference_rerun_reason='Old reference used refiner batch 128. Re-run both arms once in the same process using endpoint batch 1 to avoid candidate-dependent batch composition; preserve old results.',
        scope='Sensitivity of selected fixed weights, not separately trained padding models.',
        protocol_record_timing='Settings fixed in executable source before inference; JSON metadata materialized at completion and supplemented by saved-output audit; not an independently preregistered study.')
    save(OUT/'protocol.json',protocol)
    sourcefiles=[ROOT/'run_padding_ablation_20260927.py',ROOT/'train_endpoint_residual_refiner.py',ROOT/'evaluate_endpoint_residual_refiner.py',ARCHIVE/'src/common.py',OLD/'infer_yolo_candidates.py']
    for p in sourcefiles:inputs[str(p)]=sha256(p)
    (OUT/'source').mkdir(exist_ok=True)
    for p in sourcefiles:shutil.copy2(p,OUT/'source'/p.name)
    save(OUT/'input_sha256.json',inputs)
    result=dict(primary=summaries,paired=comparison,extra_output_geometry=extra_groups,reference_vs_previous=old_diff,
        all_detector_pairings_identical=True,original_detector_unmatched=[r['id'] for r in rows['raw_pad_0.00'] if r['status']=='detector_unmatched_at_iou_0.5'],
        gt_count=221,image_count=81,candidate_count=1498,training=False,fixed_test_images_labels_or_inference_accessed=False,
        limitation='Development validation used in prior selection. Historical test metrics in old reports were not re-evaluated.')
    save(OUT/'analysis_summary.json',result)
    save(OUT/'verification.json',dict(status='PASS',candidate_ids_and_order_identical=True,all_four_groups_1498_candidates=True,
        each_group_221_gt_rows=True,raw_refined_statuses_identical=True,all_internal_13_points_unchanged=True,
        endpoint_cap_respected=True,all_detector_pairings_identical=True,all_empty_pose_records_zero_detections=True,
        validation_image_hashes_checked=81,weights_match_original_protocol=True,manifest_hash_matches=True,
        metrics_computed_from_saved_points=True,visual_review='pending',fixed_test_images_labels_or_inference_accessed=False))
    print(json.dumps(dict(primary=summaries,paired=comparison,reference_vs_previous=old_diff),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
