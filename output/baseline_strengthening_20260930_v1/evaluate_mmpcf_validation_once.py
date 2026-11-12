"""One frozen new adaptation evaluation; reuse every unaffected historical output."""
from pathlib import Path
import csv,hashlib,json,time
import cv2,numpy as np
from mmpcf_synthetic_audit import load_exact_functions,SOURCE
from mmpcf_component_snap import joint_component_path
from evaluate_holdout import match
O=Path(__file__).resolve().parent;R=O.parents[1];D=O/'mmpcf_validation_component_snap';B=O.parent/'mmpcf_adaptation_20260928';DATA=O.parent/'baseline_hrnet_20260922_v1/data_v2';read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert read(O/'mmpcf_holdout_evaluation/verification.json')['status'].startswith('PASS');rulehash=sha(O/'mmpcf_component_snap.py');assert rulehash==read(O/'mmpcf_holdout_evaluation/protocol.json')['candidate_sha256'];assert sha(SOURCE)==read(O/'mmpcf_holdout_evaluation/protocol.json')['source_sha256']
    D.mkdir(exist_ok=False);old=read(B/'predictions_all_candidates.json');assert len(old)==1498;todo={cid for cid,p in old.items() if p['status']=='no_path'};assert len(todo)==653
    protocol={'status':'frozen_before_new_development_validation_evaluation','method':'MMPCF-CS: same-component endpoint attachment adaptation','rule_sha256':rulehash,'original_code_sha256':sha(SOURCE),'old_prediction_sha256':sha(B/'predictions_all_candidates.json'),'holdout_result_sha256':sha(O/'mmpcf_holdout_evaluation/analysis_summary.json'),'scope':'81 existing development-validation images,221 GT,1498 fixed detections; fixed test sealed','new_evaluation_count':1,'changed_input_candidates':len(todo),'unchanged_candidates_reused':1498-len(todo),'change':'Only original no_path candidates receive same-component endpoint attachment; originally successful paths and all other failures copied exactly','no_new_thresholds_or_parameter_search':True,'selection_GT':False,'training':False,'neural_inference':False,'test_access':False,'reporting':'Keep original and CS separately; common-success predictions invariant, rescued subset and extras fully reported; no claim of original-author reproduction','visual_holdout_review':'Five cases reviewed: low error still shape-offset; middle case path on lower body; worst143% polynomial extension; retained original unchanged; extra output near real target; no post-review tuning'};save(D/'protocol.json',protocol)
    ns=load_exact_functions();origpath=ns['path_curve'];state={}
    def callback(skel,start,end,silhouette):
        result,diag=joint_component_path(skel,start,end,silhouette,origpath);state.update(diag);return result
    ns['path_curve']=callback;preds=dict(old);records=[];start=time.time();done=0;ims=read(DATA/'manifest.json')['images'];inputs=[{k:im[k] for k in ['id','image','image_sha256','detections']} for im in ims if im['split']=='val'];assert len(inputs)==81
    save(D/'inference_inputs.json',inputs)
    for im in inputs:
        ds=[d for d in im['detections'] if d['id'] in todo]
        if not ds:continue
        src=DATA/im['image'];assert src.resolve().is_relative_to((DATA/'images/val').resolve()) and sha(src)==im['image_sha256'];image=cv2.imdecode(np.fromfile(str(src),np.uint8),1)
        for det in ds:
            state.clear();p=ns['one'](image,np.array(det['box'],float),.15);p.pop('_stages',None);preds[det['id']]=dict(p,component_snap_diagnostic=dict(state));done+=1
            if done%100==0:print('NEW_MMPCF_CS',done,'/653',flush=True)
    assert done==653
    for cid,p in old.items():
        if cid not in todo:assert preds[cid]==p
    save(D/'predictions_all_candidates.json',preds);val=[im for im in ims if im['split']=='val'];mapping={}
    for im in val:mapping.update(match(im))
    used={cid for cid,ov in mapping.values()};rows=[];summaries={}
    for arm,pp in [('MMPCF_original',old),('MMPCF_CS',preds)]:
        extras=[cid for cid,p in pp.items() if p['status']=='ok' and cid not in used]
        for im in val:
            for g in im['ground_truth']:
                cid,ov=mapping.get(g['id'],(None,None));p=pp[cid] if cid else None;ok=p is not None and p['status']=='ok';um=p['native_length_px']*g['um_per_pixel'] if ok else None;ae=abs(um-g['length_um']) if ok else None
                rows.append({'method':arm,'image_id':im['id'],'gt_id':g['id'],'candidate_id':cid,'status':'detector_unmatched' if cid is None else p['status'],'true_um':g['length_um'],'pred_um':um,'ae_um':ae,'ape_percent':100*ae/g['length_um'] if ok else None,'detector_iou':ov})
        good=[r for r in rows if r['method']==arm and r['status']=='ok'];apes=np.array([r['ape_percent'] for r in good]);summaries[arm]={'n_gt':221,'n_success':len(good),'prediction_rate_percent':100*len(good)/221,'mape_percent':float(apes.mean()),'median_ape_percent':float(np.median(apes)),'mae_um':float(np.mean([r['ae_um'] for r in good])),'ape_ge_10':int((apes>=10).sum()),'ape_ge_20':int((apes>=20).sum()),'extra_valid_outputs':len(extras),'extra_ids':extras,'candidate_status_counts':{s:sum(p['status']==s for p in pp.values()) for s in sorted({p['status'] for p in pp.values()})}}
    historical=read(O.parent/'baseline_structure_comparison_20260929/analysis_summary.json')['methods']['MMPCF']
    for k in ['n_success','mape_percent','median_ape_percent','mae_um','ape_ge_10','extra_valid_outputs']:assert np.isclose(summaries['MMPCF_original'][k],historical[k],rtol=1e-10,atol=1e-8),(k,summaries['MMPCF_original'][k],historical[k])
    a={r['gt_id']:r for r in rows if r['method']=='MMPCF_original' and r['status']=='ok'};b={r['gt_id']:r for r in rows if r['method']=='MMPCF_CS' and r['status']=='ok'};assert a.keys()<=b.keys()
    for k in a:assert a[k]['pred_um']==b[k]['pred_um'] and a[k]['candidate_id']==b[k]['candidate_id']
    rescued=[b[k] for k in sorted(b.keys()-a.keys())];main={r['gt_id']:r for r in read(O.parent/'baseline_structure_comparison_20260929/all_gt.json') if r['method']=='yellow_refined' and r['status']=='ok'};keys=sorted(b.keys()&main.keys());diff=[b[k]['ape_percent']-main[k]['ape_percent'] for k in keys]
    ii=[im['id'] for im in val];ss=np.zeros(81);nn=np.zeros(81)
    for k in keys:j=ii.index(b[k]['image_id']);ss[j]+=b[k]['ape_percent']-main[k]['ape_percent'];nn[j]+=1
    rng=np.random.default_rng(20260930);bs=rng.integers(0,81,(10000,81));den=nn[bs].sum(1);valid=den>0;ci=np.percentile(ss[bs].sum(1)[valid]/den[valid],[2.5,97.5]).tolist()
    paired={'old_common_success':len(a),'old_common_predictions_changed':0,'new_success':len(rescued),'new_success_mape_percent':float(np.mean([r['ape_percent'] for r in rescued])) if rescued else None,'lost_gt':[],'versus_current_main':{'n_common':len(keys),'mean_cs_minus_main_ape_pp':float(np.mean(diff)),'ci95_image_cluster_pp':ci,'bootstrap_seed':20260930,'replicates':10000,'changed_candidate_gt':[k for k in keys if b[k]['candidate_id']!=main[k]['candidate_id']],'cs_only_gt':sorted(b.keys()-main.keys()),'main_only_gt':sorted(main.keys()-b.keys())}}
    save(D/'all_gt.json',rows)
    with (D/'all_gt.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    candidate_rows=[{'candidate_id':cid,'original_status':old[cid]['status'],'new_status':p['status'],'newly_computed':cid in todo,'matched_gt_id':next((gid for gid,(did,ov) in mapping.items() if did==cid),None),'extra_valid':p['status']=='ok' and cid not in used,'native_length_px':p.get('native_length_px')} for cid,p in preds.items()]
    with (D/'all_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(candidate_rows[0]));w.writeheader();w.writerows(candidate_rows)
    save(D/'analysis_summary.json',{'methods':summaries,'paired':paired,'scope':'development validation; original and endpoint-attachment adaptation explicitly separated','test_access':False});assert len(rows)==442 and sha(O/'mmpcf_component_snap.py')==rulehash
    save(D/'verification.json',{'status':'PASS_NUMERIC_VISUAL_PENDING','all_candidates':1498,'new_candidate_computations':653,'cached_unaffected_candidates':845,'gt_rows':442,'old_baseline_metrics_reproduced':True,'all_original_success_predictions_unchanged':True,'fixed_detection_matching':True,'protocol_sha256':sha(D/'protocol.json'),'rule_sha256':rulehash,'original_predictions_sha256':sha(B/'predictions_all_candidates.json'),'test_access':False,'neural_inference':False,'training':False,'elapsed_seconds':time.time()-start})
    print(json.dumps({'methods':{k:{a:b for a,b in v.items() if a!='extra_ids'} for k,v in summaries.items()},'paired':paired},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
