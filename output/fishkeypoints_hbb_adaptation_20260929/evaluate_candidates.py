from pathlib import Path
import json,csv,sys,hashlib
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];B=R/'output/baseline_hrnet_20260922_v1';read=lambda p:json.loads(Path(p).read_text(encoding='utf8'));A=B;sys.path.insert(0,str(A/'src'));from common import match_boxes,arc,iou

def evaluate(ims,preds,kind):
 expected={d['id'] for im in ims for d in im['detections']};assert set(preds)==expected
 for p in preds.values():
  if p['status']=='ok':assert np.asarray(p['points']).shape==(15,2) and np.isfinite(p['points']).all() and arc(np.asarray(p['points']))>0
 rows=[];extras=[]
 for im in ims:
  pairs=match_boxes(im['detections'],im['ground_truth']);bygt={gi:(di,v) for di,gi,v in pairs};used={di for di,gi,v in pairs}
  for di,d in enumerate(im['detections']):
   if di not in used and preds[d['id']]['status']=='ok':extras.append(d['id'])
  for gi,g in enumerate(im['ground_truth']):
   r={'method':kind,'gt_id':g['id'],'image_id':im['id'],'true_um':g['length_um'],'candidate_id':None,'status':'detector_unmatched','ape_percent':None,'pred_um':None,'ae_um':None,'direction_correct':None}
   if gi in bygt:
    di,ov=bygt[gi];cid=im['detections'][di]['id'];p=preds[cid];r.update(candidate_id=cid,status=p['status'],detector_iou=ov)
    if p['status']=='ok':
     q=np.asarray(p['points'],float);gt=np.asarray(g['points'],float);length=arc(q) if kind=='polyline' else float(np.linalg.norm(q[0]-q[-1]));um=length*g['um_per_pixel'];ae=abs(um-g['length_um']);r.update(pred_um=um,ae_um=ae,ape_percent=100*ae/g['length_um'],direction_correct=bool(np.linalg.norm(q[[0,14]]-gt[[0,14]],axis=1).sum()<=np.linalg.norm(q[[0,14]]-gt[[14,0]],axis=1).sum()))
   rows.append(r)
 good=[r for r in rows if r['status']=='ok'];ape=[r['ape_percent'] for r in good];summary={'success':len(good),'gt_total':len(rows),'prediction_rate_percent':100*len(good)/len(rows),'mape_percent':float(np.mean(ape)) if good else None,'median_ape_percent':float(np.median(ape)) if good else None,'mae_um':float(np.mean([r['ae_um'] for r in good])) if good else None,'ape_ge_10':sum(v>=10 for v in ape),'ape_ge_20':sum(v>=20 for v in ape),'extra_valid_outputs':len(extras),'direction_correct':sum(r['direction_correct'] for r in good),'direction_success_denominator':len(good)}
 return summary,rows,extras

def main():
 ims=[x for x in read(B/'data_v2/manifest.json')['images'] if x['split']=='val'];dest=O/'validation_evaluation'
 if '--self-check' in sys.argv:
  # Verify no validity-based rematching, and regression against an already completed cache.
  gt={'id':'g','box':[0,0,14,1],'points':[[i,0] for i in range(15)],'length_um':14.,'um_per_pixel':1.};im={'id':'im','ground_truth':[gt],'detections':[{'id':'d0','box':[0,0,14,1]},{'id':'d1','box':[0,0,13,1]}]};pred={'d0':{'status':'empty_pose'},'d1':{'status':'ok','points':gt['points']}}
  s,r,e=evaluate([im],pred,'polyline');assert s['success']==0 and s['extra_valid_outputs']==1 and r[0]['candidate_id']=='d0'
  p=read(R/'output/padding_ablation_20260927/refined_pad_0.15.json');s,r,e=evaluate(ims,p,'polyline');ref=read(R/'output/padding_ablation_20260927/analysis_summary.json')['primary']['refined_pad_0.15'];assert s['success']==ref['predicted_instances'] and s['extra_valid_outputs']==ref['valid_output_false_positives'];assert abs(s['mape_percent']-ref['mape_percent'])<1e-10
  (O/'evaluator_regression.json').write_text(json.dumps({'status':'PASS','no_validity_rematching':True,'reference_rows':len(r),'reference_mape':s['mape_percent'],'reference_extras':len(e),'model_inference':False,'test_access':False},indent=2));print('EVALUATOR_REGRESSION_PASS');return
 # Guard completion and hash agreement before using a newly trained model's predictions.
 archive=read(O/'completed_archive/archive_verification.json');assert archive['status']=='PASS' and archive['epochs']==300
 audit=read(O/'inference_audit.json');assert audit['checkpoint_sha256']==archive['sha256']['train/weights/best.pt']
 preds=read(O/'validation_candidates.json');dest.mkdir(exist_ok=False);summaries={};allrows=[]
 for kind in ['polyline','chord']:
  s,rs,extras=evaluate(ims,preds,kind);summaries[kind]=dict(s,extra_ids=extras);allrows+=rs
 (dest/'all_221_gt_both_lengths.json').write_text(json.dumps(allrows,indent=2),encoding='utf8')
 with (dest/'all_221_gt_both_lengths.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in allrows for k in r)));w.writeheader();w.writerows(allrows)
 reference={r['gt_id']:r for r in read(R/'output/baseline_structure_comparison_20260929/all_gt.json') if r['method']=='yellow_refined' and r['status']=='ok'};paired=[];rng=np.random.default_rng(20260929);ii=[im['id'] for im in ims];bs=rng.integers(0,len(ii),(10000,len(ii)))
 for kind in ['polyline','chord']:
  arm={r['gt_id']:r for r in allrows if r['method']==kind and r['status']=='ok'};keys=sorted(arm.keys()&reference.keys());ss=np.zeros(len(ii));nn=np.zeros(len(ii))
  for k in keys:j=ii.index(arm[k]['image_id']);ss[j]+=arm[k]['ape_percent']-reference[k]['ape_percent'];nn[j]+=1
  den=nn[bs].sum(1);valid=den>0;ci=np.percentile(ss[bs].sum(1)[valid]/den[valid],[2.5,97.5]).tolist() if keys else None;paired.append({'arm':kind,'reference':'yellow_refined_batch1','n_common':len(keys),'mean_ape_difference_pp':float(ss.sum()/nn.sum()) if keys else None,'image_cluster_ci95_pp':ci,'arm_only_gt':sorted(arm.keys()-reference.keys()),'reference_only_gt':sorted(reference.keys()-arm.keys()),'changed_candidate_gt':[k for k in keys if arm[k]['candidate_id']!=reference[k]['candidate_id']]})
 (dest/'analysis_summary.json').write_text(json.dumps({'methods':summaries,'paired':paired,'scope':'single-view HBB 15-point adaptation, not full original stereo system or architecture ranking','test_access':False},indent=2))
 print('VALIDATION_EVALUATED')
if __name__=='__main__':main()
