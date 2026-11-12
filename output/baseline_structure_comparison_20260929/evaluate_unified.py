from pathlib import Path
import json,sys,csv,hashlib,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
B=ROOT/'output/baseline_hrnet_20260922_v1'; DATA=B/'data_v2'; A=B
sys.path.insert(0,str(A/'src'));from common import match_boxes,iou,arc

def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def csvout(name,rs):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)

def main():
 ims=[x for x in read(DATA/'manifest.json')['images'] if x['split']=='val']; ids=[d['id'] for im in ims for d in im['detections']]; assert len(ims)==81 and len(ids)==1498
 sources={}; methods={}; roles={}
 def add(name,path,kind,role):
  p=ROOT/path; d=read(p); d={x['id']:x for x in d} if isinstance(d,list) else d;assert set(d)==set(ids)
  out={}
  for cid,v in d.items():
   if kind=='unet': length=v.get('length_px');status='ok' if length is not None and length>0 else v.get('diagnostics',{}).get('reason','empty')
   elif kind=='mmpcf':status=v['status'];length=v.get('native_length_px') if status=='ok' else None
   else:
    status=v['status'];length=arc(np.asarray(v['points'])) if status=='ok' else None
   if length is not None: assert np.isfinite(length) and length>0
   out[cid]={'status':status,'length_px':length,'points':v.get('points')}
  methods[name]=out;sources[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()};roles[name]=role
 add('MMPCF','output/mmpcf_adaptation_20260928/predictions_all_candidates.json','mmpcf','Wang-inspired task baseline adaptation')
 for n in ['MS','SAG']: add(n,f'output/unet_ms_sag_20260928/decoder_evaluation_v2/{n}_all_candidates.json','unet','Wang-inspired shared UNet baseline adaptation')
 for stage in ['raw','refined']:
  add('yellow_'+stage,f'output/padding_ablation_20260927/{stage}_pad_0.15.json','pose','proposed pipeline / endpoint ablation')
  for mode in ['no_prompt','black_outside']:
   add(mode+'_'+stage,f'output/target_context_ablation_20260929/{stage}_{mode}.json','pose','fixed-weight input component sensitivity, not trained architecture baseline')
  base=methods['yellow_'+stage];out={k:dict(v) for k,v in base.items()}
  for v in out.values():
   if v['status']=='ok':v['length_px']=float(np.linalg.norm(np.asarray(v['points'])[0]-np.asarray(v['points'])[-1]))
  methods['chord_'+stage]=out;sources['chord_'+stage]=dict(sources['yellow_'+stage]);roles['chord_'+stage]='endpoint Euclidean length adaptation; NOT FishKeypoints system reproduction'
 # Protocol is fixed before any metrics.
 save('protocol.json',dict(split='validation only',images=81,gt=221,candidates=1498,matching='all detector candidates first; descending (IoU,det index,GT index); IoU>=0.5; check method validity only after match; no fallback to another detection',threshold=.001,large_error='APE >=10%',roles=roles,source_hashes=sources,fixed_test_access=False,training=False,bootstrap={'replicates':10000,'seed':20260929,'cluster':'image; sample all 81 image clusters, ratio of summed errors to successful individuals'},limitations=['adaptations not original author implementations','single checkpoint and development validation','input interventions use weights trained with yellow prompt','chord uses original 15-point checkpoint endpoints, not separately trained 2-point model','black masking uses HBB, no OBB/rotation/stereo','cannot assert structural neural architecture ranking']))
 rows=[];cs=[];summary={};match_changes={}
 for name,pred in methods.items():
  mr=[];mc=[]
  for im in ims:
   pairs=match_boxes(im['detections'],im['ground_truth']);by_gt={gi:(di,ov) for di,gi,ov in pairs};used={di for di,gi,ov in pairs}
   for gi,g in enumerate(im['ground_truth']):
    r={'method':name,'image_id':im['id'],'gt_id':g['id'],'gt_um':g['length_um'],'candidate_id':None,'status':'detector_unmatched','pred_um':None,'ae_um':None,'ape_percent':None}
    if gi in by_gt:
     di,ov=by_gt[gi];cid=im['detections'][di]['id'];p=pred[cid];r.update(candidate_id=cid,status=p['status'],detector_iou=ov)
     if p['status']=='ok':
      val=p['length_px']*g['um_per_pixel'];err=abs(val-g['length_um']);r.update(pred_um=val,ae_um=err,ape_percent=err/g['length_um']*100)
    mr.append(r)
   for di,d in enumerate(im['detections']):
    p=pred[d['id']];mx=max(iou(d['box'],g['box']) for g in im['ground_truth']);extra=di not in used and p['status']=='ok'
    mc.append({'method':name,'image_id':im['id'],'candidate_id':d['id'],'status':p['status'],'length_px':p['length_px'],'matched_detector':di in used,'extra_valid':extra,'geometry':'matched' if di in used else 'duplicate' if mx>=.5 else 'partial_overlap' if mx>0 else 'no_box_overlap'})
  good=[r for r in mr if r['status']=='ok'];assert len(mr)==221 and len(mc)==1498;apes=np.array([r['ape_percent'] for r in good]);summary[name]={'n_success':len(good),'n_gt':221,'prediction_rate_percent':len(good)/221*100,'mape_percent':float(apes.mean()),'median_ape_percent':float(np.median(apes)),'mae_um':float(np.mean([r['ae_um'] for r in good])),'ape_ge_10':int((apes>=10).sum()),'extra_valid_outputs':sum(r['extra_valid'] for r in mc),'role':roles[name]};rows+=mr;cs+=mc
  if name in ['MS','SAG']:
   old=[r for r in read(ROOT/'output/unet_ms_sag_20260928/decoder_evaluation_v2/all_221_gt_both_methods.json') if r['method']==name]; oldmap={r['gt_id']:r for r in old};match_changes[name]=[{'gt_id':r['gt_id'],'old_candidate':oldmap[r['gt_id']]['candidate_id'],'new_candidate':r['candidate_id']} for r in mr if (r['candidate_id'] if r['status']=='ok' else '')!=oldmap[r['gt_id']]['candidate_id']]
 csvout('all_gt.csv',rows);save('all_gt.json',rows);csvout('all_candidates.csv',cs);save('matching_audit.json',match_changes)
 pairs=[('yellow_raw','yellow_refined'),('MS','SAG')]+[(n,'yellow_refined') for n in ['MMPCF','MS','SAG','no_prompt_refined','black_outside_refined','chord_refined']]+[(n,'yellow_raw') for n in ['no_prompt_raw','black_outside_raw','chord_raw']]
 paired=[];detailed=[];rng=np.random.default_rng(20260929);image_ids=[im['id'] for im in ims];boot=rng.integers(0,81,size=(10000,81))
 for a,b in pairs:
  am={r['gt_id']:r for r in rows if r['method']==a and r['status']=='ok'};bm={r['gt_id']:r for r in rows if r['method']==b and r['status']=='ok'};common=sorted(am.keys()&bm.keys());ss=np.zeros(81);nn=np.zeros(81);diff=[]
  for k in common:
   d=am[k]['ape_percent']-bm[k]['ape_percent'];diff.append(d);ix=image_ids.index(am[k]['image_id']);ss[ix]+=d;nn[ix]+=1;detailed.append({'a':a,'b':b,'gt_id':k,'image_id':am[k]['image_id'],'a_ape':am[k]['ape_percent'],'b_ape':bm[k]['ape_percent'],'delta_pp':d,'same_candidate':am[k]['candidate_id']==bm[k]['candidate_id']})
  den=nn[boot].sum(1);valid=den>0;vals=ss[boot].sum(1)[valid]/den[valid];paired.append({'a':a,'b':b,'n_common':len(common),'mean_a_minus_b_pp':float(np.mean(diff)),'ci95_image_cluster_pp':np.percentile(vals,[2.5,97.5]).tolist(),'a_only_gt':sorted(am.keys()-bm.keys()),'b_only_gt':sorted(bm.keys()-am.keys()),'changed_matches':[k for k in common if am[k]['candidate_id']!=bm[k]['candidate_id']],'a_common_mape':float(np.mean([am[k]['ape_percent'] for k in common])),'b_common_mape':float(np.mean([bm[k]['ape_percent'] for k in common])),'nonempty_bootstrap_replicates':int(valid.sum())})
 save('analysis_summary.json',{'methods':summary,'paired':paired});csvout('paired_differences.csv',detailed)
 save('verification.json',{'status':'PASS_DATA_CHECKS_VISUAL_REVIEW_PENDING','methods':len(methods),'gt_rows':len(rows),'candidate_rows':len(cs),'all_lengths_finite_positive':True,'matching_audit':match_changes,'test_access':False})
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
