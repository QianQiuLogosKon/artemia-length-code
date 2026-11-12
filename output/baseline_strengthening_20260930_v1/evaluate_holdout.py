"""Score the frozen component-selection probe on training internal holdout only."""
from pathlib import Path
import csv,hashlib,json,time
import cv2,numpy as np
from selection_rules import select,original_extract,box_iou
from decode_ms_sag import ms,sag
O=Path(__file__).resolve().parent;B=O.parent/'unet_ms_sag_20260928';DATA=O.parent/'baseline_hrnet_20260922_v1/data_v2';C=O/'holdout_probability_cache';D=O/'holdout_evaluation'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def match(im):
    pairs=sorted(((box_iou(d['box'],g['box']),di,gi) for di,d in enumerate(im['detections']) for gi,g in enumerate(im['ground_truth'])),reverse=True);usedd=set();usedg=set();out={}
    for ov,di,gi in pairs:
        if ov>=.5 and di not in usedd and gi not in usedg:out[im['ground_truth'][gi]['id']]=(im['detections'][di]['id'],ov);usedd.add(di);usedg.add(gi)
    return out
def main():
    assert read(C/'verification.json')['status']=='PASS';protocol=read(O/'holdout_protocol.json');assert sha(O/'holdout_protocol.json')==read(C/'verification.json')['protocol_sha256'];index=read(C/'index.json');assert len(index)==670
    ids={r['image_id'] for r in index};m=read(DATA/'manifest.json');ims=[im for im in m['images'] if im['split']=='train' and im['id'] in ids];assert len(ims)==39 and sum(len(im['ground_truth']) for im in ims)==101
    overrides={x['id']:x for x in read(DATA.parent/'train_box_overrides.json')['entries']};applied=[]
    for im in ims:
        for g in im['ground_truth']:
            if g['id'] in overrides:g['box']=overrides[g['id']]['training_box'];applied.append(g['id'])
    D.mkdir(exist_ok=False);cv2.setNumThreads(1);start=time.time();outputs={m+'__'+r:{} for m in ['MS','SAG'] for r in protocol['rules']}
    for i,record in enumerate(index,1):
        assert sha(C/record['probability'])==record['sha256'];p=np.load(C/record['probability']);l,t,r,b=record['crop'];nw,nh=record['resize'];ox,oy=record['offset'];scale=np.array([nw/(r-l),nh/(b-t)]);ref=(np.array(record['box']).reshape(2,2)-[l,t]+.5)*scale-.5+[ox,oy];ref=ref.ravel()
        for method,decoder in [('MS',ms),('SAG',sag)]:
            mask=decoder(p);original,_=original_extract(mask,p)
            for rule in protocol['rules']:
                path,diag=select(mask,p,ref,rule)
                if rule=='mass_v2':assert path==original
                q=np.array(path,float).reshape(-1,2);q=(q-[ox,oy]+.5)/scale-.5+[l,t] if len(q) else q
                length=float(np.linalg.norm(np.diff(q,axis=0),axis=1).sum()) if len(q)>1 else None
                outputs[method+'__'+rule][record['id']]={'status':'ok' if length is not None and length>0 else 'empty_path','length_px':length,'points':q.tolist(),'diagnostics':diag}
        if i%100==0:print('DECODED',i,'/670',flush=True)
    save(D/'predictions.json',outputs);rows=[];summaries={};mapping={};matched_ids=set()
    for im in ims:mapping.update(match(im))
    matched_ids={cid for cid,ov in mapping.values()}
    for arm,preds in outputs.items():
        extras=[cid for cid,p in preds.items() if p['status']=='ok' and cid not in matched_ids]
        for im in ims:
            for g in im['ground_truth']:
                cid,ov=mapping.get(g['id'],(None,None));p=preds[cid] if cid else None;ok=p is not None and p['status']=='ok';length=p['length_px'] if ok else None
                truepx=float(g['length_px']);ape=100*abs(length/truepx-1) if ok else None;scale=g.get('um_per_pixel');trueum=g.get('length_um');apeum=100*abs(length*scale/trueum-1) if ok and scale is not None and trueum is not None else None
                rows.append(dict(method=arm,image_id=im['id'],gt_id=g['id'],candidate_id=cid,detector_iou=ov,status='detector_unmatched' if cid is None else p['status'],true_length_px=truepx,pred_length_px=length,ape_percent=ape,physical_ape_percent=apeum))
        rr=[x for x in rows if x['method']==arm and x['status']=='ok'];apes=[x['ape_percent'] for x in rr];um=[x['physical_ape_percent'] for x in rr if x['physical_ape_percent'] is not None]
        summaries[arm]={'n_gt':101,'success':len(rr),'mape_percent':float(np.mean(apes)) if apes else None,'median_ape_percent':float(np.median(apes)) if apes else None,'ape_ge_10':sum(x>=10 for x in apes),'physical_mape_percent':float(np.mean(um)) if um else None,'physical_success_n':len(um),'extra_valid_outputs':len(extras),'extra_ids':extras,'candidate_status_counts':{s:sum(p['status']==s for p in preds.values()) for s in ['ok','empty_path']}}
    pairs=[];allpass=True;strict=False
    for method in ['MS','SAG']:
        a=summaries[method+'__mass_v2'];b=summaries[method+'__box_component_v1'];base={r['gt_id']:r for r in rows if r['method']==method+'__mass_v2' and r['status']=='ok'};new={r['gt_id']:r for r in rows if r['method']==method+'__box_component_v1' and r['status']=='ok'};keys=sorted(base.keys()&new.keys());diff=[new[k]['ape_percent']-base[k]['ape_percent'] for k in keys]
        checks={'coverage_non_decreasing':b['success']>=a['success'],'extras_non_increasing':b['extra_valid_outputs']<=a['extra_valid_outputs'],'mean_ape_non_increasing':b['mape_percent']<=a['mape_percent'],'median_ape_non_increasing':b['median_ape_percent']<=a['median_ape_percent'],'large_errors_non_increasing':b['ape_ge_10']<=a['ape_ge_10']};allpass=allpass and all(checks.values());strict=strict or b['mape_percent']<a['mape_percent']
        pairs.append({'method':method,'n_common':len(keys),'candidate_minus_original_mean_ape_pp':float(np.mean(diff)),'improved':sum(v<-1e-9 for v in diff),'worsened':sum(v>1e-9 for v in diff),'unchanged':sum(abs(v)<=1e-9 for v in diff),'lost_gt':sorted(base.keys()-new.keys()),'gained_gt':sorted(new.keys()-base.keys()),'changed_detector_matches':[k for k in keys if base[k]['candidate_id']!=new[k]['candidate_id']],'gate_checks':checks})
    gate=allpass and strict;save(D/'all_gt.json',rows)
    with (D/'all_gt.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(D/'analysis_summary.json',{'methods':summaries,'paired':pairs,'promotion_gate_passed':gate,'decision':'Eligible for one-time formal-validation cache evaluation under frozen rule' if gate else 'Do not promote; retain diagnostic and do not tune on formal validation','holdout_role':'Used previously for epoch55 selection and now rule diagnosis; not independent generalization','reference_box':'actual cached detector boxes, not GT boxes','training_box_overrides_used_offline':applied,'test_access':False})
    assert len(rows)==404 and all(len(v)==670 for v in outputs.values());assert all(not p['changed_detector_matches'] for p in pairs)
    save(D/'verification.json',{'status':'PASS','candidates_per_arm':670,'arms':4,'gt_rows':404,'original_rule_regression':True,'fixed_matching_before_validity':True,'protocol_sha256':sha(O/'holdout_protocol.json'),'rule_sha256':sha(O/'selection_rules.py'),'checkpoint_sha256':protocol['checkpoint_sha256'],'formal_validation_evaluation':False,'test_access':False,'training':False,'elapsed_seconds':time.time()-start})
    print(json.dumps({'methods':{k:{kk:vv for kk,vv in v.items() if kk!='extra_ids'} for k,v in summaries.items()},'paired':pairs,'promotion_gate_passed':gate},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
