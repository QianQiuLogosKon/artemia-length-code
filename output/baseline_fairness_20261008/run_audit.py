from pathlib import Path
import json, hashlib, copy, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

O=Path(__file__).resolve().parent; R=O.parents[1]
DATA=R/'output/baseline_hrnet_20260922_v1/data_v2'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def iou(a,b):
    z=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-z
    return z/u if u>0 else 0.
def match(im,t):
    ds=[d for d in im['detections'] if d['confidence']>=t]; gs=im['ground_truth']; ud=set();ug=set();out={}
    for ov,di,gi in sorted(((iou(d['box'],g['box']),di,gi) for di,d in enumerate(ds) for gi,g in enumerate(gs)),reverse=True):
        if ov>=.5 and di not in ud and gi not in ug:
            out[gs[gi]['id']]=ds[di]['id'];ud.add(di);ug.add(gi)
    return out
def mappings(ims,t): return {gid:cid for im in ims for gid,cid in match(im,t).items()}
def main():
    protocol=read(O/'protocol.json')
    manifest=DATA/'manifest.json'
    assert sha(manifest)=='6631a3de2ee64f6600848a8593daff86f46e1d8274e961647d976dae53c40fa8'
    ims=read(manifest)['images']; assert {i['split'] for i in ims}=={'train','val'}
    idxpath=R/'output/baseline_strengthening_20260930_v1/holdout_probability_cache/index.json'
    ids={r['image_id'] for r in read(idxpath)}
    train=copy.deepcopy([i for i in ims if i['split']=='train' and i['id'] in ids])
    assert len(train)==39 and sum(len(i['detections']) for i in train)==670
    override=DATA.parent/'train_box_overrides.json'; ov={g['id']:g['training_box'] for g in read(override)['entries']}
    for im in train:
        for g in im['ground_truth']:
            if g['id'] in ov:g['box']=ov[g['id']]
    base=mappings(train,.001);assert len(base)==99
    grid=[]
    for t in protocol['grid']:
        m=mappings(train,t)
        grid.append({'threshold':t,'matched_gt':len(m),'retained_candidates':sum(d['confidence']>=t for i in train for d in i['detections']),
                     'preserves_gt_set':set(m)==set(base),'lost_gt':sorted(set(base)-set(m))})
    selected=max(x['threshold'] for x in grid if x['preserves_gt_set'])
    cal={'status':'FROZEN_TRAIN_ONLY_SELECTION','threshold':selected,'images':39,'targets':101,'candidates':670,'matched_at_base':99,'grid':grid,
         'source_sha256':{str(p.relative_to(R)):sha(p) for p in [O/'protocol.json',manifest,idxpath,override]},'development_outputs_used_to_select':False}
    cp=O/'calibration.json'
    if cp.exists():assert read(cp)==cal,'Frozen calibration changed'
    else:save(cp,cal)
    print('FROZEN threshold',selected,flush=True)
    val=[i for i in ims if i['split']=='val'];assert len(val)==81
    specs={
      'MMPCF':('output/mmpcf_adaptation_20260928/predictions_all_candidates.json','mmpcf'),
      'MMPCF-CS':('output/baseline_strengthening_20260930_v1/mmpcf_validation_component_snap/predictions_all_candidates.json','mmpcf'),
      'UNet-MS':('output/unet_ms_sag_20260928/decoder_evaluation_v2/MS_all_candidates.json','unet'),
      'UNet-SAG':('output/unet_ms_sag_20260928/decoder_evaluation_v2/SAG_all_candidates.json','unet'),
      'Fish-HBB':('output/fishkeypoints_hbb_adaptation_20260929/validation_candidates.json','fish'),
      'Main raw':('output/padding_ablation_20260927/raw_pad_0.15.json','pose'),
      'Main refined':('output/padding_ablation_20260927/refined_pad_0.15.json','pose')}
    predictions={};hashes={};allids={d['id'] for i in val for d in i['detections']}
    for name,(path,kind) in specs.items():
        p=R/path; hashes[path]=sha(p); data=read(p)
        if isinstance(data,list):data={d['id']:d for d in data}
        assert set(data)==allids
        preds={}
        for cid,x in data.items():
            pts=x.get('points',[]); length=None
            if kind=='unet':length=x.get('length_px')
            elif x.get('status')=='ok':
                if kind=='mmpcf':length=x.get('native_length_px')
                elif kind=='fish':length=x.get('length_polyline_px')
                else:length=float(np.linalg.norm(np.diff(np.array(pts),axis=0),axis=1).sum())
            ok=length is not None and math.isfinite(length) and length>0
            preds[cid]={'ok':ok,'length_px':length if ok else None,'points':pts}
        predictions[name]=preds
    results={};rows=[];basemap=mappings(val,.001)
    for t in dict.fromkeys([.001,selected]):
        mp=mappings(val,t);used=set(mp.values());retained={d['id'] for im in val for d in im['detections'] if d['confidence']>=t};methods={}
        for name,preds in predictions.items():
            good=[]
            for im in val:
                for g in im['ground_truth']:
                    cid=mp.get(g['id']);p=preds.get(cid);ok=p is not None and p['ok']
                    ae=abs(p['length_px']*g['um_per_pixel']-g['length_um']) if ok else None
                    row={'threshold':t,'method':name,'image_id':im['id'],'gt_id':g['id'],'candidate_id':cid,'success':ok,'ae_um':ae,'ape_percent':100*ae/g['length_um'] if ok else None}
                    rows.append(row)
                    if ok:good.append(row)
            apes=[r['ape_percent'] for r in good]
            methods[name]={'success':len(good),'total':221,'mape_percent':float(np.mean(apes)) if apes else None,'median_ape_percent':float(np.median(apes)) if apes else None,
              'mae_um':float(np.mean([r['ae_um'] for r in good])) if good else None,'ape_ge_10':sum(a>=10 for a in apes),
              'extras':sum(preds[c]['ok'] for c in retained-used),'success_gt_ids':[r['gt_id'] for r in good]}
        results[str(t)]={'retained_candidates':len(retained),'matched_gt':len(mp),'lost_gt':sorted(set(basemap)-set(mp)),
          'changed_matches':[g for g in mp if g in basemap and mp[g]!=basemap[g]],'methods':methods}
    historical=read(R/'output/baseline_structure_comparison_20260930_v2/analysis_summary.json')['methods']
    keys={'MMPCF':'MMPCF','UNet-MS':'MS','UNet-SAG':'SAG','Main raw':'yellow_raw','Main refined':'yellow_refined','Fish-HBB':'Fish_HBB_polyline'}
    for name,key in keys.items():
        a=results['0.001']['methods'][name];b=historical[key]
        assert a['success']==b['n_success'] and a['extras']==b['extra_valid_outputs'],name
        for metric in ['mape_percent','median_ape_percent','mae_um']:assert np.isclose(a[metric],b[metric],rtol=1e-10), (name,metric)
    cs=read(R/'output/baseline_strengthening_20260930_v1/mmpcf_validation_component_snap/analysis_summary.json')['methods']['MMPCF_CS']
    a=results['0.001']['methods']['MMPCF-CS'];assert a['extras']==cs['extra_valid_outputs'] and a['success']==cs['n_success'] and np.isclose(a['mape_percent'],cs['mape_percent'])
    for name in specs:
        a=results['0.001']['methods'][name];b=results[str(selected)]['methods'][name]
        b['lost_success_gt_ids']=sorted(set(a['success_gt_ids'])-set(b['success_gt_ids']))
        b['gained_success_gt_ids']=sorted(set(b['success_gt_ids'])-set(a['success_gt_ids']))
    save(O/'all_gt.json',rows);save(O/'results.json',{'calibration_threshold':selected,'conditions':results,'prediction_sha256':hashes,'fixed_test_access':False,'new_inference':False})
    # Select cases from detector geometry, not from downstream errors.
    pools={'duplicate':[],'partial':[],'no_overlap':[]}; lookup={i['id']:i for i in val}
    used=set(basemap.values())
    for im in val:
        for d in im['detections']:
            if d['id'] in used:continue
            mx=max(iou(d['box'],g['box']) for g in im['ground_truth'])
            category='duplicate' if mx>=.5 else 'partial' if mx>0 else 'no_overlap'
            pools[category].append((im,d,mx))
    figindex=[]
    for category,pool in pools.items():
        im,d,mx=sorted(pool,key=lambda x:(-x[1]['confidence'],x[1]['id']))[0]
        src=DATA/im['image'];assert src.resolve().is_relative_to((DATA/'images/val').resolve()) and sha(src)==im['image_sha256']
        img=Image.open(src).convert('RGB');x1,y1,x2,y2=d['box'];dx=.15*(x2-x1);dy=.15*(y2-y1)
        l,t=max(0,int(np.floor(x1-dx))),max(0,int(np.floor(y1-dy)));r,b=min(img.width,int(np.ceil(x2+dx))),min(img.height,int(np.ceil(y2+dy)))
        crop=np.array(img.crop((l,t,r,b)));fig,axes=plt.subplots(2,4,figsize=(16, max(3.5,8*(b-t)/(r-l)+1.2)),layout='constrained')
        for ax,name in zip(axes.flat,['Reference']+list(specs)):
            ax.imshow(crop);ax.set_xlim(0,r-l);ax.set_ylim(b-t,0);ax.axis('off')
            ax.plot([x1-l,x2-l,x2-l,x1-l,x1-l],[y1-t,y1-t,y2-t,y2-t,y1-t],color='#db9400',lw=1)
            if name=='Reference':
                for g in im['ground_truth']:
                    pts=np.asarray(g.get('points',g.get('keypoints',[])))
                    if pts.ndim==2 and len(pts):ax.plot(pts[:,0]-l,pts[:,1]-t,color='#00ad50',lw=1.4)
            else:
                v=predictions[name][d['id']];pts=np.asarray(v['points'])
                if v['ok'] and pts.ndim==2 and len(pts):ax.plot(pts[:,0]-l,pts[:,1]-t,color='#ec3273',lw=1.4)
                name+=' | valid' if v['ok'] else ' | no curve'
            ax.set_title(name,fontsize=10)
        fig.suptitle(f"{category}: {d['id']} | detector={d['confidence']:.5f} | max GT IoU={mx:.3f}\nSame crop; all curves from frozen predictions. Unmatched does not imply background.",fontsize=12)
        fname='same_crop_'+category+'.png';fig.savefig(O/fname,dpi=150);plt.close(fig)
        figindex.append({'category':category,'candidate_id':d['id'],'image_id':im['id'],'detector_confidence':d['confidence'],'max_gt_iou':mx,'path':fname,'rule':protocol['case_rule'],'sha256':sha(O/fname)})
    save(O/'figure_index.json',figindex)
    lines=['# Shared detector-confidence screening (training-side calibration)','','The calibration rule is written to protocol.json before the threshold is computed; development predictions are read only after calibration.json is written. This is a post-hoc engineering diagnostic, not a pre-registered or independent evaluation.',f'','39 training-source images, 101 individuals, 670 candidates; the largest grid threshold that retains all 99 matchable individuals is '+str(selected)+'. It is shared by all methods and does not change the deployment default.','','|Method|Condition|Candidates|Success/221|MAPE/%|MAE/um|Extra valid outputs|','|---|---|---:|---:|---:|---:|---:|']
    for name in specs:
        for ts,v in results.items():
            m=v['methods'][name];lines.append(f"|{name}|conf≥{ts}|{v['retained_candidates']}|{m['success']}/221|{m['mape_percent']:.6f}|{m['mae_um']:.2f}|{m['extras']}|")
    lines+=['','The 0.001 condition reproduces the main results of all 7 methods. Matching changes and lost individuals under the new condition are recorded in results.json; success-subset MAPE alone cannot show improvement. Method-specific no-target rejection is not calibrated; shared screening is not a full reproduction of the original systems.']
    (O/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(O/'verification.json',{'status':'PASS_NUMERICAL_AUDIT_VISUAL_PENDING','primary_methods_reproduced':7,'source_sha256':hashes,'calibration_sha256':sha(cp),'protocol_sha256':sha(O/'protocol.json'),'fixed_test_access':False,'new_inference':False,'threshold_selection_on_development':False})
    print(json.dumps({t:{'candidates':v['retained_candidates'],'matched':v['matched_gt'],'methods':{n:{k:m[k] for k in ['success','mape_percent','extras']} for n,m in v['methods'].items()}} for t,v in results.items()},indent=2))
if __name__=='__main__':main()
