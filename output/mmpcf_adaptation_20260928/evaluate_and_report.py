"""Offline native-arc evaluation, diagnostic plots, and independent record."""
from pathlib import Path
import json,csv,sys,math,html,hashlib
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont
OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
BASE=ROOT/'output/baseline_hrnet_20260922_v1'; DATA=BASE/'data_v2'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
ARCHIVE=BASE
sys.path.insert(0,str(ARCHIVE/'src')); from common import match_boxes,iou,sha256,read_image
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def csvout(name,rows,fields=None):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
def stats(rows):
    good=[r for r in rows if r['status']=='ok']; ape=[r['ape_percent'] for r in good]
    return dict(n_success=len(good),n_gt=len(rows),prediction_rate_percent=len(good)/len(rows)*100,mape_percent=float(np.mean(ape)) if ape else None,median_ape_percent=float(np.median(ape)) if ape else None,mae_um=float(np.mean([r['ae_um'] for r in good])) if good else None,ape_ge_10=sum(v>=10 for v in ape),ape_ge_20=sum(v>=20 for v in ape),direction='N/A: no learned head/tail semantics')
def main():
    manifest=read(DATA/'manifest.json'); ims=[x for x in manifest['images'] if x['split']=='val']; pred=read(OUT/'predictions_all_candidates.json')
    expected=[d['id'] for im in ims for d in im['detections']]; assert list(pred)==expected and len(pred)==1498
    rows=[]; candidates=[]; gm={}; dm={}
    for im in ims:
        pairs=match_boxes(im['detections'],im['ground_truth']); by_gt={gi:(di,v) for di,gi,v in pairs}; by_det={di:gi for di,gi,v in pairs}
        for gi,g in enumerate(im['ground_truth']):
            gm[g['id']]=(im,g); r=dict(id=g['id'],image=im['source_name'],image_id=im['id'],true_um=g['length_um'])
            if gi not in by_gt:r.update(status='detector_unmatched_at_iou_0.5',max_detector_iou=max(iou(d['box'],g['box']) for d in im['detections']))
            else:
                di,overlap=by_gt[gi]; d=im['detections'][di]; p=pred[d['id']]; r.update(status=p['status'],candidate_id=d['id'],detector_iou=overlap)
                if p['status']=='ok':
                    length=p['native_length_px']*g['um_per_pixel']; ae=abs(length-g['length_um']); r.update(predicted_um=length,ae_um=ae,ape_percent=100*ae/g['length_um'])
                    assert np.isfinite(p['points']).all() and length>0
            rows.append(r)
        for di,d in enumerate(im['detections']):
            dm[d['id']]=(im,d); p=pred[d['id']]; mx=max(iou(d['box'],g['box']) for g in im['ground_truth'])
            cat='matched' if di in by_det else 'duplicate' if mx>=.5 else 'partial_overlap' if mx>0 else 'no_gt_box_overlap'
            candidates.append(dict(candidate_id=d['id'],image=im['source_name'],status=p['status'],detector_confidence=d['confidence'],matched_gt=im['ground_truth'][by_det[di]]['id'] if di in by_det else '',geometry=cat,extra_valid=bool(di not in by_det and p['status']=='ok')))
    assert len(rows)==221
    s=stats(rows); s.update(candidate_status_counts=dict(Counter(p['status'] for p in pred.values())),gt_status_counts=dict(Counter(r['status'] for r in rows)),extra_valid_outputs=sum(c['extra_valid'] for c in candidates),extra_geometry=dict(Counter(c['geometry'] for c in candidates if c['extra_valid'])))
    oldrows=read(ROOT/'output/baseline_comparison_20260922_v1/paired_all_221_gt.json'); oldmap={r['id']:r for r in oldrows}; joined=[]
    for r in rows:
        o=oldmap[r['id']]; j=dict(r); j['yolo_refined_status']=o['yolo_refined_status']; j['yolo_refined_ape_percent']=o.get('yolo_refined_ape_percent')
        if r['status']=='ok' and o['yolo_refined_status']=='ok':j['mmpcf_minus_yolo_refined_ape_pp']=r['ape_percent']-o['yolo_refined_ape_percent']
        joined.append(j)
    common=[r for r in joined if 'mmpcf_minus_yolo_refined_ape_pp' in r]; paired=dict(n_common=len(common),mmpcf_common_mape=float(np.mean([r['ape_percent'] for r in common])),yolo_refined_common_mape=float(np.mean([r['yolo_refined_ape_percent'] for r in common])),mean_difference_pp=float(np.mean([r['mmpcf_minus_yolo_refined_ape_pp'] for r in common])),note='Coverage is separately reported; adapted traditional baseline and selected YOLO development checkpoints, not strict architecture ranking.')
    csvout('all_221_gt.csv',rows); csvout('all_1498_candidates.csv',candidates); csvout('paired_yolo_all_221.csv',joined); csvout('failures.csv',[r for r in rows if r['status']!='ok']); save(OUT/'all_221_gt.json',rows)
    save(OUT/'analysis_summary.json',dict(mmpcf=s,paired=paired,training=False,test_access=False))
    good=sorted([r for r in rows if r['status']=='ok'],key=lambda r:r['ape_percent']); chosen=[]; used=set()
    for r,why in [(good[0],'Lowest error'),(good[len(good)//2],'Median successful error'),(good[-1],'Largest error'),(good[-2],'Second largest error')]:
        if r['id'] not in used:chosen.append((r,why));used.add(r['id'])
    for status in sorted({r['status'] for r in rows if r['status']!='ok'}):
        r=next(r for r in rows if r['status']==status); chosen.append((r,status))
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n)
    (OUT/'figures').mkdir(exist_ok=True); index=[]
    for n,(r,why) in enumerate(chosen,1):
        im,g=gm[r['id']]; cid=r.get('candidate_id'); d=dm[cid][1] if cid else max(im['detections'],key=lambda d:iou(d['box'],g['box'])); p=pred[d['id']]; a=read_image(DATA/im['image']); h,w=a.shape[:2]; rgb=Image.fromarray(a[:,:,::-1])
        pts=np.array(g['points']); box=np.array(d['box']); low=np.minimum(pts.min(0),box[:2]); high=np.maximum(pts.max(0),box[2:]); margin=(high-low)*.2+10; l,t=np.maximum(low-margin,0).astype(int); rr,b=np.minimum(high+margin,[w,h]).astype(int)
        canvas=Image.new('RGB',(1800,880),'white'); draw=ImageDraw.Draw(canvas); draw.text((22,14),f'{r["id"]} | {why} | status: {r["status"]}',font=font(28),fill='#172a42')
        draw.text((22,52),f'Candidate {d["id"]} | detector conf={d["confidence"]:.4f} | native polynomial length; head/tail N/A',font=font(22),fill='#475569')
        tile=rgb.crop((l,t,rr,b)); scale=min(840/tile.width,670/tile.height); tile=tile.resize((round(tile.width*scale),round(tile.height*scale))); tx=25+(840-tile.width)//2; ty=105+(670-tile.height)//2;canvas.paste(tile,(tx,ty))
        def xy(q):return (tx+(q[0]-l)*scale,ty+(q[1]-t)*scale)
        draw.line([xy(q) for q in pts],fill='#00b6ac',width=5);draw.rectangle([xy(box[:2]),xy(box[2:])],outline='#955bc0',width=2)
        if p['status']=='ok':
            q=np.array(p['curve_257']); points=[xy(v) for v in q]
            # clip predicted curve segments to the displayed image area
            import cv2
            for u,v in zip(points,points[1:]):
                ok,u,v=cv2.clipLine((tx,ty,tile.width,tile.height),tuple(round(x) for x in u),tuple(round(x) for x in v))
                if ok:draw.line([u,v],fill='#ef7519',width=3)
        stagefile=OUT/'stages'/(d['id'].replace(':','_')+'.npz')
        if stagefile.exists():
            st=np.load(stagefile)
            for k,key in enumerate(['gray','mask','aligned','opened','skeleton']):
                v=st[key]; v=v if key=='gray' else v*255; tile2=Image.fromarray(v.astype(np.uint8)).convert('RGB'); tile2.thumbnail((280,270)); x=900+(k%3)*290; y=115+(k//3)*325;draw.text((x,y-28),key,font=font(21),fill='#334155');canvas.paste(tile2,(x,y))
        else:draw.text((900,130),'No valid fitted curve. Failure remains in denominator.',font=font(24),fill='#a13425')
        txt=f'GT: {g["length_um"]:.1f} um';txt+=f' | predicted: {r["predicted_um"]:.1f} um | APE: {r["ape_percent"]:.3f}%' if r['status']=='ok' else ' | prediction: missing (not imputed as zero)'
        draw.text((22,804),txt,font=font(23),fill='#25354a');draw.text((22,842),'Teal: GT (offline only) | orange: fitted curve | purple: detector box. Examples are diagnostic, not random.',font=font(22),fill='#475569')
        file=f'figures/{n:02d}_{r["id"].replace(":","_")}.png';canvas.save(OUT/file);index.append(dict(id=r['id'],reason=why,file=file))
    save(OUT/'figure_index.json',index)
    cards=''.join(f'<h2>{e["id"]} | {e["reason"]}</h2><a href="{e["file"]}"><img src="{e["file"]}"></a>' for e in index)
    (OUT/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>MMPCF adaptation diagnostics</title><style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px}img{width:100%}</style><h1>MMPCF多目标适配版：验证诊断</h1><p>非作者代码复现；保留失败分母，GT仅用于离线图示。</p>'+cards,encoding='utf-8')
    for n,start in enumerate(range(0,len(index),4),1):
        sheet=Image.new('RGB',(1800,880),'#eeeeee')
        for j,e in enumerate(index[start:start+4]):
            tile=Image.open(OUT/e['file']);tile.thumbnail((900,440));sheet.paste(tile,((j%2)*900,(j//2)*440))
        sheet.save(OUT/f'contact_{n}.png')
    print(json.dumps(dict(summary=s,paired=paired,figures=len(index)),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
