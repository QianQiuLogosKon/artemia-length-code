"""Build train/holdout/val ribbon supervision from existing 15-point labels.

Only train/val manifest entries are read. GT boxes/points are used to construct
training supervision and validation diagnostics, never for future candidate
selection. The resulting ribbon is a project-derived pseudo-label, not the
manual line-map annotation used by Wang et al.
"""
from pathlib import Path
import json,hashlib,csv,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
BASE=ROOT/'output/baseline_hrnet_20260922_v1'; DATA=BASE/'data_v2'; DS=OUT/'dataset_v2'
sys.path.insert(0,str(ROOT/'output/baseline_comparison_20260922_v1'))
from infer_yolo_candidates import iou
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def draw_ribbon(points,size,width=9):
 mask=np.zeros((size,size),np.uint8); p=np.round(points).astype(np.int32)
 cv2.polylines(mask,[p],False,255,thickness=max(1,int(width)),lineType=cv2.LINE_8)
 return mask
def main():
 if DS.exists() and any(DS.rglob('*.png')): raise SystemExit('dataset files already exist; refusing overwrite')
 for s in ['train','holdout','val']:
  (DS/'images'/s).mkdir(parents=True,exist_ok=True); (DS/'labels'/s).mkdir(parents=True,exist_ok=True)
 m=json.loads((DATA/'manifest.json').read_text(encoding='utf-8')); ims=[x for x in m['images'] if x['split'] in ['train','val']]
 overrides={e['id']:e for e in json.loads((BASE/'train_box_overrides.json').read_text(encoding='utf-8'))['entries']}
 rows=[]; counts={'train':0,'holdout':0,'val':0}
 for im in ims:
  assert digest(DATA/im['image'])==im['image_sha256']
  image=cv2.imdecode(np.fromfile(str(DATA/im['image']),np.uint8),cv2.IMREAD_COLOR); h,w=image.shape[:2]
  for gi,g in enumerate(im['ground_truth']):
   # Stable image-level holdout: never split instances from the same source image.
   split='val' if im['split']=='val' else ('holdout' if int(hashlib.sha1(im['id'].encode()).hexdigest()[:8],16)%10==0 else 'train')
   box=overrides.get(g['id'],{}).get('training_box',g['box'])
   if g['id'] in overrides:
    assert im['split']=='train' and np.allclose(overrides[g['id']]['original_box'],g['box'])
   x1,y1,x2,y2=np.asarray(box,float); dx,dy=.15*(x2-x1),.15*(y2-y1)
   l=max(0,int(np.floor(x1-dx))); t=max(0,int(np.floor(y1-dy))); r=min(w,int(np.ceil(x2+dx))); b=min(h,int(np.ceil(y2+dy)))
   # Supervised crops expose every target point; never applied to inference ROIs.
   original_bounds=[l,t,r,b]; allpts=np.asarray(g['points'],float)
   l=max(0,min(l,int(np.floor(allpts[:,0].min()))-1)); t=max(0,min(t,int(np.floor(allpts[:,1].min()))-1))
   r=min(w,max(r,int(np.ceil(allpts[:,0].max()))+2)); b=min(h,max(b,int(np.ceil(allpts[:,1].max()))+2))
   crop=image[t:b,l:r];
   if crop.size==0: continue
   scale=256/max(crop.shape[:2]); nw=max(8,round(crop.shape[1]*scale)); nh=max(8,round(crop.shape[0]*scale)); resized=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA)
   pts=(np.asarray(g['points'],float)-[l,t]+.5)*[nw/crop.shape[1],nh/crop.shape[0]]-.5
   # center on 256 canvas as in the Wang input convention
   ox=(256-nw)//2; oy=(256-nh)//2; canvas=np.zeros((256,256,3),np.uint8); canvas[oy:oy+nh,ox:ox+nw]=resized
   pts=pts+[ox,oy]; mask=draw_ribbon(pts,256,9)
   name=f'{im["id"]}__g{gi+1}'.replace(':','_').replace('/','_')
   ip=DS/'images'/split/(name+'.png'); lp=DS/'labels'/split/(name+'.png'); cv2.imencode('.png',canvas)[1].tofile(str(ip)); cv2.imencode('.png',mask)[1].tofile(str(lp)); counts[split]+=1
   rows.append(dict(id=g['id'],image=im['image'],source_image_id=im['id'],split=split,instance=gi+1,image_path=str(ip.relative_to(OUT)).replace('\\','/'),label_path=str(lp.relative_to(OUT)).replace('\\','/'),crop_bounds=[l,t,r,b],original_bounds=original_bounds,supervision_roi_expanded=original_bounds!=[l,t,r,b],resize=[nw,nh],offset=[ox,oy],ribbon_width_input_px=9,point_count=15,image_sha256=digest(ip),label_sha256=digest(lp),true_length_um=g['length_um']))
 with (OUT/'ribbon_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 proto=dict(source_manifest_sha256=digest(DATA/'manifest.json'),label_type='project-derived fixed-width ribbon from 15-point trunk polyline',ribbon_width_input_px=9,input_size=[256,256],padding=.15,train_images=counts['train'],holdout_images=counts['holdout'],val_instances=counts['val'],gt_used='Only for constructing supervision and offline evaluation; never for inference candidate selection.',fixed_test_access=False,training_performed=False)
 save(OUT/'ribbon_protocol.json',proto); save(OUT/'ribbon_build_audit.json',dict(status='PASS',rows=len(rows),counts=counts,source_manifest_sha256=digest(DATA/'manifest.json'),fixed_test_access=False,training_performed=False))
 print(json.dumps(dict(status='PASS',counts=counts,rows=len(rows)),ensure_ascii=False))
if __name__=='__main__':main()
