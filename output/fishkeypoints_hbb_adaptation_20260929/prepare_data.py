from pathlib import Path
import json,csv,hashlib,tarfile
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
O=Path(__file__).resolve().parent;R=O.parents[1];D=R/'output/baseline_hrnet_20260922_v1/data_v2';read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 m=read(D/'manifest.json');over={e['id']:e for e in read(D.parent/'train_box_overrides.json')['entries']}; rows=[];counts={'train':0,'holdout':0};examples=[]
 for split in counts:
  for k in ['images','labels']:(O/'dataset'/k/split).mkdir(parents=True,exist_ok=True)
 for im in m['images']:
  if im['split']!='train':continue
  assert sha(D/im['image'])==im['image_sha256'];a=cv2.imdecode(np.fromfile(str(D/im['image']),np.uint8),1);h,w=a.shape[:2];split='holdout' if int(hashlib.sha1(im['id'].encode()).hexdigest()[:8],16)%10==0 else 'train'
  for g in im['ground_truth']:
   pts=np.array(g['points'],float);box=np.array(over.get(g['id'],{}).get('training_box',g['box']),float);delta=(box[2:]-box[:2])*.15;lo=np.floor(box[:2]-delta);hi=np.ceil(box[2:]+delta);lo=np.minimum(lo,np.floor(pts.min(0))-1);hi=np.maximum(hi,np.ceil(pts.max(0))+2);l,t=np.maximum(lo,0).astype(int);rr,b=np.minimum(hi,[w,h]).astype(int);crop=a[t:b,l:rr];scale=min(1.,1024/max(crop.shape[:2]));nw,nh=round(crop.shape[1]*scale),round(crop.shape[0]*scale);v=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA);q=(pts-[l,t]+.5)*[nw/(rr-l),nh/(b-t)]-.5
   bblo=np.minimum(box[:2],pts.min(0));bbhi=np.maximum(box[2:],pts.max(0));bblo=np.clip((bblo-[l,t])*[nw/(rr-l),nh/(b-t)],[0,0],[nw,nh]);bbhi=np.clip((bbhi-[l,t])*[nw/(rr-l),nh/(b-t)],[0,0],[nw,nh]);vals=[0,*((bblo+bbhi)/2/[nw,nh]),*((bbhi-bblo)/[nw,nh])]
   for pt in q: vals.extend([*(pt/[nw,nh]),2])
   assert len(vals)==50 and all(0<=v<=1 for v in vals[1:5]) and np.all(q>=0) and np.all(q<[nw,nh])
   name=g['id'].replace(':','__');ip=O/'dataset/images'/split/(name+'.jpg');lp=O/'dataset/labels'/split/(name+'.txt');cv2.imencode('.jpg',v,[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(ip));lp.write_text(' '.join(f'{x:.9g}' for x in vals)+'\n',encoding='utf8');counts[split]+=1;rows.append({'id':g['id'],'source_image_id':im['id'],'split':split,'image':ip.relative_to(O).as_posix(),'label':lp.relative_to(O).as_posix(),'image_sha256':sha(ip),'label_sha256':sha(lp),'crop':[int(l),int(t),int(rr),int(b)]})
   if len(examples)<2 or g['id'] in ['train_0296:g3','train_0296:g4','train_0056:g1']:
    examples.append((g['id'],v,q))
 assert counts=={'train':910,'holdout':101};assert {r['source_image_id'] for r in rows if r['split']=='train'}.isdisjoint({r['source_image_id'] for r in rows if r['split']=='holdout'})
 (O/'dataset_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
 (O/'data.yaml').write_text('path: dataset\ntrain: images/train\nval: images/holdout\nkpt_shape: [15, 3]\nflip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]\nnames:\n  0: target_shrimp\n',encoding='utf8')
 proto={'name':'FishKeypoints-inspired HBB crop YOLOv8s-Pose single-view 15-point adaptation','paper_doi':'10.1016/j.compag.2026.111852','not_author_reproduction':True,'source_manifest_sha256':sha(D/'manifest.json'),'train_instances':910,'internal_holdout_instances':101,'image_level_split':'same deterministic training-image hash split as shared UNet','epochs':300,'imgsz':1024,'batch':8,'optimizer':'AdamW','lr0':.001,'momentum':.9,'cos_lr':True,'seed':20260929,'checkpoint_selection':'best internal holdout Ultralytics pose fitness, not formal validation length','supervision':'15 ordered keypoints, even points derived; original eight odd points remain; single target label per GT HBB crop','preprocessing':'GT HBB padding .15 for training; raw RGB without yellow; expanded to contain target annotation for supervision only; JPEG95 longest side <=1024','inference':'reuse 1498 detector HBB candidates; .15 padding no GT crop expansion; raw RGB; class0 maximum reference IoU and confidence tie break; keep failures','adaptations':['HBB route only; no stereo/OBB/tracking','15 Artemia centreline points replacing fish anatomical landmarks','s scale is project choice; architecture scale not verified in paper','padding and train holdout are project choices','no yellow prompt; no endpoint refiner for primary comparison','native 15-point polyline and endpoint chord both evaluated without choosing best metric','YOLO histories and labels differ from current selected main model; no pure architecture ranking'],'augmentation':{'degrees':45,'flipud':.5,'fliplr':.5,'mixup':.1,'mosaic':1.0,'hsv_h':.015,'hsv_s':.7,'hsv_v':.4,'note':'settings not completely specified by paper; amplitudes project defaults/choices; endpoint semantics unchanged under flips'},'formal_val_used_for_training':False,'fixed_test_access':False}
 (O/'training_protocol.json').write_text(json.dumps(proto,ensure_ascii=False,indent=2),encoding='utf8')
 canvas=Image.new('RGB',(1000,220*len(examples)),'white');dr=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 for i,(name,img,pts) in enumerate(examples):
  t=Image.fromarray(img[:,:,::-1]);sc=min(950/t.width,180/t.height);t=t.resize((round(t.width*sc),round(t.height*sc)));canvas.paste(t,(10,i*220+30));dr.text((10,i*220),name,font=font,fill='black');dr.line([(10+x*sc,i*220+30+y*sc) for x,y in pts],fill='#00ff99',width=2)
 canvas.save(O/'dataset_qa.png');(O/'dataset_verification.json').write_text(json.dumps({'status':'PASS_DATA_VISUAL_PENDING','counts':counts,'split_disjoint':True,'labels_50_fields':True,'fixed_test_access':False},indent=2));print(counts)
if __name__=='__main__':main()
