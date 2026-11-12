from pathlib import Path
import json,sys,time,hashlib
import numpy as np,torch
from ultralytics import YOLO
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;D=R/'output/baseline_hrnet_20260922_v1/data_v2';read=lambda p:json.loads(p.read_text(encoding='utf8'));A=D.parent;sys.path.insert(0,str(A/'src'));from common import read_image,crop_bounds,arc,iou

def main():
 ck=O/'completed_archive/train/weights/best.pt'
 if not ck.exists():raise SystemExit('Training completion and checkpoint archive required before inference.')
 verified=read(O/'completed_archive/archive_verification.json');assert verified['status']=='PASS' and verified['epochs']==300
 assert hashlib.sha256(ck.read_bytes()).hexdigest()==verified['sha256']['train/weights/best.pt']
 dest=O/'validation_candidates.json'
 if dest.exists():raise SystemExit('Candidate output already exists; refusing duplicate inference')
 model=YOLO(str(ck));ims=[i for i in read(D/'manifest.json')['images'] if i['split']=='val'];out={};assert len(ims)==81
 for im in ims:
  img=read_image(D/im['image']);h,w=img.shape[:2]
  for d in im['detections']:
   l,t,r,b=crop_bounds(d['box'],w,h,.15);reference=np.array(d['box'])-[l,t,l,t];result=model.predict(img[t:b,l:r],imgsz=1024,device=0,conf=.001,iou=.7,max_det=300,half=False,augment=False,rect=True,verbose=False)[0];v={'status':'empty_pose','pose_detection_count':0}
   if result.boxes is not None and result.keypoints is not None:
    boxes=result.boxes.xyxy.cpu().numpy();conf=result.boxes.conf.cpu().numpy();cls=result.boxes.cls.cpu().numpy().astype(int);v['pose_detection_count']=len(cls);indices=np.flatnonzero(cls==0)
    if len(indices):
     ov=np.array([iou(boxes[j],reference) for j in indices]);ties=indices[ov==ov.max()];chosen=int(ties[np.argmax(conf[ties])]);pts=result.keypoints.xy.cpu().numpy()[chosen]+[l,t]
     if pts.shape==(15,2) and np.isfinite(pts).all() and arc(pts)>0:v.update(status='ok',points=pts.tolist(),pose_confidence=float(conf[chosen]),pose_reference_iou=float(ov.max()),length_polyline_px=arc(pts),length_chord_px=float(np.linalg.norm(pts[0]-pts[-1])))
   out[d['id']]=v
 assert len(out)==1498;dest.write_text(json.dumps(out,indent=2),encoding='utf8');(O/'inference_audit.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'candidates':1498,'GT_for_selection':False,'test_access':False,'device':torch.cuda.get_device_name(0),'postprocess':'no endpoint refiner; both native polyline and chord, no metric selection'},indent=2));print('INFERENCE_COMPLETE')
if __name__=='__main__':main()
