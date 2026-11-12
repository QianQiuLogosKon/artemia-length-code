from pathlib import Path
import copy, json, sys, time, platform
import cv2, numpy as np, torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/baseline_hrnet_20260922_v1'
OUT=ROOT/'output/target_context_ablation_20260929'
DATA=BASE/'data_v2'
WEIGHTS=ROOT/'deployment/endpoint_refined_v1/weights'
ARCHIVE=BASE
sys.path.insert(0,str(ARCHIVE/'src'))
from common import read_image, crop_bounds, prompt, arc, sha256
sys.path.insert(0,str(ROOT/'output/baseline_comparison_20260922_v1'))
from infer_yolo_candidates import iou
from train_endpoint_residual_refiner import ANCHOR, EndpointRefiner, endpoint_patch
from evaluate_endpoint_residual_refiner import feature

def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    ims=[x for x in manifest['images'] if x['split']=='val']
    assert len(ims)==81 and sum(len(x['detections']) for x in ims)==1498
    weights={n:sha256(WEIGHTS/n) for n in ['pose_visual_prompt_best.pt','endpoint_refiner_epoch10.pt']}
    pose=YOLO(str(WEIGHTS/'pose_visual_prompt_best.pt')); device=torch.device('cuda:0')
    ref=EndpointRefiner().to(device); payload=torch.load(WEIGHTS/'endpoint_refiner_epoch10.pt',map_location=device,weights_only=True); ref.load_state_dict(payload['model']); ref.eval()
    allout={}; started=time.perf_counter()
    for mode in ("no_prompt", "black_outside"):
        pad=0.15
        if (OUT/f"refined_{mode}.json").exists(): continue
        raw={}; refined={}; pending=[]
        # explicit batch loop avoids hidden GT use; refiner calls are kept deterministic
        for image_number,im in enumerate(ims):
            if image_number%10==0: print(mode,image_number,"/81",flush=True)
            image=read_image(DATA/im['image']); h,w=image.shape[:2]
            for d in im['detections']:
                cid=d['id']; box=np.asarray(d['box'],float); left,top,right,bottom=crop_bounds(box,w,h,pad)
                if right<=left or bottom<=top:
                    raw[cid]=refined[cid]={'status':'empty_crop'}; continue
                crop=image[top:bottom,left:right]; reference=box-[left,top,left,top]; inp=crop.copy()
                if mode=="black_outside":
                    xx1,yy1,xx2,yy2=np.round(reference).astype(int)
                    mask=np.zeros(crop.shape[:2],np.uint8)
                    cv2.rectangle(mask,(max(0,xx1),max(0,yy1)),(min(crop.shape[1]-1,xx2),min(crop.shape[0]-1,yy2)),1,-1)
                    inp[mask==0]=0
                result=pose.predict(inp,device='0',imgsz=1024,conf=.001,iou=.7,max_det=300,half=False,augment=False,rect=True,verbose=False)[0]
                val={'status':'no_target_class_pose','pose_detection_count':0}
                if result.boxes is not None and result.keypoints is not None:
                    boxes=result.boxes.xyxy.cpu().numpy(); cls=result.boxes.cls.cpu().numpy().astype(int); conf=result.boxes.conf.cpu().numpy(); val['pose_detection_count']=len(cls)
                    cand=np.flatnonzero(cls==0)
                    if len(cand):
                        ovs=np.asarray([iou(boxes[j],reference) for j in cand]); ties=cand[ovs==ovs.max()]; chosen=int(ties[np.argmax(conf[ties])]); pts=result.keypoints.xy.cpu().numpy()[chosen]+[left,top]
                        if pts.shape==(15,2) and np.isfinite(pts).all() and arc(pts)>0:
                            val.update(status='ok',points=pts.tolist(),pose_confidence=float(conf[chosen]),pose_reference_iou=float(ovs.max()),pose_box=(boxes[chosen]+[left,top,left,top]).tolist())
                raw[cid]=copy.deepcopy(val); refined[cid]=copy.deepcopy(val)
                if val['status']=='ok':
                    points=np.asarray(val['points'],float); shifts=[0.,0.]; caps=[False,False]; deg=[False,False]
                    for kind,idx,inside in ((0,0,2),(1,14,12)):
                        extracted=endpoint_patch(image,points[idx],points[inside])
                        if extracted is None: deg[kind]=True; continue
                        patch,aff=extracted; x=feature(patch,kind); xt=torch.from_numpy(x).unsqueeze(0).to(device)
                        with torch.inference_mode(): residual=ref(xt).float().cpu().numpy()[0]
                        p=cv2.transform((ANCHOR+residual*64).reshape(1,1,2).astype(np.float32),cv2.invertAffineTransform(aff))[0,0]
                        delta=p-points[idx]; cap=.03*arc(points); mag=float(np.linalg.norm(delta))
                        if mag>cap: delta*=cap/mag; caps[kind]=True
                        refined[cid]['points'][idx]=(points[idx]+delta).tolist(); shifts[kind]=float(np.linalg.norm(delta))
                    refined[cid].update(endpoint_shifts_px=shifts,cap_active=caps,degenerate_patch=deg)
        save(OUT/f'raw_{mode}.json',raw); save(OUT/f'refined_{mode}.json',refined); allout[mode]=dict(raw=raw,refined=refined)
        print(json.dumps({'mode':mode,'padding':pad,'candidates':len(raw),'ok':sum(v['status']=='ok' for v in raw.values()),'seconds':round(time.perf_counter()-started,2)}),flush=True)
    save(OUT/'run_audit.json',{'status':'COMPLETE','elapsed_seconds':time.perf_counter()-started,'training':False,'fixed_test_access':False})
if __name__=='__main__': main()
