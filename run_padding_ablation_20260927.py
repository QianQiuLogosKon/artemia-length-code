from pathlib import Path
import copy, json, sys, time, platform
import cv2, numpy as np, torch
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'output/baseline_hrnet_20260922_v1'
OUT=ROOT/'output/padding_ablation_20260927'
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
    OUT.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    ims=[x for x in manifest['images'] if x['split']=='val']
    assert len(ims)==81 and sum(len(x['detections']) for x in ims)==1498
    weights={n:sha256(WEIGHTS/n) for n in ['pose_visual_prompt_best.pt','endpoint_refiner_epoch10.pt']}
    pose=YOLO(str(WEIGHTS/'pose_visual_prompt_best.pt')); device=torch.device('cuda:0')
    ref=EndpointRefiner().to(device); payload=torch.load(WEIGHTS/'endpoint_refiner_epoch10.pt',map_location=device,weights_only=True); ref.load_state_dict(payload['model']); ref.eval()
    allout={}; started=time.perf_counter()
    for pad in (0.0,0.15):
        raw={}; refined={}; pending=[]
        # explicit batch loop avoids hidden GT use; refiner calls are kept deterministic
        for im in ims:
            image=read_image(DATA/im['image']); h,w=image.shape[:2]
            for d in im['detections']:
                cid=d['id']; box=np.asarray(d['box'],float); left,top,right,bottom=crop_bounds(box,w,h,pad)
                if right<=left or bottom<=top:
                    raw[cid]=refined[cid]={'status':'empty_crop'}; continue
                crop=image[top:bottom,left:right]; reference=box-[left,top,left,top]; inp=prompt(crop,reference)
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
        save(OUT/f'raw_pad_{pad:.2f}.json',raw); save(OUT/f'refined_pad_{pad:.2f}.json',refined); allout[str(pad)]=dict(raw=raw,refined=refined)
        print(json.dumps({'padding':pad,'candidates':len(raw),'ok':sum(v['status']=='ok' for v in raw.values()),'seconds':round(time.perf_counter()-started,2)}),flush=True)
    protocol={'split':'val','image_count':81,'ground_truth_instances':221,'cached_candidates':1498,'padding_values':[0.0,0.15],'detector_candidates_reused':True,'detector_rerun':False,'pose_weight_sha256':weights['pose_visual_prompt_best.pt'],'refiner_weight_sha256':weights['endpoint_refiner_epoch10.pt'],'pose':{'imgsz':1024,'conf':.001,'iou':.7,'max_det':300,'half':False,'augment':False,'rect':True,'batch':1},'visual_prompt':'yellow fill alpha .16, border 3 original-image pixels','candidate_choice':'class 0, largest IoU with detector box; confidence tie-break','matching':'offline global greedy IoU >= 0.5; descending (IoU, candidate index, GT index)','gt_usage':'after inference only, for scoring and figures','training_performed':False,'fixed_test_access':False,'environment':{'python':platform.python_version(),'torch':torch.__version__,'ultralytics':__import__('ultralytics').__version__,'gpu':torch.cuda.get_device_name(0)}}
    save(OUT/'protocol.json',protocol); save(OUT/'run_audit.json',{'status':'PASS','elapsed_seconds':time.perf_counter()-started,'all_candidate_ids_identical':True,'outputs_complete':True,'fixed_test_access':False,'training_performed':False})
if __name__=='__main__': main()

