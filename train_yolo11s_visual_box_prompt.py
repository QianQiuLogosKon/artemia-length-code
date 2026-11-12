"""YOLO11s fine-tuning with an explicit highlighted target-box prompt."""
from pathlib import Path
import argparse,csv,shutil,cv2,numpy as np,torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils.loss import v8PoseLoss

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'brine_shrimp_yolo_pose_archive_15pt_headtail_fully_reviewed_crops_targetclass_jitter'
DERIVED=ROOT/'brine_shrimp_yolo_pose_archive_15pt_headtail_fully_reviewed_crops_visual_box_prompt'
INIT=ROOT/'runs/brine_yolo11s_hybrid15_continuity/weights/best.pt';RUNS=ROOT/'runs'

def prompt(im,box):
    x1,y1,x2,y2=np.round(box).astype(int); x1=max(0,x1);y1=max(0,y1);x2=min(im.shape[1]-1,x2);y2=min(im.shape[0]-1,y2)
    out=im.copy(); overlay=out.copy(); cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,255),-1); out=cv2.addWeighted(overlay,.16,out,.84,0); cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,255),3); return out
def build():
    for split in ('train','val'):
        (DERIVED/'images'/split).mkdir(parents=True,exist_ok=True);(DERIVED/'labels'/split).mkdir(parents=True,exist_ok=True)
        for ip in (SOURCE/'images'/split).iterdir():
            dst=DERIVED/'images'/split/ip.name
            if dst.exists() and (DERIVED/'labels'/split/f'{ip.stem}.txt').exists():continue
            im=cv2.imdecode(np.fromfile(str(ip),np.uint8),cv2.IMREAD_COLOR); h,w=im.shape[:2]; lines=(SOURCE/'labels'/split/f'{ip.stem}.txt').read_text().splitlines(); target=None
            for line in lines:
                v=np.asarray([float(x) for x in line.split()]);
                if int(v[0])==0: target=v[1:5];break
            if target is None:continue
            cx,cy,bw,bh=target; box=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]
            cv2.imencode('.jpg',prompt(im,box),[cv2.IMWRITE_JPEG_QUALITY,95])[1].tofile(str(DERIVED/'images'/split/ip.name))
            target_line=next(line for line in lines if int(float(line.split()[0]))==0)
            (DERIVED/'labels'/split/f'{ip.stem}.txt').write_text(target_line+'\n',encoding='utf-8')
    rows=list(csv.DictReader((SOURCE/'metadata/targets.csv').open(encoding='utf-8-sig')));train=[]
    for row in rows:
        p=Path(row['image'])
        if row['split']=='train' and '_jit' not in p.stem:train.extend([str(DERIVED/p)]*(3 if int(row['neighbour_instance_count'])>0 else 1))
    tf=DERIVED/'prompt_train.txt';tf.write_text('\n'.join(train)+'\n'); yaml=DERIVED/'visual_prompt.yaml';yaml.write_text(f"path: {DERIVED.as_posix()}\ntrain: {tf.as_posix()}\nval: images/val\n\nnames:\n  0: target_shrimp\nkpt_shape: [15, 3]\nflip_idx: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]\n")
    print('PROMPT_DATA',DERIVED,'DRAWS',len(train),flush=True);return yaml

ORIGINAL=v8PoseLoss.calculate_keypoints_loss
def geometry_loss(self,masks,target_gt_idx,keypoints,batch_idx,stride_tensor,target_bboxes,pred_kpts):
    loc,obj=ORIGINAL(self,masks,target_gt_idx,keypoints,batch_idx,stride_tensor,target_bboxes,pred_kpts)
    if masks.any():
        selected=self._select_target_keypoints(keypoints,batch_idx,target_gt_idx,masks);selected[...,:2]/=stride_tensor.view(1,-1,1,1)
        gt=selected[masks];pred=pred_kpts[masks];visible=gt[...,2]!=0 if gt.shape[-1]==3 else torch.ones_like(gt[...,0],dtype=torch.bool);pair=visible[:,1:]&visible[:,:-1]
        if pair.any():
            ps=pred[:,1:,:2]-pred[:,:-1,:2];gs=gt[:,1:,:2]-gt[:,:-1,:2];gl=gs.norm(dim=-1);pl=ps.norm(dim=-1)
            scale=gl.masked_fill(~pair,float('nan')).nanmedian(dim=1).values.clamp_min(.25);e=(ps-gs)/scale[:,None,None]
            continuity=F.smooth_l1_loss(e[pair],torch.zeros_like(e[pair]),beta=.5)
            pred_arc=(pl*pair).sum(1);gt_arc=(gl*pair).sum(1).clamp_min(.5);arc=F.smooth_l1_loss(pred_arc/gt_arc,torch.ones_like(pred_arc),beta=.1)
            endpoint=((pred[:,0,:2]-gt[:,0,:2]).norm(dim=-1)+(pred[:,-1,:2]-gt[:,-1,:2]).norm(dim=-1))/(2*scale)
            loc=loc+.10*continuity+.12*arc+.04*endpoint.mean()
    return loc,obj
v8PoseLoss.calculate_keypoints_loss=geometry_loss
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args();data=build();model=YOLO(str(INIT));model.train(data=str(data),epochs=3 if a.smoke else 40,imgsz=1024,batch=4,device=0,workers=4,amp=True,optimizer='AdamW',cos_lr=True,fliplr=.5,flipud=0.,degrees=5.,translate=.03,scale=.08,mosaic=0.,mixup=0.,erasing=0.,pose=15.,kobj=1.,lr0=8e-5,lrf=.05,warmup_epochs=1,patience=15,save_period=5,seed=20260920,project=str(RUNS),name='brine_yolo11s_visual_box_prompt_smoke' if a.smoke else 'brine_yolo11s_visual_box_prompt',exist_ok=False)
if __name__=='__main__':main()
