"""Validation detector crops only; no GT read by model preprocessing."""
from pathlib import Path
import hashlib,json,time
import cv2,numpy as np,torch
from unet_shared import UNet
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
DATA=ROOT/'output/baseline_hrnet_20260922_v1/data_v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    weight=OUT/'completed_archive_20260928/train/best.pt'
    assert sha(weight)=='79c8e682f3ee0f78daebe31b8119f405bae5d827e20ac28bf597edfc36cfa5c4'
    manifest=DATA/'manifest.json';assert sha(manifest)=='6631a3de2ee64f6600848a8593daff86f46e1d8274e961647d976dae53c40fa8'
    # Strip GT fields before the inference loop.
    images=[{k:im[k] for k in ['id','image','image_sha256','detections']} for im in json.loads(manifest.read_text(encoding='utf-8'))['images'] if im['split']=='val']
    assert len(images)==81 and sum(len(im['detections']) for im in images)==1498
    dest=OUT/'probability_cache_v1';dest.mkdir(exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(1);model=UNet().cuda().eval();payload=torch.load(weight,map_location='cpu',weights_only=False);model.load_state_dict(payload['model'])
    records=[];start=time.time()
    for im in images:
        src=DATA/im['image'];assert sha(src)==im['image_sha256'];image=cv2.imdecode(np.fromfile(str(src),np.uint8),1);h,w=image.shape[:2]
        for d in im['detections']:
            x1,y1,x2,y2=d['box'];dx=.15*(x2-x1);dy=.15*(y2-y1)
            l=max(0,int(np.floor(x1-dx)));t=max(0,int(np.floor(y1-dy)));r=min(w,int(np.ceil(x2+dx)));b=min(h,int(np.ceil(y2+dy)))
            crop=image[t:b,l:r];assert crop.size
            scale=256/max(crop.shape[:2]);nw=max(8,round(crop.shape[1]*scale));nh=max(8,round(crop.shape[0]*scale));ox=(256-nw)//2;oy=(256-nh)//2
            canvas=np.zeros((256,256,3),np.uint8);canvas[oy:oy+nh,ox:ox+nw]=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA)
            x=torch.from_numpy(canvas[:,:,::-1].transpose(2,0,1).copy()).cuda().float()[None]/255
            with torch.inference_mode():p=model(x).sigmoid()[0,0].cpu().numpy()
            assert np.isfinite(p).all();name=d['id'].replace(':','__')+'.npy';np.save(dest/name,p)
            records.append(dict(id=d['id'],image_id=im['id'],box=d['box'],confidence=d['confidence'],crop=[l,t,r,b],resize=[nw,nh],offset=[ox,oy],probability=name,sha256=sha(dest/name)))
        (dest/'progress.json').write_text(json.dumps({'image':im['id'],'candidates':len(records),'elapsed_seconds':time.time()-start}))
    (dest/'index.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    (dest/'verification.json').write_text(json.dumps(dict(status='PASS',candidates=len(records),images=len(images),weight_sha256=sha(weight),manifest_sha256=sha(manifest),dtype='float32',batch=1,gt_in_inference=False,fixed_test_access=False,elapsed_seconds=time.time()-start),indent=2))
    print('CACHE_COMPLETE',len(records),flush=True)
if __name__=='__main__':main()
