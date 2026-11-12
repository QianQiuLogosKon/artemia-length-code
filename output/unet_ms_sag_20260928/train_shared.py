"""Train once on train ribbons, select by image-level training holdout loss.
Formal val is NEVER loaded here. Both future decoders must share best.pt.
"""
from pathlib import Path
import argparse,csv,json,time,random,hashlib,os,sys
import numpy as np,cv2,torch
from unet_shared import UNet,weighted_loss

ROOT=Path(__file__).resolve().parent
def save(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def read_image(path):
    result=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_UNCHANGED)
    if result is None:raise ValueError(path)
    return result
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    run=ROOT/('smoke' if args.smoke else 'train');run.mkdir(exist_ok=False)
    seed=20260928;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    assert torch.cuda.is_available();device=torch.device('cuda:0')
    with (ROOT/'ribbon_manifest_v2.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    groups={k:[r for r in rows if r['split']==k] for k in ['train','holdout']}
    assert {r['source_image_id'] for r in groups['train']}.isdisjoint(r['source_image_id'] for r in groups['holdout'])
    if args.smoke:groups={k:v[:8] for k,v in groups.items()}
    arrays={}
    for k,rr in groups.items():
        images=np.stack([read_image(ROOT/r['image_path'])[:,:,::-1] for r in rr]); masks=np.stack([read_image(ROOT/r['label_path']) for r in rr])
        assert images.shape[1:]==(256,256,3) and set(np.unique(masks))=={0,255}
        arrays[k]=(images,masks)
    model=UNet().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999));rng=np.random.default_rng(seed)
    config=dict(seed=seed,epochs=1 if args.smoke else 200,batch=4,lr=1e-4,optimizer='Adam',betas=[.9,.999],precision='FP32',holdout_selection='minimum weighted BCE; no MS/SAG or formal val used to select',augmentation='random D4 rotations/reflections; explicit reduced augmentation vs article',parameters=sum(p.numel() for p in model.parameters()),gpu=torch.cuda.get_device_name(),torch=torch.__version__,train_n=len(groups['train']),holdout_n=len(groups['holdout']),formal_val_access=False,smoke=args.smoke)
    save(run/'config.json',config);metrics=[];best=float('inf');start=time.time()
    def batch(k,ids,aug):
        xx,yy=arrays[k];x=xx[ids].copy();y=yy[ids].copy()
        if aug:
            for j in range(len(ids)):
                n=int(rng.integers(4));x[j]=np.rot90(x[j],n);y[j]=np.rot90(y[j],n)
                if rng.random()<.5:x[j]=np.flip(x[j],1);y[j]=np.flip(y[j],1)
        return torch.from_numpy(x.transpose(0,3,1,2).copy()).to(device).float()/255,torch.from_numpy(y[:,None].copy()).to(device).float()/255
    for epoch in range(1,config['epochs']+1):
        model.train();total=0.;seen=0;order=rng.permutation(len(groups['train']))
        for step,first in enumerate(range(0,len(order),4),1):
            ids=order[first:first+4];x,y=batch('train',ids,True);optimizer.zero_grad(set_to_none=True);loss=weighted_loss(model(x),y)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(ids);seen+=len(ids)
            if step%25==0:save(run/'status.json',dict(status='running',phase='train',epoch=epoch,total_epochs=config['epochs'],batch=step,pid=os.getpid(),updated_unix=time.time()))
        model.eval();vtotal=0.;vseen=0
        with torch.inference_mode():
            for first in range(0,len(groups['holdout']),4):
                ids=np.arange(first,min(first+4,len(groups['holdout'])));x,y=batch('holdout',ids,False);loss=weighted_loss(model(x),y)
                if not torch.isfinite(loss):raise ValueError('Nonfinite holdout loss')
                vtotal+=float(loss)*len(ids);vseen+=len(ids)
        record=dict(epoch=epoch,training_loss=total/seen,holdout_loss=vtotal/vseen,seen_targets=seen,holdout_targets=vseen,elapsed_seconds=time.time()-start)
        if record['holdout_loss']<best:
            best=record['holdout_loss'];torch.save(dict(model=model.state_dict(),epoch=epoch,config=config,holdout_loss=best),run/'best.pt');save(run/'best_summary.json',record)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,config=config),run/'last.pt')
        metrics.append(record);save(run/'metrics.json',metrics);print(json.dumps(record),flush=True)
        save(run/'status.json',dict(status='completed' if epoch==config['epochs'] else 'running',phase='epoch_complete',epoch=epoch,total_epochs=config['epochs'],pid=os.getpid(),updated_unix=time.time(),best_holdout_loss=best,elapsed_seconds=time.time()-start))
if __name__=='__main__':
    try:main()
    except Exception as error:
        dest=ROOT/('smoke' if '--smoke' in sys.argv else 'train')
        if dest.exists():save(dest/'failure.json',dict(error=repr(error),time=time.time()))
        raise
