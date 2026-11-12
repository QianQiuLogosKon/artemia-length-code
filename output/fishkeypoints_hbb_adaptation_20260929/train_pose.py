from pathlib import Path
import json,os,sys,time,traceback,hashlib
from ultralytics import YOLO
O=Path(__file__).resolve().parent;os.chdir(O);smoke='--smoke' in sys.argv;name='smoke' if smoke else 'train';run=O/name
if run.exists():raise SystemExit('Refuse overwrite existing run '+str(run))
run.mkdir();status=run/'status.json'
def write(obj):
 p=status.with_suffix('.tmp');p.write_text(json.dumps(obj,indent=2));p.replace(status)
def epoch_end(t):
 write({'status':'running','epoch':t.epoch+1,'total_epochs':t.epochs,'pid':os.getpid(),'updated_unix':time.time(),'save_dir':str(t.save_dir),'best_fitness':float(t.best_fitness) if t.best_fitness is not None else None,'metrics':{k:float(v) for k,v in t.metrics.items()}})
try:
 write({'status':'starting','pid':os.getpid(),'updated_unix':time.time()})
 model=YOLO(str(O/'yolov8s-pose.pt'));model.add_callback('on_fit_epoch_end',epoch_end)
 # Absolute dataset path avoids user-specific Ultralytics dataset roots.
 import yaml
 data=yaml.safe_load((O/'data.yaml').read_text());data['path']=str(O/'dataset');(O/'data_runtime.yaml').write_text(yaml.safe_dump(data))
 config=dict(data=str(O/'data_runtime.yaml'),epochs=1 if smoke else 300,imgsz=1024,batch=8,device=0,workers=4,optimizer='AdamW',lr0=.001,momentum=.9,cos_lr=True,seed=20260929,deterministic=True,patience=0,amp=False,project=str(O),name=name,exist_ok=True,pretrained=True,degrees=45,flipud=.5,fliplr=.5,mixup=.1,mosaic=1.,hsv_h=.015,hsv_s=.7,hsv_v=.4,plots=True,save=True,save_period=25,cache=False,fraction=.05 if smoke else 1.)
 (run/'requested_config.json').write_text(json.dumps(config,indent=2));model.train(**config)
 write({'status':'completed','pid':os.getpid(),'epoch':config['epochs'],'total_epochs':config['epochs'],'updated_unix':time.time(),'best_sha256':hashlib.sha256((run/'weights/best.pt').read_bytes()).hexdigest()})
except Exception as e:
 write({'status':'failed','pid':os.getpid(),'error':repr(e),'traceback':traceback.format_exc(),'updated_unix':time.time()});raise
