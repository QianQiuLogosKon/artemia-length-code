"""Candidate component selection only; never receives target annotations."""
from pathlib import Path
import sys,cv2,numpy as np
BASE=Path(__file__).resolve().parents[1]/'unet_ms_sag_20260928'
sys.path.insert(0,str(BASE))
from evaluate_decoders_v2 import extract as original_extract

def box_iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
    return float(inter/union) if union>0 else 0.

def select(mask,prob,reference_box,rule):
    assert rule in ['mass_v2','box_component_v1']
    count,labels=cv2.connectedComponents(mask.astype(np.uint8),8)
    if count<2:return [],{'reason':'empty','components':0,'selected_label':None,'selection_iou':None}
    masses=np.bincount(labels.ravel(),weights=prob.ravel());choices=[]
    for label in range(1,count):
        yy,xx=np.where(labels==label);box=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]
        ov=box_iou(box,reference_box)
        score=(float(masses[label]),-label) if rule=='mass_v2' else (ov,float(masses[label]),-label)
        choices.append((score,label,ov))
    _,label,ov=max(choices);path,diag=original_extract((labels==label).astype(np.uint8),prob)
    diag.update(components=count-1,selected_label=label,selection_iou=ov)
    return path,diag
