"""MMPCF multi-object adaptation, validation only.

The article's MMPCF description is implemented as an explicit adaptation:
detector candidate crop -> Otsu silhouette -> moment alignment -> radius-2
opening and morphological skeleton -> shortest skeleton path -> degree-4 fit.
The paper does not specify a multi-object crop/selection rule, opening kernel,
or endpoint tie rules; all such choices are recorded in protocol.json.
No GT is read by this script.
"""
from pathlib import Path
import json,sys,time,platform,hashlib
import cv2,numpy as np
from scipy import ndimage
from scipy.integrate import quad
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
BASE=ROOT/'output/baseline_hrnet_20260922_v1'; DATA=BASE/'data_v2'
ARCHIVE=BASE
sys.path.insert(0,str(ARCHIVE/'src')); from common import read_image,crop_bounds,arc,sha256

def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
DISK=((np.mgrid[-2:3,-2:3]**2).sum(0)<=4).astype(np.uint8)
def skeleton(mask):
    # Blum morphological skeleton with disk radius 2, following Eq. 17.
    se=DISK; cur=mask.copy(); out=np.zeros_like(mask)
    while cur.any():
        er=cv2.erode(cur,se,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        opened=cv2.dilate(er,se,borderType=cv2.BORDER_CONSTANT,borderValue=0)
        out |= (cur & ~opened); cur=er
    return out>0
def principal(mask):
    yy,xx=np.nonzero(mask)
    if len(xx)<20:return None
    c=np.cov(np.stack([xx,yy])); val,vec=np.linalg.eigh(c); direction=vec[:,np.argmax(val)];
    if direction[0]<0:direction=-direction
    theta=float(np.arctan2(direction[1],direction[0])); center=np.array([xx.mean(),yy.mean()])
    # rotate coordinates into horizontal principal-axis frame (no interpolation).
    R=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    q=(np.stack([xx,yy],1)-center)@R.T
    return theta,center,R,q
def nearest_pair(coords,target):
    return int(np.argmin(np.sum((coords-target)**2,axis=1)))
def path_curve(skel, start, end, silhouette):
    yy,xx=np.nonzero(skel); coords=np.stack([xx,yy],1); n=len(coords)
    if n<10:return None
    index={tuple(p):i for i,p in enumerate(coords)}; edges=[]; weights=[]
    for i,(x,y) in enumerate(coords):
        for dx,dy in [(dx,dy) for dy in range(-2,3) for dx in range(-2,3) if (dy>0 or dy==0 and dx>0)]:
            j=index.get((x+dx,y+dy));
            if j is not None:
                # Explicit bounded digital-skeleton bridging, never outside silhouette.
                samples=np.round(np.linspace([x,y],[x+dx,y+dy],5)).astype(int)
                if np.all((samples[:,0]>=0)&(samples[:,1]>=0)&(samples[:,0]<silhouette.shape[1])&(samples[:,1]<silhouette.shape[0])) and np.all(silhouette[samples[:,1],samples[:,0]]):
                    edges += [(i,j),(j,i)]; weights += [float(np.hypot(dx,dy))]*2
    if not edges:return None
    graph=csr_matrix((weights,(np.array(edges)[:,0],np.array(edges)[:,1])),shape=(n,n)); a=nearest_pair(coords,start); b=nearest_pair(coords,end)
    dist,prev=dijkstra(graph,directed=True,indices=a,return_predecessors=True); 
    if not np.isfinite(dist[b]):return None
    ids=[b]
    while ids[-1]!=a:
        p=int(prev[ids[-1]]);
        if p<0:return None
        ids.append(p)
    return coords[ids[::-1]].astype(float)
def one(image,box,pad=.15):
    h,w=image.shape[:2]; l,t,r,b=crop_bounds(box,w,h,pad); crop=image[t:b,l:r]
    if crop.size==0:return dict(status='empty_crop')
    scale=256/max(crop.shape[:2]); nw=max(4,round(crop.shape[1]*scale)); nh=max(4,round(crop.shape[0]*scale))
    resized=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_AREA)
    sx=nw/crop.shape[1]; sy=nh/crop.shape[0]
    ref=(box-[l,t,l,t])*[sx,sy,sx,sy]
    gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY); threshold,bw=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    border=np.zeros_like(bw,dtype=bool); border[[0,-1],:]=True; border[:,[0,-1]]=True
    # Foreground is the Otsu side less frequent on crop border; ties choose dark.
    dark=bw==0; mask=dark if dark[border].mean()<=.5 else ~dark
    choices=[]
    n,lab,stats,cent=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    for k in range(1,n):
        x,y,ww,hh,area=stats[k]; cb=[x,y,x+ww,y+hh]
        inter=max(0,min(cb[2],ref[2])-max(cb[0],ref[0]))*max(0,min(cb[3],ref[3])-max(cb[1],ref[1]))
        union=ww*hh+(ref[2]-ref[0])*(ref[3]-ref[1])-inter
        if area>=20 and inter>0:choices.append(((inter/max(union,1e-9),area,-k),k))
    if not choices:return dict(status='no_silhouette')
    selected=max(choices,key=lambda x:x[0])[1]; mask=(lab==selected).astype(np.uint8)
    info=principal(mask)
    if info is None:return dict(status='short_silhouette')
    theta,center,R,q=info
    corners=(np.array([[0,0],[nw-1,0],[0,nh-1],[nw-1,nh-1]])-center)@R.T
    offset=-np.floor(corners.min(0))+4
    shape=np.ceil(corners.max(0)+offset+5).astype(int)
    affine=np.column_stack([R,offset-R@center]).astype(np.float64)
    aligned=cv2.warpAffine(mask,affine,tuple(shape),flags=cv2.INTER_NEAREST,borderValue=0)
    opened=cv2.morphologyEx(aligned,cv2.MORPH_OPEN,DISK,borderType=cv2.BORDER_CONSTANT,borderValue=0)
    oy,ox=np.nonzero(opened)
    if len(ox)<20:return dict(status='opening_empty')
    left=int(ox.min()); right=int(ox.max())
    start=np.array([left,float(np.median(oy[ox==left]))]); end=np.array([right,float(np.median(oy[ox==right]))])
    sk=skeleton(aligned); yy,xx=np.nonzero(sk)
    if len(xx)<10:return dict(status='short_skeleton')
    path=path_curve(sk,start,end,aligned)
    if path is None or len(path)<8:return dict(status='no_path')
    qp=np.vstack([start,path,end])
    if np.ptp(qp[:,0])<5 or len(np.unique(qp[:,0]))<5:return dict(status='degenerate_path')
    polynomial=np.polynomial.Polynomial.fit(qp[:,0],qp[:,1],4); derivative=polynomial.deriv()
    def transform(points):return ((points-offset)@R+center)/[sx,sy]+[l+(0.5/sx-0.5),t+(0.5/sy-0.5)]
    def curve(n):
        xs=np.linspace(left,right,n); return transform(np.stack([xs,polynomial(xs)],1))
    def speed(x):return float(np.linalg.norm((np.array([1.,derivative(x)])@R)/[sx,sy]))
    length,err=quad(speed,left,right,epsabs=1e-6,epsrel=1e-9)
    original=curve(15); dense=curve(257)
    if not np.isfinite(original).all() or not np.isfinite(length) or length<=0:return dict(status='invalid_curve')
    value=dict(status='ok',points=original.tolist(),curve_257=dense.tolist(),native_length_px=float(length),quadrature_error_px=float(err),silhouette_area=int(mask.sum()),skeleton_pixels=int(len(path)),polynomial_degree=4,crop_bounds=[l,t,r,b],threshold=float(threshold),polarity='dark' if dark[border].mean()<=.5 else 'bright',component_count=n-1,selected_component=int(selected),orientation_radians=theta,
        coefficients=polynomial.coef.tolist(),polynomial_domain=polynomial.domain.tolist(),integration_domain=[left,right],R=R.tolist(),center=center.tolist(),offset=offset.tolist(),resize_scale_xy=[sx,sy],aligned_endpoints=[start.tolist(),end.tolist()])
    value['_stages']=dict(gray=gray,mask=mask,aligned=aligned,opened=opened,skeleton=sk.astype(np.uint8),path=path)
    return value

def main():
    OUT.mkdir(exist_ok=True); manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8')); ims=[x for x in manifest['images'] if x['split']=='val']; assert len(ims)==81
    assert not (OUT/'predictions_all_candidates.json').exists(),'Refusing duplicate inference'
    protocol={'method':'MMPCF multi-object adaptation v1','paper_doi':'10.1016/j.compag.2019.105102','split':'val','detector_candidates_reused':True,'detector_rerun':False,'padding':.15,'candidate_count':1498,'detector_threshold':.001,'gt_usage':'none during inference; GT only in offline evaluation','steps':['Otsu grayscale silhouette','moment principal-axis horizontal alignment','radius-2 disk opening for extremal endpoints','radius-2 morphological skeleton of full aligned silhouette','bounded-neighborhood shortest skeleton path','least-squares degree-4 polynomial','native polynomial arc-length integration in original coordinates'],'adaptation_choices':{'crop':'same cached detector crop, padding .15; raw pixels WITHOUT yellow prompt (intensity segmentation baseline, not prompt ablation)','resolution':'longest crop side 256, aspect preserved; minimum side 4; pixel-center inverse transform','polarity':'choose Otsu side occupying <=50% of crop border; tie dark','component':'max bbox IoU to detector reference; ties area then component index; at least20 pixels and positive detector intersection','opening':'exact Euclidean integer disk radius2; unspecified by paper for appendage opening','rotation':'covariance largest eigenvector; nearest-neighbor raster, zero border','multi_object':'all detector candidates independently; no GT','endpoint':'left/right extrema of opened silhouette, tied y median; snapped to nearest skeleton pixel','skeleton_graph':'Chebyshev radius2 edges with Euclidean weight, 5 rounded segment samples inside silhouette; explicit bounded gap bridging for digital morphological skeleton; disconnected remains failure','fit':'all shortest path pixels plus two opened-silhouette extrema; degree4 least squares with normalized x; integrate between opened extrema','length':'scipy quad native polynomial in original coordinates; 15 and257 sample curves are diagnostic, not length metric','head_tail':'not provided; direction accuracy N/A'},'limitations':['Not author-code reproduction; repository snapshot contains data/license only.','Original paper has single-object resized256x256 images; multi-target selection and graph/opening details are adaptations.','No validation search or rejection based on GT error.'],'training':False,'fixed_test_access':False,'environment':{'python':platform.python_version(),'opencv':cv2.__version__,'numpy':np.__version__,'scipy':__import__('scipy').__version__}}
    save(OUT/'protocol.json',protocol); predictions={}; started=time.perf_counter(); (OUT/'stages').mkdir(exist_ok=True)
    inputs={str(DATA/'manifest.json'):sha256(DATA/'manifest.json'),str(Path(__file__).resolve()):sha256(__file__)}
    for ni,im in enumerate(ims,1):
        p=(DATA/im['image']).resolve(); assert p.is_relative_to((DATA/'images/val').resolve()); assert sha256(p)==im['image_sha256']; inputs[str(p)]=sha256(p)
        image=read_image(DATA/im['image'])
        for d in im['detections']:
            v=one(image,np.asarray(d['box'],float),.15); stages=v.pop('_stages',None)
            if stages is not None:np.savez_compressed(OUT/'stages'/(d['id'].replace(':','_')+'.npz'),**stages)
            predictions[d['id']]=v
        if ni%10==0:print(json.dumps({'images':ni,'candidates':len(predictions),'ok':sum(x['status']=='ok' for x in predictions.values()),'seconds':round(time.perf_counter()-started,1)}),flush=True)
    assert len(predictions)==1498; save(OUT/'predictions_all_candidates.json',predictions); save(OUT/'input_sha256.json',inputs); save(OUT/'run_audit.json',{'status':'PASS','candidates':len(predictions),'elapsed_seconds':time.perf_counter()-started,'training_performed':False,'fixed_test_access':False,'gt_used_during_inference':False})
    print(json.dumps({'status':'PASS','candidates':len(predictions),'ok':sum(x['status']=='ok' for x in predictions.values()),'seconds':round(time.perf_counter()-started,1)}),flush=True)
if __name__=='__main__':main()
