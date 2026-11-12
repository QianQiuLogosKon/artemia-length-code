"""Exercise exact archived MMPCF functions on new controlled silhouettes only."""
from pathlib import Path
import ast,csv,hashlib,json
import cv2,numpy as np
from scipy.integrate import quad
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
O=Path(__file__).resolve().parent;D=O/'mmpcf_synthetic_audit';SOURCE=O.parent/'mmpcf_adaptation_20260928/run_mmpcf_adaptation.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def crop_bounds(box,w,h,padding):
    x1,y1,x2,y2=box;dx=padding*(x2-x1);dy=padding*(y2-y1)
    return [max(0,int(np.floor(x1-dx))),max(0,int(np.floor(y1-dy))),min(w,int(np.ceil(x2+dx))),min(h,int(np.ceil(y2+dy)))]
def load_exact_functions():
    tree=ast.parse(SOURCE.read_text(encoding='utf-8'));names=['skeleton','principal','nearest_pair','path_curve','one'];nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(nodes)==5;ns=dict(cv2=cv2,np=np,quad=quad,csr_matrix=csr_matrix,dijkstra=dijkstra,crop_bounds=crop_bounds,DISK=((np.mgrid[-2:3,-2:3]**2).sum(0)<=4).astype(np.uint8))
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SOURCE),'exec'),ns);return ns
def make(family,angle,width):
    if family in ['straight','appendaged']:
        pts=np.column_stack([np.linspace(-75,75,501),np.zeros(501)])
    else:
        alpha=np.deg2rad(60 if family=='arc60' else 120);rad=150/alpha;t=np.linspace(-alpha/2,alpha/2,501);pts=np.column_stack([rad*np.sin(t),rad*(1-np.cos(t))]);pts[:,1]-=(pts[:,1].max()+pts[:,1].min())/2
    theta=np.deg2rad(angle);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]);pts=pts@rot.T+128
    mask=np.zeros((256,256),np.uint8);cv2.polylines(mask,[np.rint(pts).astype(np.int32)],False,1,width,cv2.LINE_8)
    if family=='appendaged':
        for x in [-45,-15,15,45]:
            branch=np.array([[x,0],[x-10,25]])@rot.T+128;cv2.line(mask,tuple(np.rint(branch[0]).astype(int)),tuple(np.rint(branch[1]).astype(int)),1,1)
    im=np.full((256,256,3),235,np.uint8);im[mask>0]=35;yy,xx=np.where(mask);box=np.array([xx.min()-4,yy.min()-4,xx.max()+5,yy.max()+5],float)
    return im,mask,pts,box
def main():
    D.mkdir(exist_ok=False);cv2.setNumThreads(1);ns=load_exact_functions();original_skel=ns['skeleton'];original_path=ns['path_curve'];trace={}
    def skeleton(mask):
        out=original_skel(mask);trace['skeleton']=out.astype(np.uint8);trace['aligned']=mask;return out
    def path_curve(skel,start,end,silhouette):
        trace['start']=start;trace['end']=end;return original_path(skel,start,end,silhouette)
    ns['skeleton']=skeleton;ns['path_curve']=path_curve
    protocol={'status':'frozen_before_execution','source_sha256':sha(SOURCE),'scope':'synthetic image controls only, no true Artemia performance estimate','functions':'Exact original function AST bodies, including one, skeleton, path_curve; source module main is never executed','families':['straight','arc60','arc120','appendaged'],'angles':[0,30,60,90],'source_widths':[3,5,9,15,25],'rasterization':'OpenCV LINE_8 tubes; appendages fixed four one-pixel branches; uniform background235 foreground35','reference':'synthetic silhouette bounding box +4px; existing padding.15 and resize256 preserved','metrics':'success/failure, skeleton components and native length vs intended150px spine; cap/extrema definition differs, length bias is diagnostic only','opening_radius':2,'skeleton_radius':2,'graph_radius':2,'parameter_search':False,'real_images_read':False,'test_access':False};save(D/'protocol.json',protocol);rows=[]
    for family in protocol['families']:
      for angle in protocol['angles']:
       for width in protocol['source_widths']:
        case=f'{family}_a{angle}_w{width}';trace.clear();im,mask,truth,box=make(family,angle,width);r=ns['one'](im,box,.15);stages=r.pop('_stages',None)
        record={'case_id':case,'family':family,'angle':angle,'source_width':width,'status':r['status'],'native_length_px':r.get('native_length_px'),'signed_difference_from_spine_percent':100*(r['native_length_px']/150-1) if r['status']=='ok' else None,'skeleton_pixels':int(trace['skeleton'].sum()) if 'skeleton' in trace else None,'skeleton_8_components':int(cv2.connectedComponents(trace['skeleton'],8)[0]-1) if 'skeleton' in trace else None}
        rows.append(record);save(D/(case+'_prediction.json'),r)
        payload={'image':im,'silhouette':mask,'truth':truth,'reference_box':box,**trace}
        if stages is not None:payload.update(stages)
        np.savez_compressed(D/(case+'.npz'),**payload)
      print('MMPCF',family,len(rows),flush=True)
    save(D/'results.json',rows)
    with (D/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for family in protocol['families']:
     rr=[r for r in rows if r['family']==family];summary.append({'family':family,'n':len(rr),'status_counts':{s:sum(r['status']==s for r in rr) for s in sorted({r['status'] for r in rr})},'failures':[r['case_id'] for r in rr if r['status']!='ok']})
    save(D/'summary.json',summary);assert len(rows)==80 and sha(SOURCE)==protocol['source_sha256']
    save(D/'verification.json',{'status':'PASS_NUMERIC_VISUAL_PENDING','cases':80,'original_function_bodies_unchanged':True,'source_sha256':sha(SOURCE),'real_images_read':False,'training':False,'model_inference':False,'test_access':False})
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
