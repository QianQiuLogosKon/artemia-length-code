"""Deterministic MS/SAG decoders for a shared UNet probability map.

This module contains only image-space decoding. It does not select detector
candidates and it never reads ground truth. The grid is an explicit adaptation
of Wang et al. (2020): scales 2--5, control scale 4, and 16 directions.
"""
from pathlib import Path
import json,cv2,numpy as np

SCALES=(2.,3.,4.,5.)
SIGMA_CON=4.
DIRECTIONS=16

def _skeletonize(binary):
    """Zhang-Suen thinning, returning uint8 0/1."""
    a=(binary>0).astype(np.uint8); changed=True
    while changed:
        changed=False
        for phase in (0,1):
            p=np.pad(a,1); c=p[1:-1,1:-1]
            n=[p[:-2,1:-1],p[:-2,2:],p[1:-1,2:],p[2:,2:],p[2:,1:-1],p[2:,:-2],p[1:-1,:-2],p[:-2,:-2]]
            B=sum(n); A=sum(((n[i]==0)&(n[(i+1)%8]==1)) for i in range(8))
            if phase==0: m=(c==1)&(B>=2)&(B<=6)&(A==1)&((n[0]*n[2]*n[4])==0)&((n[2]*n[4]*n[6])==0)
            else: m=(c==1)&(B>=2)&(B<=6)&(A==1)&((n[0]*n[2]*n[6])==0)&((n[0]*n[4]*n[6])==0)
            if np.any(m): a[m]=0;changed=True
    return a

def ms(prob):
    p=np.clip(np.asarray(prob,float),0,1)
    u=np.uint8(np.round(p*255)); _,b=cv2.threshold(u,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return _skeletonize(b)

def _sag_kernel(sigma,theta):
    phi=max(SIGMA_CON**2/sigma**2,1.)
    radius=int(np.ceil(4*sigma*phi)); y,x=np.mgrid[-radius:radius+1,-radius:radius+1].astype(float)
    u=np.cos(theta)*x+np.sin(theta)*y; v=-np.sin(theta)*x+np.cos(theta)*y
    phi=max(SIGMA_CON**2/sigma**2,1.)
    g=np.exp(-(u*u+v*v/(phi*phi))/(2*sigma*sigma))/(2*np.pi*phi*sigma*sigma)
    return -(3*np.sqrt(3)/2)*sigma*sigma*(u*u/sigma**4-1/sigma**2)*g

def sag(prob):
    p=np.asarray(prob,float); best=np.full(p.shape,-np.inf,float); direction=np.zeros(p.shape,float)
    for sigma in SCALES:
        for j in range(DIRECTIONS):
            theta=np.pi*j/DIRECTIONS; r=cv2.filter2D(p.astype(np.float32),cv2.CV_32F,_sag_kernel(sigma,theta).astype(np.float32),borderType=cv2.BORDER_REPLICATE)
            take=r>best; best[take]=r[take];direction[take]=theta
    # theta is the derivative (normal) direction in Eq.6, not the tangent.
    yy,xx=np.indices(p.shape); nx=np.cos(direction); ny=np.sin(direction)
    x1=np.clip(np.rint(xx+nx).astype(int),0,p.shape[1]-1); y1=np.clip(np.rint(yy+ny).astype(int),0,p.shape[0]-1)
    x2=np.clip(np.rint(xx-nx).astype(int),0,p.shape[1]-1); y2=np.clip(np.rint(yy-ny).astype(int),0,p.shape[0]-1)
    _,support=cv2.threshold(np.uint8(np.round(np.clip(p,0,1)*255)),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    keep=(best>=best[y1,x1])&(best>=best[y2,x2])&(best>0)&(support>0)
    return _skeletonize(keep)

def synthetic_check():
    y,x=np.mgrid[:256,:256]; line=np.exp(-((y-(100+.25*x))**2)/(2*5**2)); a=ms(line); b=sag(line)
    for skeleton in [a,b]:
        yy,xx=np.where(skeleton>0); interior=(xx>20)&(xx<235)
        assert 150<int(interior.sum())<400
        assert np.quantile(np.abs(yy[interior]-(100+.25*xx[interior])),.95)<2
    assert not ms(np.zeros((256,256))).any() and not sag(np.zeros((256,256))).any()
    return {'status':'PASS','ms_pixels':int(a.sum()),'sag_pixels':int(b.sum()),'scales':list(SCALES),'sigma_con':SIGMA_CON,'directions':DIRECTIONS}

if __name__=='__main__':
    out=synthetic_check(); print(json.dumps(out,indent=2)); Path(__file__).with_name('decoder_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
