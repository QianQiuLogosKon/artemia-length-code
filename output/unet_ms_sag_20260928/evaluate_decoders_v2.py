"""Fixed validation evaluation; adaptation choices recorded before decoding."""
from pathlib import Path
import json,hashlib,heapq,csv,time
import cv2,numpy as np
from decode_ms_sag import ms,sag
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];CACHE=OUT/'probability_cache_v1'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def extract(mask,p):
    count,labels=cv2.connectedComponents(mask.astype(np.uint8),8)
    if count<2:return [],{'reason':'empty','components':0}
    # One component per detector crop, chosen by total model probability only.
    masses=np.bincount(labels.ravel(),weights=p.ravel());masses[0]=-1;k=int(masses.argmax())
    yy,xx=np.where(labels==k);nodes=set(zip(xx.tolist(),yy.tolist()))
    if len(nodes)<2:return [],{'reason':'single_pixel','components':count-1}
    adj={}
    for x,y in nodes:
        neighbors=[]
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                q=(x+dx,y+dy)
                if not(dx or dy) or q not in nodes:continue
                neighbors.append((q,float(np.hypot(dx,dy))))
        adj[(x,y)]=sorted(neighbors)
    endpoints=sorted(n for n in nodes if len(adj[n])==1)
    def walk(start):
        dist={start:0.};prev={};heap=[(0.,start)]
        while heap:
            d,u=heapq.heappop(heap)
            if d!=dist[u]:continue
            for v,w in adj[u]:
                nd=d+w
                if nd<dist.get(v,float('inf')):dist[v]=nd;prev[v]=u;heapq.heappush(heap,(nd,v))
        end=max(sorted(dist),key=dist.get);return end,dist,prev
    a,_,_=walk(endpoints[0] if endpoints else min(nodes));b,dist,prev=walk(a)
    path=[b]
    while path[-1]!=a:path.append(prev[path[-1]])
    path.reverse()
    return path,dict(reason='ok',components=count-1,endpoints=len(endpoints),branches=sum(len(v)>2 for v in adj.values()),path_nodes=len(path),component_nodes=len(nodes))
