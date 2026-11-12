"""Explicit MMPCF adaptation candidate: jointly attach endpoints to one component."""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components,dijkstra

def joint_component_path(skel,start,end,silhouette,original_path):
    prior=original_path(skel,start,end,silhouette)
    if prior is not None:return prior,{'fallback_used':False,'rule':'original_success_retained'}
    yy,xx=np.nonzero(skel);coords=np.stack([xx,yy],1);n=len(coords)
    if n<10:return None,{'fallback_used':False,'rule':'original_short_skeleton_retained'}
    index={tuple(p):i for i,p in enumerate(coords)};edges=[];weights=[]
    # Exact same radius-2, silhouette-contained edge definition as the archived adaptation.
    for i,(x,y) in enumerate(coords):
        for dx,dy in [(dx,dy) for dy in range(-2,3) for dx in range(-2,3) if dy>0 or dy==0 and dx>0]:
            j=index.get((x+dx,y+dy))
            if j is not None:
                samples=np.round(np.linspace([x,y],[x+dx,y+dy],5)).astype(int)
                if np.all((samples[:,0]>=0)&(samples[:,1]>=0)&(samples[:,0]<silhouette.shape[1])&(samples[:,1]<silhouette.shape[0])) and np.all(silhouette[samples[:,1],samples[:,0]]):
                    edges.extend([(i,j),(j,i)]);weights.extend([float(np.hypot(dx,dy))]*2)
    if not edges:return None,{'fallback_used':True,'rule':'no_graph_edges'}
    graph=csr_matrix((weights,(np.array(edges)[:,0],np.array(edges)[:,1])),shape=(n,n));count,labels=connected_components(graph,directed=False);choices=[]
    for label in range(count):
        ids=np.flatnonzero(labels==label)
        if len(ids)<8:continue
        q=coords[ids];ia=int(np.argmin(((q-start)**2).sum(1)));ib=int(np.argmin(((q-end)**2).sum(1)));a=int(ids[ia]);b=int(ids[ib])
        if a==b:continue
        dist,prev=dijkstra(graph,directed=True,indices=a,return_predecessors=True)
        if not np.isfinite(dist[b]):continue
        path=[b]
        while path[-1]!=a:
            parent=int(prev[path[-1]])
            if parent<0:break
            path.append(parent)
        if path[-1]!=a or len(path)<8:continue
        path=coords[path[::-1]].astype(float);cost=float(np.linalg.norm(coords[a]-start)+np.linalg.norm(coords[b]-end))
        choices.append(((cost,-len(path),label),path))
    if not choices:return None,{'fallback_used':True,'rule':'no_viable_shared_component','graph_components':int(count)}
    key,path=min(choices,key=lambda item:item[0])
    return path,{'fallback_used':True,'rule':'minimum_joint_endpoint_attachment_distance','graph_components':int(count),'endpoint_attachment_sum_px':key[0],'component_label':key[2]}
