"""Audit Stage-B tissue point clouds for coverage and likely tool leakage."""
from __future__ import annotations
import json, glob
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

def main():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path('data/super/tissue_calibration_v1/stage_b_surface_observations')
    asset=np.load('data/super/grasp5_native/tissue_multiview_v1/soft_tissue_adaptive_v1/tissue_soft_adaptive.npz')
    rest=asset['rest_positions_table']; tree=cKDTree(rest)
    rows=[]
    for p in glob.glob(str(root/'frames/*.npz')):
        d=np.load(p); f=int(d['left_frame']); q=d['points_table']; dist=tree.query(q,k=1,workers=1)[0]*1000
        meta=json.loads(Path(p).with_suffix('.json').read_text())
        rows.append({'frame':f,'count':len(q),'p50_nearest_rest_mm':float(np.quantile(dist,.5)),'p95_nearest_rest_mm':float(np.quantile(dist,.95)),'max_nearest_rest_mm':float(dist.max()),'fraction_nearest_rest_lt3mm':float(np.mean(dist<3)),'right_semantic_retained_fraction':meta['right_semantic_retained_fraction'],'visible_depth_fraction':meta['visible_depth_fraction']})
    rows.sort(key=lambda x:x['frame']); summary={'schema':'super_tissue_pointcloud_segmentation_audit_v1','frame_count':len(rows),'point_count_min_median_max':[min(x['count'] for x in rows),float(np.median([x['count'] for x in rows])),max(x['count'] for x in rows)],'nearest_rest_p95_mm_quantiles':np.quantile([x['p95_nearest_rest_mm'] for x in rows],[0,.5,.95,1]).tolist(),'fraction_nearest_rest_lt3mm_quantiles':np.quantile([x['fraction_nearest_rest_lt3mm'] for x in rows],[0,.5,.95,1]).tolist(),'semantic_retained_fraction_quantiles':np.quantile([x['right_semantic_retained_fraction'] for x in rows],[0,.5,.95,1]).tolist(),'interpretation':'semantic mask removes tool pixels; high nearest-rest fraction supports tissue consistency, but this audit cannot prove no tissue was hidden behind the tool'}
    out=root/'segmentation_audit'; out.mkdir(exist_ok=True); (out/'segmentation_audit.json').write_text(json.dumps({'summary':summary,'frames':rows},indent=2)+'\n')
    for frame in (0,548,906,1439):
        d=np.load(root/f'frames/{frame:06d}.npz'); q=d['points_table']; fig=plt.figure(figsize=(14,5));
        for i,(a,b,title) in enumerate([(0,1,'XY'),(0,2,'XZ'),(1,2,'YZ')]):
            ax=fig.add_subplot(1,3,i+1); ax.scatter(rest[::10,a]*1000,rest[::10,b]*1000,s=1,c='lightgray',label='rest tissue'); ax.scatter(q[:,a]*1000,q[:,b]*1000,s=1,c=q[:,2]*1000,cmap='turbo',label='Stage-B points'); ax.set_xlabel(f'axis {a} (mm)'); ax.set_ylabel(f'axis {b} (mm)'); ax.set_title(f'frame {frame} {title}'); ax.axis('equal')
        fig.tight_layout(); fig.savefig(out/f'pointcloud_frame_{frame:04d}.png',dpi=180); plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
