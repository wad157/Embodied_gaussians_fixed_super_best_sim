"""Build a deterministic, spatially smooth 3-region stiffness profile."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', type=Path, default=Path('data/super/tissue_calibration_v1/stage_f_local_stiffness'))
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path('data/super/grasp5_native/tissue_multiview_v1/soft_tissue_adaptive_v1')
    asset = np.load(root/'tissue_soft_adaptive.npz')
    d = np.load('data/super/tissue_calibration_v1/stage_e_residual_mapping/top_residual_rms_by_split.npz')
    top_ids = d['top_particle_ids'].astype(np.int64)
    xyz = asset['rest_positions_table'].astype(np.float64)
    score = np.exp(np.mean(np.log(np.maximum(np.stack([d['calibration'],d['validation'],d['test']],1),1e-9)),1))
    corr = np.corrcoef(np.stack([d['calibration'],d['validation'],d['test']]))
    # Persistent high-residual locus; center is weighted by the geometric score.
    cut = float(np.quantile(score, .75)); high = score >= cut
    center = np.average(xyz[top_ids], axis=0, weights=score)
    xy = xyz[:, :2]; dist = np.linalg.norm(xy - center[:2], axis=1)
    hd = np.quantile(dist[top_ids[high]], .75) if np.any(high) else np.quantile(dist[top_ids], .5)
    hd = max(float(hd), 1e-4)
    primary = np.exp(-0.5*(dist/(0.70*hd))**2)
    transition = np.exp(-0.5*(dist/(1.45*hd))**2) - primary
    primary = np.clip(primary,0,1); transition = np.clip(transition,0,1)
    primary[asset['fixed_mask']] = 0; transition[asset['fixed_mask']] = 0
    far = np.clip(1-primary-transition,0,1); far[asset['fixed_mask']] = 0
    w = np.stack([primary, transition, far],1); w /= np.maximum(w.sum(1,keepdims=True),1e-9)
    # Extend top-region weights smoothly through tetrahedra by vertex averaging.
    tw = w[np.asarray(asset['tet_indices'],dtype=np.int64)].mean(1)
    np.savez_compressed(args.output_dir/'region_profile.npz', particle_region_weights=w.astype(np.float32), tet_region_weights=tw.astype(np.float32), top_ids=top_ids, top_score=score.astype(np.float32), center=center.astype(np.float32), cutoff=np.float32(cut), spatial_scale=np.float32(hd), correlations=corr.astype(np.float32))
    report={'schema':'super_tissue_stage_f_region_profile_v1','passed':bool(np.all(np.isfinite(w)) and np.allclose(w.sum(1),1,atol=1e-5)), 'top_particle_count':int(len(top_ids)), 'persistent_high_residual_count':int(high.sum()), 'score_cutoff_m':cut, 'center_m':center.tolist(), 'spatial_scale_m':hd, 'correlations':corr.tolist(), 'particle_region_fraction':w.mean(0).tolist(), 'tet_region_fraction':tw.mean(0).tolist(), 'outputs':{'region_profile':'region_profile.npz'}}
    (args.output_dir/'stage_f_region_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
