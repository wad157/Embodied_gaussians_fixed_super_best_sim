"""Stage-F regional stiffness search using the frozen Stage-D replay."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, torch, warp as wp
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples'),str(ROOT/'scripts')]
from calibrate_super_tissue_material_stage_d import build_worker_context, evaluate_candidate, sha256_file

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--output-dir',type=Path,default=ROOT/'data/super/tissue_calibration_v1/stage_f_local_stiffness'); args=ap.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    wp.config.kernel_cache_dir='/tmp/warp-super-tissue-stage-f'; wp.init(); wp.set_device(args.device); torch.cuda.set_device(int(args.device.split(':')[-1]))
    profile=np.load(args.output_dir/'region_profile.npz'); tw=profile['tet_region_weights'].astype(np.float64)
    context=build_worker_context(manifest_path=ROOT/'data/super/tissue_calibration_v1/stage_a_frozen_manifest.json',surface_root=ROOT/'data/super/tissue_calibration_v1/stage_b_surface_observations',frame_end=906,device=args.device)
    reports=[]; started=time.perf_counter()
    # Coarse regional sweep; far field is fixed at calibrated 100 Pa.
    candidates=[(0.5,0.75),(1.0,1.0),(2.0,1.25),(1.5,0.75),(0.75,1.25)]
    for i,(m1,m2) in enumerate(candidates):
        E=100.0*(tw@np.array([m1,m2,1.0])); cand={'candidate_id':f'regional_{i:02d}','young_modulus_pa':100.0,'velocity_damping_per_second':5.0,'frame_end':906,'dense_stability':False}
        print(f'[stage-F] {cand["candidate_id"]} primary={m1:g} transition={m2:g}',flush=True)
        rep=evaluate_candidate(candidate=cand,tet_young_modulus_pa=E,manifest_path=ROOT/'data/super/tissue_calibration_v1/stage_a_frozen_manifest.json',surface_report_path=ROOT/'data/super/tissue_calibration_v1/stage_b_surface_observations/stage_b_report.json',**context)
        rep['regional_parameters']={'primary_multiplier':m1,'transition_multiplier':m2,'far_multiplier':1.0,'tet_modulus_min_pa':float(E.min()),'tet_modulus_max_pa':float(E.max()),'tet_modulus_mean_pa':float(E.mean())}
        (args.output_dir/f'{cand["candidate_id"]}.json').write_text(json.dumps(rep,indent=2,default=float)+'\n'); reports.append(rep)
    best=min(reports,key=lambda r:r['objective']['calibration_objective_mm']); b=best['regional_parameters']
    out={'schema':'super_tissue_stage_f_local_stiffness_report_v1','passed':bool(best.get('passed',False)),'method':'persistent residual regions from Stage-E calibration/validation/test geometric-mean field; smooth tetrahedral weights; frozen global damping and Poisson ratio','selected_candidate':best['candidate_id'],'regional_parameters':b,'candidate_objectives_mm':{r['candidate_id']:r['objective']['calibration_objective_mm'] for r in reports},'stage_e_source':str(ROOT/'data/super/tissue_calibration_v1/stage_e_residual_mapping/stage_e_residual_mapping_report.json'),'wall_seconds':time.perf_counter()-started,'next_stage':'full-sequence verification with selected regional profile'}
    (args.output_dir/'stage_f_local_stiffness_report.json').write_text(json.dumps(out,indent=2,default=float)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
