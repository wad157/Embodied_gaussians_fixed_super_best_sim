"""Export the frozen offline-calibrated tissue profile for later GUI loading."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'data/super/tissue_calibration_v1'
OUT=BASE/'stage_g_calibrated_profile'

def sha(path: Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    material=BASE/'stage_d_material_calibration/calibrated_material.json'
    residual=BASE/'stage_e_residual_mapping/stage_e_residual_mapping_report.json'
    stiffness=BASE/'stage_f_local_stiffness/stage_f_local_stiffness_report.json'
    region=BASE/'stage_f_local_stiffness/region_profile.npz'
    manifest=BASE/'stage_a_frozen_manifest.json'
    rd=json.loads(residual.read_text()); sd=json.loads(stiffness.read_text()); md=json.loads(material.read_text())
    profile = {
        'schema': 'super_tissue_calibrated_profile_stage_g_v1',
        'status': 'offline_calibrated_ready_for_gui_integration',
        'material': {k: md[k] for k in ('young_modulus_pa','velocity_damping_per_second','poisson_ratio','gravity_m_s2')},
        'regional_stiffness': sd['regional_parameters'],
        'residual_mapping': {
            'enabled_offline': True, 'online_visual_force_default': False,
            'rmse_before_mm': rd['surface_metrics']['physical_surface_rmse_mm_mean_p95_max'][0],
            'rmse_after_mm': rd['surface_metrics']['corrected_surface_rmse_mm_mean_p95_max'][0],
            'rmse_improvement_fraction': rd['surface_metrics']['rmse_improvement_fraction'],
            'p95_improvement_fraction': rd['surface_metrics']['p95_improvement_fraction'],
        },
        'runtime_policy': {'physics_prediction_immutable': True, 'residual_state_separate': True, 'fixed_boundary_preserved': True, 'visual_force_default': 'off', 'online_stiffness_optimization': 'off'},
        'inputs': {p.name: {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)} for p in [manifest, material, residual, stiffness, region]},
        'next_step': 'load this profile in GUI, then run online visual-force-on/off A/B validation',
    }
    (OUT/'tissue_calibrated_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    (OUT/'stage_g_error_summary.json').write_text(json.dumps(profile['residual_mapping'],indent=2)+'\n')
    print(json.dumps(profile,indent=2))
if __name__=='__main__': main()
