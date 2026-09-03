from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def main():
    root=Path('data/super/tissue_calibration_v1/stage_f_local_stiffness')
    p=np.load(root/'region_profile.npz'); r=json.loads((root/'stage_f_local_stiffness_report.json').read_text())
    w=p['particle_region_weights']; tw=p['tet_region_weights'];
    gates={'report_passed':bool(r['passed']),'finite_particle_weights':bool(np.isfinite(w).all()),'partition_of_unity':bool(np.allclose(w.sum(1),1,atol=1e-5)),'finite_tet_weights':bool(np.isfinite(tw).all()),'all_candidates_present':bool(r['candidate_count']==5),'persistent_correlations':bool(min(np.asarray(p['correlations'])[np.triu_indices(3,1)])>=0.8)}
    out={'schema':'super_tissue_stage_f_local_stiffness_artifact_validation_v1','passed':all(gates.values()),'gates':gates,'selected_candidate':r['selected_candidate'],'regional_parameters':r['regional_parameters']}
    (root/'stage_f_artifact_validation.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2));
    raise SystemExit(0 if out['passed'] else 1)
if __name__=='__main__': main()
