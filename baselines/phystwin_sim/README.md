# PhysTwin SIM baseline adapter

This adapter evaluates the official [PhysTwin](https://github.com/jianghanxiao/phystwin)
implementation at commit `81c718790a37e5e0102eb77af2c6edd34a9db25f` under the
same fixed SIM protocol as the existing baselines.

## Trajectory choice

The trajectory is generated directly from PhysTwin's persistent spring-mass
particles. Gaussian and fixed query-point motion is propagated with PhysTwin's
upstream K-nearest-neighbour LBS (`gaussian_splatting.dynamic_utils`, K=16).
Shape of Motion is not trained, imported, or used. This preserves PhysTwin's
native physical state and avoids replacing the baseline's motion model with a
separate learned trajectory field.

## Fixed protocol

- SIM01/SIM02: frames `[0, 240)` are the prefix and `[240, 300)` is future.
- SIM03: frames `[0, 288)` are the prefix and `[288, 360)` is future.
- Prefix frames with `t % 8 == 7` are reconstruction holdouts. Only the other
  prefix RGB, public tissue masks, and cached FoundationStereo depth are opened
  during preprocessing.
- The articulated PSM link poses are the protocol-provided known control input.
- Future images, future depth/masks, evaluation node IDs, and simulation truth
  are not opened until physics and appearance are frozen.
- Evaluation uses the immutable 30 non-grasp nodes, exact IDs, and no spatial,
  scale, or temporal alignment.
- Rendering uses both cameras and the existing tissue-layer union-crop
  PSNR/SSIM/LPIPS implementation.

Formal physics uses the official `configs/real.yaml` values: 30 FPS,
`dt=5e-5`, 667 substeps per video frame, 20 CMA-ES generations, and 200 Adam
epochs. Static query-frame appearance uses 50,000 Gaussians and 1,000 updates.

The official simulator and optimization core are unchanged. The only upstream
patch lazily imports interactive X11 visualization code so the original core can
run on a headless node; its required diff hash is audited.

## Run

Environment setup (clones the existing `eg_codex` environment by default):

```bash
bash scripts/setup_phystwin_baseline_env.sh
```

One formal run:

```bash
SIM_GPU_ID=0 bash scripts/run_phystwin_sim_baseline_once.sh \
  sim01 repeat_01 0 outputs/phystwin_sim_baseline/runs/repeat_01/sim01
```

The run is fail-closed and refuses to overwrite output. It writes a protocol
audit, preprocessing provenance, formal physics and appearance checkpoints,
native particle rollout, rendered evaluation frames, both metric JSON/CSV
pairs, a completion status, and SHA-256 manifest.

After all datasets and seeds 0/1/2 are complete:

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/summarize_phystwin_sim_baseline.py \
  --root outputs/phystwin_sim_baseline
```
