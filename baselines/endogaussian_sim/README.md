# EndoGaussian SIM baseline

This adapter evaluates the upstream EndoGaussian implementation without editing its
algorithm sources. The runtime checkout is pinned to
`8d12793838a1595b299df0696c8149c07329e980`. Shape of Motion commit
`579753e1c7ba96f60cd7690e5b835627bd1935e9` supplies only the trajectory-decoding
design: target-time Gaussian 3D features are rasterized through query-time geometry.
No Shape of Motion weights, preprocessing, split, dataset, or evaluator are used.

## Isolation and inputs

As rechecked against both official default branches on 2026-09-11, those hashes are
still their branch heads. The current [EndoGaussian README](https://github.com/CUHK-AIM-Group/EndoGaussian)
specifies Python 3.7, PyTorch 1.13.1 and CUDA 11.7; the dedicated environment is
`/Media_HDD/jwshan/conda_envs/endogaussian_baseline`, beside `eg_codex`. The wrapper
clears compiler, Python, and CUDA variables inherited from an activated `eg_codex`
shell. It uses Python 3.7, Torch 1.13.1 and CUDA 11.7 to compile the extensions from the
pinned checkout with GCC 11.

Training reads only:

- prefix RGB from both calibrated cameras;
- the dataset's current FoundationStereo depth version;
- the current packed tissue masks under `gui_assets/visual_force_masks`;
- fixed camera intrinsics, extrinsics, and timestamps.

It never opens `ground_truth` or the evaluation-point manifest. After the checkpoint
is frozen, the exporter reads the immutable 30 node IDs and their visible frame-0
left-camera query pixels. It back-projects each query pixel with EndoGaussian's own
rendered depth, then adds target-minus-query Gaussian displacement rasterized through
the fixed query-time geometry. This enforces the defining point-tracking invariant
that every prediction starts at its query pixel without reading GT depth or GT 3D.
The existing repository evaluators are the only consumers of future trajectory truth
and rendering masks.

The upstream CUDA rasterizer culls camera depth at `z <= 0.2`. SIM coordinates are in
meters, so the adapter applies a declared input-unit conversion of 1000 internal units
per meter to depth, camera translation, and initialized points. Exported coordinates
are divided by exactly 1000 before evaluation. This constant is not fitted to
ground truth, and no ICP, Procrustes, pose, scale, or time alignment is performed.

## Fixed protocol

- SIM-01/SIM-02: 300 frames; training 210 frames; reconstruction holdout 30 frames;
  future 60 frames.
- SIM-03: 360 frames; training 252 frames; reconstruction holdout 36 frames; future
  72 frames.
- Training uses `t < floor(4T/5) and t % 8 != 7`.
- Time is always normalized as `t / (T - 1)` over the full sequence.
- Three formal repeats use seeds 0, 1 and 2 and are summarized with arithmetic mean
  and sample standard deviation. No best run is selected.

The reconstruction result is an offline EndoGaussian interpolation benchmark. The
last 20% is a prefix-trained, zero-shot temporal-field extrapolation. EndoGaussian
does not consume tool control, so the latter must not be described as a physics
rollout equivalent to the proposed method.

## Commands

Create or repair the environment and pinned checkout:

```bash
bash scripts/setup_endogaussian_baseline_env.sh
```

Audit one dataset without training:

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/audit_endogaussian_sim_protocol.py --dataset-key sim01
```

Run one formal repeat:

```bash
SIM_GPU_ID=0 bash scripts/run_endogaussian_sim_baseline_once.sh \
  sim01 repeat_01 0 outputs/endogaussian_sim01_repeat_01
```

Run and summarize all three datasets and three seeds serially:

```bash
SIM_GPU_ID=0 bash scripts/run_endogaussian_sim_baseline_three_repeats.sh \
  outputs/endogaussian_sim_unified_three_repeats_v1
```

`configs/smoke.py` is an engineering-only two-step configuration. Results produced
with it must never enter a paper table.
