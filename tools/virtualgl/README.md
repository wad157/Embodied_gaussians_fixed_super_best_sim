# ThinLinc NVIDIA OpenGL runtime

This directory contains the host-local VirtualGL setup used by
`scripts/run_demo_thinlinc.sh`. It does not replace the system NVIDIA driver.
The launcher now detects the loaded NVIDIA kernel-module version and selects
matching system GLVND libraries first. The repository bundle is only a fallback
when its exact version matches the loaded driver.

## Runtime versions

- VirtualGL: `3.1.4`
- Preferred NVIDIA OpenGL/EGL userspace libraries: matching host system GLVND
- Repository fallback NVIDIA runtime: `535.104.12`
- Default VirtualGL device: `egl0`

The NVIDIA userspace and kernel-module versions must match exactly. The launch
script checks this before starting the GUI. It no longer assumes that the host
driver remains at `535.104.12` after an administrator upgrades it.

Check version and library selection without requiring a ThinLinc display:

```bash
bash scripts/run_demo_thinlinc.sh --check-nvidia-runtime
```

The output should report one identical version for the loaded kernel module and
the selected OpenGL runtime. On the current host after the 2026-08-15 upgrade,
this is `580.159.04` from `/usr/lib/x86_64-linux-gnu`.

## Verification

Run this inside a ThinLinc desktop terminal:

```bash
bash scripts/run_demo_thinlinc.sh --check-virtualgl
```

The renderer must contain `NVIDIA A800` and must not contain `llvmpipe`.

ThinLinc display numbers are session assignments, not project constants. A
previous desktop may have used `:14` while a new session uses `:10`. When
`DISPLAY` is already set, the launcher preserves it; otherwise it probes the
current user's live ThinLinc X socket. Do not hard-code an old display number.

If the runtime check passes but `nvidia-smi` or the VirtualGL check cannot access
the GPU, inspect `nvidia-smi` and `/dev/nvidia*`. Missing device nodes or a failed
`nvidia-smi` require a host driver/module fix; changing the OpenGL search path is
not sufficient.

## Demo

```bash
bash scripts/run_demo_thinlinc.sh --psm-pose-driver depth_then_visual
```

Use the second A800 with:

```bash
VGL_DEVICE=egl1 bash scripts/run_demo_thinlinc.sh \
  --psm-pose-driver depth_then_visual
```

Software OpenGL remains available only for diagnosis:

```bash
USE_VIRTUALGL=0 bash scripts/run_demo_thinlinc.sh \
  --psm-pose-driver depth_then_visual
```

## Local files

`cache/` stores the official VirtualGL package and NVIDIA installer. `vendor/`
stores their extracted runtime. Both directories are host-specific and are
intentionally ignored by Git.
