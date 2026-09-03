# Embodied Gaussians 手术软组织仿真扩展

本仓库在 Physically Embodied Gaussian Splatting 基础上，面向手术软组织牵拉任务实现了可变形 PBD 仿真、视觉轨迹校正、在线材料参数更新和完整评估流程。当前工作目录为 `embodied_gaussians_fixed_super_best_sim`，不修改原始 `embodied_gaussians_fixed_super_best`。

## 当前方法

- 四面体 PBD 组织：distance、volume、shape-matching 约束；
- 表面 Gaussian 与物理三角面绑定，随组织形变更新位置和尺度；
- CoTracker 从 RGB 提供二维材料点轨迹；
- FoundationStereo 仅由双目 RGB 估计深度并反投影三维轨迹；
- 已知夹持区域轨迹作为三种方法统一的运动边界；
- 在线全局 distance stiffness 与全局阻尼辨识，夹持耦合固定为 1；
- 7:1 因果重建和前 80% 更新、后 20% 开环未来预测；
- 评估 30 个固定非夹持点的 3D/2D Tracking Error，以及 PSNR、SSIM、LPIPS。

正式对比包含：

- A：纯 PBD；
- B：PBD + CoTracker 轨迹校正 + FoundationStereo RGB 深度；
- C：B + 全局 distance stiffness 与全局阻尼在线更新。

## 正式测评结果

SIM-01、SIM-02、SIM-03 的完整结果、逐帧指标、轨迹、材料诊断和刚度更新信号位于：

[三数据集全局刚度完整测评](outputs/sim_three_datasets_global_only_foundation_complete_v3/README.md)

逐帧渲染 PNG 约 900 MB，属于可重新生成的缓存，因此未纳入 Git；PSNR、SSIM、LPIPS 的逐帧 CSV 和最终 JSON 均已保留。

## 环境与依赖

项目使用两个独立 Conda 环境：

- `eg_codex`：运行 Embodied Gaussians、PBD、重建与测评；
- `eg_sim`：生成或检查 SuFIA/Orbit Surgical 仿真资产。

FoundationStereo 以子模块固定到实验使用的提交。克隆时使用：

```bash
git clone --recurse-submodules https://github.com/wad157/Embodied_gaussians_fixed_super_best_sim.git
```

已有克隆可运行：

```bash
git submodule update --init --recursive
```

原始数据集、RGB、深度图和模型权重体积较大，不随 Git 仓库分发，分别放在 `data/` 与 `weights/`。实验所需目录及生成流程见 [PROGRESS.md](PROGRESS.md)。

## ThinLinc GUI

在 ThinLinc 桌面中启动仿真 GUI：

```bash
bash scripts/run_sim_reconstruction_thinlinc.sh
```

无界面的正式三数据集全局刚度测评：

```bash
bash scripts/run_sim_three_datasets_global_only_complete.sh \
  outputs/sim_three_datasets_global_only_foundation_complete_v3
```

脚本拒绝覆盖已有输出，请为复测指定新的输出目录。

## 上游项目

以下为原始 Physically Embodied Gaussian Splatting 项目信息。

# Physically Embodied Gaussian Splatting

<div align="left" style="display: left; align-items: center; justify-content: center; gap: 20px;">
    <img src="static/logo.jpeg" alt="Embodied Gaussians Logo" width="400px">
</div>

[Project Page](https://embodied-gaussians.github.io/) | [Paper](https://openreview.net/pdf?id=AEq0onGrN2)

## Overview

Embodied Gaussians introduces a novel dual "Gaussian-Particle" representation that bridges the gap between physical simulation and visual perception for robotics. Our approach:

- 🎯 **Unifies** geometric, physical, and visual world representations
- 🔮 Enables **predictive simulation** of future states
- 🔄 Allows **online correction** from visual observations
- 🌐 Integrates with an **XPBD physics** system
- 🎨 Renders high-quality images through **3D Gaussian splatting**

## Demo
<div align="left" style="display: left; align-items: center; justify-content: center; gap: 20px;">
    <img src="static/embodied_demo.gif" alt="Embodied Gaussians Demo" width="640">
</div>

## Abstract

For robots to robustly understand and interact with the physical world, it is highly beneficial to have a comprehensive representation -- modelling geometry, physics, and visual observations -- that informs perception, planning, and control algorithms. We propose a novel dual "Gaussian-Particle" representation that models the physical world while (i) enabling predictive simulation of future states and (ii) allowing online correction from visual observations in a dynamic world.

Our representation comprises particles that capture the geometrical aspect of objects in the world and can be used alongside a particle-based physics system to anticipate physically plausible future states. Attached to these particles are 3D Gaussians that render images from any viewpoint through a splatting process thus capturing the visual state. By comparing the predicted and observed images, our approach generates "visual forces" that correct the particle positions while respecting known physical constraints.

By integrating predictive physical modeling with continuous visually-derived corrections, our unified representation reasons about the present and future while synchronizing with reality. We validate our approach on 2D and 3D tracking tasks as well as photometric reconstruction quality.

## Implementation Notes

This repository provides a reference implementation of Embodied Gaussians with some differences from the paper:

- 🔷 **Rigid Bodies Only**: Currently, this implementation only supports rigid body dynamics. The shape matching functionality described in the paper is not included.
- 🔨 **Simplified Physics**: Due to the rigid body constraint, the physics simulation is more straightforward but less flexible than the full implementation described in the paper.

## Getting Started

### Installation

1. First, install [pixi](https://pixi.sh/latest/#installation)

2. Then build the dependencies:
```bash
pixi r build
```

### Running the Demo

To run the included demo:
```bash
pixi r demo
```


## Scene Building

These scripts require Realsense cameras directly connected to your device. Note: Offline image processing is not currently supported.

### 1. Ground Plane Detection
First, detect the ground plane by running:
```bash
python scripts/find_ground.py temp/ground_plane.json --extrinsics scripts/example_extrinsics.json --visualize
```
You will be prompted to segment the ground in the interface. The script will then calculate the ground points and plane parameters.

<div align="left">
    <img src="static/ground_detection_example.png" alt="Ground Detection" width="320">
</div>

### 2. Generate Ground Gaussians
Convert the detected ground plane into Gaussian representations:
```bash
python scripts/build_body_from_pointcloud.py temp/ground_body.json --extrinsics scripts/example_extrinsics.json --points scripts/example_ground_plane.npy --visualize
```

### 3. Object Generation
Generate embodied Gaussian representations of objects using multiple viewpoints (more viewpoints yield better results):

1. Run the scene building script:
```bash
python scripts/build_simple_body.py objects/tblock.json \
    --extrinsics scripts/example_extrinsics.json \
    --ground scripts/example_ground_plane.json \
    --visualize
```

2. For each camera viewpoint:
   - A segmentation GUI will appear
   - Click to select the target object
   - Press `Escape` when satisfied with the selection
   - Repeat for all viewpoints

3. The script will generate a JSON file containing both particle and Gaussian representations of your object.

<div align="left">
    <img src="static/scene_builder_segmentation.png" alt="Segmentation Window" width="640">
</div>

You can visualize the object with
```bash
python scripts/visualize_object.py scripts/example_object.json
```


## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@inproceedings{
    abouchakra-embodiedgaussians,
    title={Physically Embodied Gaussian Splatting: A Realtime Correctable World Model for Robotics},
    author={Jad Abou-Chakra and Krishan Rana and Feras Dayoub and Niko Suenderhauf},
    booktitle={8th Annual Conference on Robot Learning},
    year={2024},
    url={https://openreview.net/forum?id=AEq0onGrN2}
}
```

## More Information

For videos and additional information, visit our [project page](https://embodied-gaussians.github.io/).

## Disclaimer
This software is provided as a research prototype and is not production-quality software. Please note that the code may contain missing features, bugs and errors. RAI Institute does not offer maintenance or support for this software.
