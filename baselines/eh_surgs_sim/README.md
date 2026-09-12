# EH-SurGS SIM baseline

该适配器在不改变 EH-SurGS 核心算法的前提下，将官方实现接入本仓库冻结的 SIM-01/02/03 联合协议。

## 方法与固定版本

- 官方仓库：<https://github.com/IRMVLab/EH-SurGS>
- 固定提交：`73fa04e6f5c21cc1685f728eccb1332e81ce620c`
- Shape of Motion 轨迹解码设计参考：`579753e1c7ba96f60cd7690e5b835627bd1935e9`
- 轨迹解码版本：`som_query_anchored_displacement_v2`
- 内部单位：固定 `1000 units/m`，导出时除以 1000；不从 GT 拟合尺度
- 正式训练：官方 3000 次迭代配置，初始 30,000 个 Gaussians，seed 0/1/2

EH-SurGS 使用 canonical Gaussians 表示外观与几何，再由时间条件形变模型给出位置、尺度和旋转变化。自适应运动层级允许不同空间区域使用不同复杂度的时间基函数。本仓库只增加数据、相机、冻结划分和结果导出接口。

官方训练代码内有面向 640×512 EndoNeRF 的固定 `focal/cx/cy`，它们用于自适应运动分块。本适配保存一份可审计的最小补丁，使该步骤读取当前相机的标定内参。补丁哈希由协议审计器固定；形变网络、损失、优化器和 CUDA rasterizer 不变。

## 冻结协议

- SIM-01/02：300 帧；前 240 帧训练/重建，后 60 帧 Future。
- SIM-03：360 帧；前 288 帧训练/重建，后 72 帧 Future。
- 前 80% 中 `frame % 8 == 7` 为 Reconstruction 留出帧，其余帧才允许训练。
- 左右相机均参与训练；深度只来自当前 FoundationStereo 缓存。
- checkpoint 冻结后才读取 frame 0 的 30 个固定非夹持查询像素。
- 最后 20% 不读取 RGB、深度、mask、轨迹或器械控制。
- 轨迹指标不做尺度、位姿、ICP、刚体或时间对齐。

EH-SurGS 没有原生点轨迹 API。适配器在查询时刻渲染模型自身的 alpha/premultiplied depth 得到查询起点，再把每个 Gaussian 的目标时刻位移作为属性，在固定查询几何上进行光栅化。这个设计只参考 Shape of Motion 的查询几何属性渲染方式，不使用其模型、权重、数据、深度或 evaluator。

## 环境

正式环境位于 `/Media_HDD/jwshan/conda_envs/eh_surgs_baseline`，Python 3.7、PyTorch 1.13.1。两项 CUDA 扩展使用一致的 CUDA 11.8 工具链与 GCC 11 从固定上游源码编译，避免混用上游预编译二进制或 CUDA 11.7/11.8 头文件。

```bash
bash scripts/setup_eh_surgs_baseline_env.sh
```

setup 脚本会克隆固定上游、应用可审计相机补丁、创建独立环境、重新编译 `simple-knn` 与 `depth-diff-gaussian-rasterization`，并在 GPU 上执行扩展 smoke test。

## 运行与审计

单次运行：

```bash
SIM_GPU_ID=0 bash scripts/run_eh_surgs_sim_baseline_once.sh \
  sim03 repeat_01 0 outputs/eh_surgs_sim03_once_v1/repeat_01/sim03
```

三数据集、三 seed 双 GPU 调度：

```bash
bash scripts/run_eh_surgs_sim_three_repeats_two_gpus.sh \
  outputs/eh_surgs_sim_unified_three_repeats_v1 \
  outputs/eh_surgs_sim03_once_v1/repeat_01/sim03
```

协议审计可独立执行：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/audit_eh_surgs_sim_protocol.py \
  --dataset-key sim03 \
  --dataset data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm \
  --output /tmp/eh_surgs_sim03_audit.json
```

审计会校验上游提交与唯一补丁哈希、数据集/深度/mask/相机哈希、帧集合互斥、30 点非夹持清单和 frame-0 投影。单次脚本在训练前 fail closed；汇总器再次逐 run 校验 seed、轨迹解码版本、完整帧表与 30/30 coverage。

## 正式结果

三套数据均完成 seed 0/1/2 三次实际训练，报告算术均值 ± 样本标准差，不挑选最优结果。结果位于：

- `outputs/eh_surgs_sim_unified_three_repeats_v1/comparison_mean_std.md`
- `outputs/eh_surgs_sim_unified_three_repeats_v1/comparison_mean_std.json`
- `outputs/eh_surgs_sim_unified_three_repeats_v1/eh_surgs_sim_mean_std.png`

Git 只发布聚合结果和图，不包含原始数据、checkpoint、逐帧预测或视频。
