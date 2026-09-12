# Embodied Gaussians：手术软组织重建与在线物理参数更新

本项目在 [Physically Embodied Gaussian Splatting](https://embodied-gaussians.github.io/) 的统一视觉—物理表示上，扩展了面向手术软组织牵拉的完整实验链路：使用 SuFIA/Orbit Surgical 的官方 PSM 器械资产生成专属双目仿真数据，将 Gaussian 外观绑定到可变形 PBD 表面，通过 RGB 跟踪和 RGB 估计深度在线修正物理状态，并因果更新组织的全局 distance stiffness 与阻尼，最后同时评估当前状态重建和未来形变预测。

本仓库是独立副本 `embodied_gaussians_fixed_super_best_sim`，不会修改原始的 `embodied_gaussians_fixed_super_best`。

> 最新正式结论来自 AllTracker + FoundationStereo 的同一 rollout 联合协议三次复测：RGB 轨迹校正是误差下降的主要来源；在线刚度更新在 SIM-03 抬升任务中将未来 3D/2D 误差再降低约 11.8%/13.4%，但在 SIM-01/02 平面牵拉的未来预测中略有退化。完整均值、样本标准差及原始结果见[最新统一协议结果](#76-最新-alltracker-统一协议三次复测)。

[EH-SurGS baseline 三次结果](outputs/eh_surgs_sim_unified_three_repeats_v1/comparison_mean_std.md) · [EH-SurGS 适配与复现说明](baselines/eh_surgs_sim/README.md)

## 1. 项目要解决的问题

给定手术器械牵拉软组织的双目 RGB 序列，项目希望建立一个能够同时回答以下问题的模型：

1. 当前组织表面在哪里，渲染结果是否与观测一致；
2. 在只看到过去和当前帧时，如何用视觉观测纠正 PBD 仿真漂移；
3. 如何根据已经发生的形变在线调整材料参数；
4. 停止视觉校正后，学习到的物理参数能否预测未来 20% 的组织运动。

正式实验不把仿真器的 GT 深度送给算法。GT 轨迹和 GT RGB 只用于最终测评；算法侧深度由 FoundationStereo 从双目 RGB 独立估计。

## 2. 整体流程

```mermaid
flowchart LR
    RGB[双目 RGB] --> TRACK[CoTracker / AllTracker 2D 材料点轨迹]
    RGB --> DEPTH[FoundationStereo 深度]
    TRACK --> OBS[固定材料绑定的 3D 观测]
    DEPTH --> OBS
    GRIP[已知夹持区域轨迹] --> BC[三组方法共用边界条件]
    BC --> PBD[四面体 PBD 预测]
    PBD --> CORR[鲁棒轨迹状态校正]
    OBS --> CORR
    CORR --> GS[表面三角面绑定 Gaussian]
    GS --> RENDER[Gaussian 渲染]
    CORR --> MHE[因果 H1/H2/H3 刚度目标]
    MHE --> PARAM[全局 distance stiffness 与阻尼]
    PARAM --> PBD
```

系统使用三层相互对应的表示：

- **物理层**：四面体粒子和 distance、volume、shape-matching PBD 约束，负责传播夹持点的运动并保持组织连续性；
- **表面层**：物理粒子的外表面三角面，连接跟踪点、物理节点和渲染属性；
- **视觉层**：绑定在表面三角面上的 3D Gaussian，位置随三角形顶点移动，已有尺度形变逻辑保持不变。

对绑定在三角面顶点 \(i,j,k\) 上的 Gaussian，其中心由固定重心坐标决定：

$$
\boldsymbol{\mu}_g^t = b_i\mathbf{x}_i^t+b_j\mathbf{x}_j^t+b_k\mathbf{x}_k^t,
\qquad b_i+b_j+b_k=1.
$$

因此 Gaussian 不会脱离材料表面；三角面的局部形变同时驱动其尺度变化。夹持区域使用数据集中的已知轨迹作为统一边界条件，并从视觉残差与最终测评点中排除，避免把已知运动当作算法预测能力。

## 3. RGB 轨迹状态校正

### 3.1 从 RGB 得到固定材料点观测

CoTracker 或 AllTracker 在 RGB 上给出二维轨迹 \((u_m^t,v_m^t)\)，FoundationStereo 给出对应深度 \(D_m^t\)。相机内外参将其反投影为世界坐标：

$$
\mathbf{y}_m^t = \mathbf{T}_{wc}
\left[D_m^t\mathbf{K}^{-1}(u_m^t,v_m^t,1)^\mathsf{T}\right].
$$

每个轨迹只在初始化时绑定到一个物理节点邻域或表面三角面，之后始终跟踪同一材料位置，不会逐帧滑到新的三角面。若其固定插值权重为 \(w_{mi}\)，当前仿真位置为

$$
\hat{\mathbf{y}}_m^t=\sum_iw_{mi}\mathbf{x}_i^t,
\qquad
\mathbf{r}_m^t=\mathbf{y}_m^t-\hat{\mathbf{y}}_m^t.
$$

### 3.2 鲁棒稀疏校正

所有可用轨迹共同求解粒子增量，而不是把单个点直接拖到观测位置：

$$
\Delta\mathbf{x}^{*}=
\arg\min_{\Delta\mathbf{x}}
\sum_m c_m\,\rho\!\left(\|\mathbf{A}_m\Delta\mathbf{x}-\mathbf{r}_m\|_2\right)
+\lambda\|\Delta\mathbf{x}\|_2^2.
$$

其中 \(c_m\) 综合跟踪置信度、双目一致性和可见性，\(\rho\) 为鲁棒损失，\(\mathbf{A}_m\) 是固定材料绑定的插值矩阵。正式配置使用：位置增益 0.70、绝对位置权重 0.85、24 次求解迭代、5 mm 鲁棒阈值、单帧位置校正上限 2 mm。速度由相邻校正状态因果更新，增益 0.20、上限 0.06 m/s。

这一步在后文称为“轨迹校正”。GUI 仍保留原项目的像素级 visual residual 模式；正式 A/B/C 测评采用更可复现的 `trajectory` 模式，即 RGB 跟踪加 FoundationStereo 深度的材料点校正。

## 4. 因果刚度与阻尼更新

当前正式 C 组采用**一个全局 distance stiffness 和一个全局阻尼**，不是六区域、十二区域或逐粒子刚度。代码中的十二个空间组只用于构造抗离群的损失统计，绝不代表十二个材料参数。

参数使用对数形式以保证非负：

$$
k_d=k_{d,0}\exp(\theta_k),
\qquad
\gamma=\gamma_0\exp(\theta_\gamma),
\qquad
\boldsymbol\theta=(\theta_k,\theta_\gamma).
$$

正式配置的初值为 \(k_{d,0}=0.20\)、\(\gamma_0=10\ \mathrm{s}^{-1}\)；shape stiffness 固定为 0.004，volume stiffness 固定为 \(10^{10}\)，夹持耦合固定为 1。

每个时间点只使用最近三个已经发生的转移构造 H1/H2/H3 目标。H1、H2、H3 表示一个、两个、三个时间步的因果预测跨度，不是三个数据集，也不是三个空间区域。轨迹项采用分布鲁棒汇总：

$$
L_{\mathrm{track}}=
0.85L_{\mathrm{mean}}
+0.10L_{\mathrm{group}}
+0.05L_{\mathrm{tail}},
$$

其中 \(L_{\mathrm{group}}\) 是十二个固定空间组的等权均值，\(L_{\mathrm{tail}}\) 强调误差最大的四分之一空间组。总目标为

$$
L(\boldsymbol\theta)=
\sum_{h=1}^{3}\bar\omega_h
\left[0.8L_{\mathrm{track}}^{(h)}+0.2L_{\mathrm{strain}}^{(h)}\right]
+0.4\|\boldsymbol\theta\|_2^2,
\qquad
(\omega_1,\omega_2,\omega_3)=(1.5,2,4).
$$

`distance + volume + shape` 的完整 PBD 时间步参与展开。系统同时计算 Torch 自动微分梯度 \(g_{AD}\) 和 Warp 有限差分梯度 \(g_W\)，只有二者方向足够一致时才允许提交：

$$
\cos(g_{AD},g_W)=
\frac{g_{AD}^{\mathsf T}g_W}{\|g_{AD}\|\,\|g_W\|}\ge 0.95.
$$

通过门控后，由 Warp 梯度驱动 Adam 更新；对数学习率为 0.03、梯度 EMA 为 0.90、单次对数参数变化上限为 0.02。该过程没有离线分支选择，也不会读取未来帧。进入未来 20% 开环阶段后，视觉状态校正和参数提交同时冻结，只保持已经学习到的参数与已知夹持边界向前仿真。

## 5. 数据集

三套数据均为 30 FPS 双目序列，使用 SuFIA/Orbit Surgical 官方 PSM 器械视觉资产；组织、相机、夹持动作和轨迹是为本项目重新构建并录制的专属任务。原始数据体积较大，不随 Git 分发，默认放在 `data/sim/`。

| 编号 | 本地目录 | 动作 | 帧数 | 优化轨迹 | 最终测评点 |
|---|---|---|---:|---:|---:|
| SIM-01 | `tissue_retraction_free_support_front_v2` | 平面 X 方向牵拉 | 300 | 188 | 30 |
| SIM-02 | `tissue_retraction_free_support_side_v2` | 平面 Y 方向牵拉 | 300 | 175 | 30 |
| SIM-03 | `tissue_long_edge_lift_return_sufia_v2_lift30mm` | 从长边侧面夹持、抬升并放回 | 360 | 203 | 30 |

优化轨迹与评估点严格分离。每个数据集最终只用同一份固定的 30 个非夹持点清单报告 3D/2D Tracking Error；优化时不会看到这 30 个点的 GT。三套数据分别生成深度、CoTracker 轨迹、夹持边界和评估清单，不会混用另一数据集的缓存。

## 6. 测评协议

### 6.1 三种方法

| 方法 | PBD | RGB 轨迹校正 | 在线刚度/阻尼更新 |
|---|:---:|:---:|:---:|
| A：纯 PBD | ✓ | — | — |
| B：PBD + 轨迹校正 | ✓ | ✓ | — |
| C：B + 在线物理参数更新 | ✓ | ✓ | ✓ |

三组方法使用完全相同的初始状态、PBD 参数、相机、帧率和已知夹持区域轨迹。B/C 的状态校正参数相同，C 只额外开启因果材料参数更新。

### 6.2 两种能力

- **前 80% 内 7:1 重建**：参考 EH-SurGS 的抽帧思想，在前 80% 时间段内每七帧观测后留一帧检查当前状态重建；更新和评分严格按时间顺序进行。
- **后 20% 未来预测**：前 80% 序列允许视觉状态校正和在线参数更新；最后 20% 完全关闭二者，以前 80% 结束时的状态和参数进行开环预测。

最新协议中，每个“数据集 × 方法 × 重复次数”只运行一个连续物理 rollout，同时导出前 80% 的留出重建帧和后 20% 的开环预测帧。每次重复共运行 3 种方法 × 3 个数据集，即 9 个 rollout，并产生 18 行能力指标；三次复测共 27 个 rollout、54 行指标。实验不把前 80% 重建误差和后 20% 开环误差混成一个指标，也不再为两种能力分别初始化和运行物理系统。

### 6.3 指标

- **3D Tracking Error (mm) ↓**：30 个固定非夹持材料点的世界坐标平均误差；
- **2D Tracking Error (px) ↓**：上述 3D 点投影到有效双目图像后的平均像素误差；
- **PSNR ↑、SSIM ↑、LPIPS ↓**：预测状态经 Gaussian 渲染后与 GT RGB 比较。

### 6.4 重复实验与统计

每个数据集均使用自身的 RGB、FoundationStereo 深度、对应跟踪器轨迹、夹持边界和 30 个固定非夹持评估点，数据集之间不共享缓存。CoTracker 与 AllTracker 分别统计，三次运行不做最优结果挑选，也不使用离线参数分支；表中报告算术均值与样本标准差：

$$
\bar{x}=\frac{1}{3}\sum_{i=1}^{3}x_i,
\qquad
s=\sqrt{\frac{1}{2}\sum_{i=1}^{3}(x_i-\bar{x})^2}.
$$

## 7. 正式测评结果

结果来自 CoTracker 三次独立复测。所有数值均为“均值 ± 样本标准差”；3D、2D、LPIPS 越低越好，PSNR、SSIM 越高越好。

### 7.1 7:1 当前状态重建

| 数据集 | 方法 | n | 3D mm ↓ | 2D px ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| SIM-01 | A：纯 PBD | 3 | 1.558 ± 0.000 | 15.543 ± 0.000 | 19.056 ± 0.000 | 0.8381 ± 0.0000 | 0.4217 ± 0.0000 |
| SIM-01 | B：轨迹校正 | 3 | **0.498 ± 0.000** | **4.852 ± 0.000** | **19.970 ± 0.000** | 0.8469 ± 0.0000 | **0.4076 ± 0.0000** |
| SIM-01 | C：轨迹校正 + 刚度更新 | 3 | 0.504 ± 0.008 | 4.951 ± 0.102 | **19.970 ± 0.005** | **0.8470 ± 0.0000** | 0.4077 ± 0.0001 |
| SIM-02 | A：纯 PBD | 3 | 1.433 ± 0.000 | 13.424 ± 0.000 | 16.425 ± 0.000 | 0.8171 ± 0.0000 | 0.4295 ± 0.0000 |
| SIM-02 | B：轨迹校正 | 3 | 0.378 ± 0.000 | **3.700 ± 0.000** | 17.001 ± 0.000 | **0.8267 ± 0.0000** | **0.4176 ± 0.0000** |
| SIM-02 | C：轨迹校正 + 刚度更新 | 3 | **0.377 ± 0.001** | 3.723 ± 0.011 | **17.011 ± 0.005** | 0.8265 ± 0.0001 | 0.4177 ± 0.0001 |
| SIM-03 | A：纯 PBD | 3 | 2.260 ± 0.000 | 19.890 ± 0.000 | 19.079 ± 0.000 | 0.7767 ± 0.0000 | 0.5488 ± 0.0000 |
| SIM-03 | B：轨迹校正 | 3 | 0.478 ± 0.000 | 4.628 ± 0.000 | **21.895 ± 0.000** | 0.7933 ± 0.0000 | 0.5370 ± 0.0000 |
| SIM-03 | C：轨迹校正 + 刚度更新 | 3 | **0.446 ± 0.001** | **4.296 ± 0.048** | 21.877 ± 0.048 | **0.7938 ± 0.0003** | **0.5344 ± 0.0005** |

### 7.2 后 20% 开环未来预测

| 数据集 | 方法 | n | 3D mm ↓ | 2D px ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| SIM-01 | A：纯 PBD | 3 | 4.281 ± 0.000 | 42.833 ± 0.000 | 16.456 ± 0.000 | 0.8160 ± 0.0000 | 0.4512 ± 0.0000 |
| SIM-01 | B：轨迹校正 | 3 | 1.650 ± 0.000 | 16.091 ± 0.000 | 18.412 ± 0.000 | 0.8349 ± 0.0000 | 0.4210 ± 0.0000 |
| SIM-01 | C：轨迹校正 + 刚度更新 | 3 | **1.543 ± 0.055** | **15.104 ± 0.593** | **18.608 ± 0.076** | **0.8369 ± 0.0006** | **0.4192 ± 0.0010** |
| SIM-02 | A：纯 PBD | 3 | 4.115 ± 0.000 | 37.595 ± 0.000 | 15.476 ± 0.000 | 0.7996 ± 0.0000 | 0.4537 ± 0.0000 |
| SIM-02 | B：轨迹校正 | 3 | 1.334 ± 0.000 | 13.726 ± 0.000 | 16.569 ± 0.000 | 0.8212 ± 0.0000 | **0.4266 ± 0.0000** |
| SIM-02 | C：轨迹校正 + 刚度更新 | 3 | **1.306 ± 0.021** | **13.284 ± 0.265** | **16.664 ± 0.011** | **0.8215 ± 0.0001** | 0.4271 ± 0.0008 |
| SIM-03 | A：纯 PBD | 3 | 5.956 ± 0.000 | 53.957 ± 0.000 | 14.343 ± 0.000 | 0.7462 ± 0.0000 | 0.5653 ± 0.0000 |
| SIM-03 | B：轨迹校正 | 3 | 1.013 ± 0.000 | 9.525 ± 0.000 | 20.878 ± 0.000 | **0.7972 ± 0.0000** | **0.5212 ± 0.0000** |
| SIM-03 | C：轨迹校正 + 刚度更新 | 3 | **0.834 ± 0.119** | **6.436 ± 0.432** | **21.923 ± 0.672** | 0.7964 ± 0.0029 | 0.5297 ± 0.0033 |

### 7.3 结果解释与可重复性

- 与 A 相比，B 在三套数据上分别将重建 3D 误差降低约 **68.0% / 73.6% / 78.9%**，将未来 3D 误差降低约 **61.5% / 67.6% / 83.0%**；CoTracker RGB 轨迹校正是总体提升的主要来源。
- 与 B 相比，C 在 SIM-01/02/03 的未来 3D 均值分别再降低 **6.46% / 2.09% / 17.76%**，未来 2D 均值分别再降低 **6.13% / 3.22% / 32.43%**。这表明前 80% 因果更新得到的刚度与阻尼对后 20% 开环动力学有正收益。
- 重建阶段 C 并非处处优于 B：SIM-01 的 3D/2D 略退化，SIM-02 基本持平，SIM-03 的 3D/2D 分别改善约 **6.7% / 7.2%**。逐帧视觉校正会覆盖一部分材料参数收益，因此不能宣称 C 全指标领先。
- A/B 的三次结果完全一致；C 的随机性主要来自 Warp 有限差分与在线提交路径。波动最大的是 SIM-03 未来预测（3D 为 **0.834 ± 0.119 mm**），但三次单次结果均优于固定的 B 基线 1.013 mm。
- 渲染指标并未全部随几何误差同步改善：例如 SIM-03 未来 C 的 PSNR 更高，但 SSIM 和 LPIPS 略差。当前结论应限定为几何预测改善，而不是所有感知指标全面改善。

Git 中保留了完整聚合结果，不上传体积较大的逐帧图像、深度缓存和日志：

- [三次均值、样本标准差及54条原始结果](outputs/sim_three_datasets_medium_h3w4_three_repeats_v1/comparison_mean_std.md)
- [机器可读 JSON](outputs/sim_three_datasets_medium_h3w4_three_repeats_v1/comparison_mean_std.json)

### 7.4 历史 AllTracker 分离协议三次复测

AllTracker 使用与 CoTracker 相同的首帧查询点、FoundationStereo 深度、PBD 参数、夹持边界和 30 个非夹持评估点。SIM-01/02 分别使用 188/175 条有效绑定轨迹；SIM-03 请求 203 条、有效绑定 200 条。第1次来自此前已完成的正式运行，第2、3次在 GPU0 上连续串行执行；三次全部进入统计，不选择最佳结果。

#### 7:1 当前状态重建

| 数据集 | 方法 | n | 3D mm ↓ | 2D px ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| SIM-01 | A：纯 PBD | 3 | 1.558 ± 0.000 | 15.543 ± 0.000 | 19.056 ± 0.000 | 0.8381 ± 0.0000 | 0.4217 ± 0.0000 |
| SIM-01 | B：AllTracker 轨迹校正 | 3 | 0.509 ± 0.000 | 4.995 ± 0.000 | **19.936 ± 0.000** | **0.8463 ± 0.0000** | **0.4079 ± 0.0000** |
| SIM-01 | C：B + 刚度更新 | 3 | **0.504 ± 0.002** | **4.964 ± 0.029** | 19.906 ± 0.007 | 0.8460 ± 0.0001 | 0.4081 ± 0.0001 |
| SIM-02 | A：纯 PBD | 3 | 1.433 ± 0.000 | 13.424 ± 0.000 | 16.425 ± 0.000 | 0.8171 ± 0.0000 | 0.4295 ± 0.0000 |
| SIM-02 | B：AllTracker 轨迹校正 | 3 | 0.377 ± 0.000 | **3.689 ± 0.000** | 16.981 ± 0.000 | 0.8261 ± 0.0000 | **0.4175 ± 0.0000** |
| SIM-02 | C：B + 刚度更新 | 3 | **0.377 ± 0.001** | 3.691 ± 0.008 | **16.999 ± 0.004** | **0.8262 ± 0.0001** | 0.4177 ± 0.0001 |
| SIM-03 | A：纯 PBD | 3 | 2.260 ± 0.000 | 19.890 ± 0.000 | 19.079 ± 0.000 | 0.7767 ± 0.0000 | 0.5488 ± 0.0000 |
| SIM-03 | B：AllTracker 轨迹校正 | 3 | 0.448 ± 0.000 | 4.493 ± 0.000 | 21.727 ± 0.000 | 0.7921 ± 0.0000 | 0.5395 ± 0.0000 |
| SIM-03 | C：B + 刚度更新 | 3 | **0.443 ± 0.004** | **4.359 ± 0.070** | **21.829 ± 0.010** | **0.7933 ± 0.0006** | **0.5366 ± 0.0015** |

#### 后 20% 开环未来预测

| 数据集 | 方法 | n | 3D mm ↓ | 2D px ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| SIM-01 | A：纯 PBD | 3 | 4.281 ± 0.000 | 42.833 ± 0.000 | 16.456 ± 0.000 | 0.8160 ± 0.0000 | 0.4512 ± 0.0000 |
| SIM-01 | B：AllTracker 轨迹校正 | 3 | 1.416 ± 0.005 | 13.696 ± 0.046 | 18.902 ± 0.006 | 0.8386 ± 0.0001 | 0.4193 ± 0.0001 |
| SIM-01 | C：B + 刚度更新 | 3 | **1.325 ± 0.008** | **12.596 ± 0.109** | **19.058 ± 0.048** | **0.8403 ± 0.0003** | **0.4166 ± 0.0004** |
| SIM-02 | A：纯 PBD | 3 | 4.115 ± 0.000 | 37.595 ± 0.000 | 15.476 ± 0.000 | 0.7996 ± 0.0000 | 0.4537 ± 0.0000 |
| SIM-02 | B：AllTracker 轨迹校正 | 3 | 1.082 ± 0.000 | 11.072 ± 0.000 | **17.015 ± 0.000** | **0.8266 ± 0.0000** | **0.4199 ± 0.0000** |
| SIM-02 | C：B + 刚度更新 | 3 | **1.075 ± 0.011** | **10.679 ± 0.035** | 17.002 ± 0.014 | 0.8265 ± 0.0003 | 0.4202 ± 0.0003 |
| SIM-03 | A：纯 PBD | 3 | 5.956 ± 0.000 | 53.957 ± 0.000 | 14.343 ± 0.000 | 0.7462 ± 0.0000 | 0.5653 ± 0.0000 |
| SIM-03 | B：AllTracker 轨迹校正 | 3 | **0.716 ± 0.000** | **6.264 ± 0.000** | **22.172 ± 0.000** | 0.7927 ± 0.0000 | 0.5385 ± 0.0000 |
| SIM-03 | C：B + 刚度更新 | 3 | 1.038 ± 0.260 | 7.844 ± 1.277 | 21.201 ± 1.209 | **0.7972 ± 0.0050** | **0.5256 ± 0.0084** |

### 7.5 AllTracker 结果解释

- SIM-01 未来预测中，C 相对 B 的 3D/2D 误差分别降低 **6.42%/8.03%**，PSNR、SSIM、LPIPS 也均改善。
- SIM-02 未来预测中，C 的 3D/2D 分别降低 **0.70%/3.54%**，但三项渲染指标轻微退化。
- SIM-03 重建中，C 的 3D/2D 分别降低 **1.13%/2.99%**，渲染指标同时改善；但未来预测 3D 从 **0.716 mm** 增至 **1.038 mm**，2D 从 **6.264 px** 增至 **7.844 px**。
- SIM-03 未来 C 的三次 3D 结果为 **1.0091、1.3115、0.7942 mm**，标准差为 **0.260 mm**。因此 AllTracker 在抬升任务上的刚度更新不能称为稳定提升，后续需要检查轨迹误差如何改变刚度提交方向。

完整 AllTracker 聚合结果：

- [三次均值、样本标准差及54条原始结果](outputs/sim_three_datasets_alltracker_medium_h3w4_three_repeats_v3/comparison_mean_std.md)
- [机器可读 JSON](outputs/sim_three_datasets_alltracker_medium_h3w4_three_repeats_v3/comparison_mean_std.json)

### 7.6 最新 AllTracker 统一协议三次复测

本节是当前应引用的正式结果。每种方法每次只执行一个连续 rollout：前 80% 同化阶段内部按 7:1 留出帧评估重建，随后冻结 RGB 状态校正与刚度更新，在后 20% 评估开环未来预测。SIM-01/02 的 A/B/C 均按新协议重新运行三次；SIM-03 的 A 重新运行三次，B/C 将已完成的新协议单次结果与两次新增运行共同取均值。三次全部进入统计，不挑选最佳运行。

#### 前 80% 内 7:1 重建

| 数据集 | 方法 | n | 3D mm ↓ | 2D px ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| SIM-01 | A：纯 PBD | 3 | 0.920 ± 0.000 | 9.293 ± 0.000 | 19.681 ± 0.000 | 0.8435 ± 0.0000 | 0.4146 ± 0.0000 |
| SIM-01 | B：AllTracker 轨迹校正 | 3 | 0.403 ± 0.000 | 4.008 ± 0.000 | **20.060 ± 0.000** | **0.8467 ± 0.0000** | **0.4076 ± 0.0000** |
| SIM-01 | C：B + 刚度更新 | 3 | **0.402 ± 0.000** | **4.005 ± 0.001** | 20.041 ± 0.000 | 0.8465 ± 0.0000 | 0.4078 ± 0.0000 |
| SIM-02 | A：纯 PBD | 3 | 0.804 ± 0.000 | 7.808 ± 0.000 | 16.649 ± 0.000 | 0.8212 ± 0.0000 | 0.4239 ± 0.0000 |
| SIM-02 | B：AllTracker 轨迹校正 | 3 | 0.265 ± 0.000 | 2.718 ± 0.000 | **16.938 ± 0.000** | **0.8257 ± 0.0000** | **0.4179 ± 0.0000** |
| SIM-02 | C：B + 刚度更新 | 3 | **0.264 ± 0.000** | **2.688 ± 0.001** | 16.937 ± 0.001 | 0.8256 ± 0.0000 | 0.4180 ± 0.0000 |
| SIM-03 | A：纯 PBD | 3 | 1.337 ± 0.000 | 11.203 ± 0.000 | 20.266 ± 0.000 | 0.7843 ± 0.0000 | 0.5444 ± 0.0000 |
| SIM-03 | B：AllTracker 轨迹校正 | 3 | 0.449 ± 0.000 | 4.525 ± 0.000 | 21.616 ± 0.000 | 0.7924 ± 0.0000 | 0.5380 ± 0.0000 |
| SIM-03 | C：B + 刚度更新 | 3 | **0.443 ± 0.004** | **4.369 ± 0.035** | **21.724 ± 0.027** | **0.7935 ± 0.0001** | **0.5357 ± 0.0002** |

#### 后 20% 开环未来预测

| 数据集 | 方法 | n | 3D mm ↓ | 2D px ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| SIM-01 | A：纯 PBD | 3 | 4.281 ± 0.000 | 42.833 ± 0.000 | 16.456 ± 0.000 | 0.8160 ± 0.0000 | 0.4512 ± 0.0000 |
| SIM-01 | B：AllTracker 轨迹校正 | 3 | **1.320 ± 0.000** | **12.551 ± 0.000** | **19.201 ± 0.000** | **0.8417 ± 0.0000** | **0.4164 ± 0.0000** |
| SIM-01 | C：B + 刚度更新 | 3 | 1.339 ± 0.008 | 12.638 ± 0.038 | 19.031 ± 0.035 | 0.8404 ± 0.0004 | 0.4165 ± 0.0004 |
| SIM-02 | A：纯 PBD | 3 | 4.115 ± 0.000 | 37.595 ± 0.000 | 15.476 ± 0.000 | 0.7996 ± 0.0000 | 0.4537 ± 0.0000 |
| SIM-02 | B：AllTracker 轨迹校正 | 3 | **1.235 ± 0.000** | **12.230 ± 0.000** | 16.759 ± 0.000 | 0.8232 ± 0.0000 | 0.4241 ± 0.0000 |
| SIM-02 | C：B + 刚度更新 | 3 | 1.254 ± 0.020 | 12.443 ± 0.242 | **16.815 ± 0.032** | **0.8243 ± 0.0006** | **0.4239 ± 0.0012** |
| SIM-03 | A：纯 PBD | 3 | 5.956 ± 0.000 | 53.957 ± 0.000 | 14.343 ± 0.000 | 0.7462 ± 0.0000 | 0.5653 ± 0.0000 |
| SIM-03 | B：AllTracker 轨迹校正 | 3 | 0.874 ± 0.000 | 7.022 ± 0.000 | 21.564 ± 0.000 | 0.7920 ± 0.0000 | 0.5409 ± 0.0000 |
| SIM-03 | C：B + 刚度更新 | 3 | **0.771 ± 0.097** | **6.079 ± 0.293** | **21.756 ± 0.268** | **0.7923 ± 0.0034** | **0.5385 ± 0.0085** |

最新结果表明：B 相对 A 在三套数据上均显著降低轨迹误差；C 在三个数据集的重建轨迹误差上均略优于 B。未来预测中，C 在 SIM-03 的 3D/2D 均值分别改善约 **11.8%/13.4%**，但在 SIM-01/02 分别轻微退化约 **1%–2%**。刚度诊断显示每次 C rollout 实际提交约 33–51 次更新，distance stiffness 累计下降约 25%–35%，因此平面任务增益不足并非更新次数太少，而是当前全局刚度/阻尼梯度对平面形变的泛化不足。

Git 中保存完整聚合表、54条原始结果和机器可读数据：

- [最新统一协议三次结果](outputs/sim_three_datasets_alltracker_unified_protocol_three_repeats_v1/comparison_mean_std.md)
- [最新统一协议机器可读 JSON](outputs/sim_three_datasets_alltracker_unified_protocol_three_repeats_v1/comparison_mean_std.json)

## 8. 运行方法

### 8.1 克隆

FoundationStereo 作为子模块固定到本实验使用的版本：

```bash
git clone --recurse-submodules \
  https://github.com/wad157/Embodied_gaussians_fixed_super_best_sim.git
cd Embodied_gaussians_fixed_super_best_sim
```

已有克隆执行：

```bash
git submodule update --init --recursive
```

本机使用两个独立 Conda 环境：

- `eg_codex`：Embodied Gaussians、PBD、重建和测评；
- `eg_sim`：SuFIA/Orbit Surgical 数据生成与资产检查。

原始 RGB 数据、生成深度和模型权重分别放在 `data/` 与 `weights/`，不随仓库分发。更完整的实验历史见 [PROGRESS.md](PROGRESS.md)。

### 8.2 ThinLinc GUI

在 ThinLinc 图形会话内运行，例如打开 SIM-03：

```bash
SIM_DATASET="$PWD/data/sim/tissue_long_edge_lift_return_sufia_v2_lift30mm" \
  bash scripts/run_sim_reconstruction_thinlinc.sh
```

GUI 可观察 RGB、Gaussian 渲染、PBD 组织、PSM 器械和夹持边界，并可切换纯物理、轨迹校正等调试模式。

### 8.3 三数据集 CoTracker 三次正式复测

```bash
bash scripts/run_sim_three_datasets_medium_h3w4_three_repeats.sh \
  outputs/sim_three_datasets_medium_h3w4_three_repeats_v2
```

脚本对 SIM-01/02/03 各独立运行三次完整 A/B/C 重建与未来预测，并为每套数据独立读取对应的 FoundationStereo 深度和 CoTracker 缓存。汇总脚本自动生成均值、样本标准差和54条原始结果。为保护正式结果，脚本拒绝覆盖已有输出目录；复测时应使用新的目录名。

### 7.3 EH-SurGS baseline

EH-SurGS 在每套数据上使用 seed 0/1/2 独立训练三次。表中为算术均值 ± 样本标准差，不挑选最优运行；2D 指标按左右相机有效样本数加权，30 个非夹持点的 3D coverage 均为 1.0。

| 数据集 | 分区 | 3D mm ↓ | 2D px ↓ | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|
| SIM-01 | Reconstruction 7:1 | 1.908 ± 0.300 | 8.725 ± 0.202 | 16.277 ± 1.088 | 0.4364 ± 0.0309 | 0.3319 ± 0.0440 |
| SIM-01 | Future 80:20 | 5.259 ± 0.146 | 51.047 ± 0.693 | 15.540 ± 1.014 | 0.3836 ± 0.0190 | 0.4165 ± 0.0538 |
| SIM-02 | Reconstruction 7:1 | 2.585 ± 0.735 | 9.209 ± 0.338 | 15.682 ± 0.401 | 0.3908 ± 0.0605 | 0.3427 ± 0.0186 |
| SIM-02 | Future 80:20 | 5.625 ± 0.484 | 50.647 ± 0.128 | 14.722 ± 0.234 | 0.3222 ± 0.0530 | 0.4373 ± 0.0104 |
| SIM-03 | Reconstruction 7:1 | 3.541 ± 0.572 | 13.771 ± 0.290 | 15.678 ± 0.090 | 0.4034 ± 0.0418 | 0.3423 ± 0.0118 |
| SIM-03 | Future 80:20 | 2.775 ± 0.439 | 6.715 ± 0.535 | 15.051 ± 0.141 | 0.3464 ± 0.0481 | 0.3930 ± 0.0159 |

完整逐次数据、协议元数据和聚合方式见[机器可读结果](outputs/eh_surgs_sim_unified_three_repeats_v1/comparison_mean_std.json)。

### 8.4 AllTracker 追加两次并汇总三次

在已有一次完整 AllTracker 测评及其观测资产的前提下，连续运行第2、3次并与第1次汇总：

```bash
SIM_GPU_ID=0 \
SIM_ALLTRACKER_BASELINE_ROOT="$PWD/outputs/sim_three_datasets_alltracker_medium_h3w4_complete_v1" \
  bash scripts/run_sim_three_datasets_alltracker_three_repeats.sh \
  outputs/sim_three_datasets_alltracker_medium_h3w4_three_repeats_v4
```

脚本只复用固定的 AllTracker/FoundationStereo 观测资产，不重复提取轨迹；第2、3次在指定 GPU 上串行执行，完成后自动计算三次均值与样本标准差。

### 8.5 最新同一 rollout 联合协议

```bash
SIM_GPU_ID=0 \
  bash scripts/run_sim_three_datasets_unified_protocol_three_repeats.sh \
  outputs/sim_three_datasets_alltracker_unified_protocol_three_repeats_v1
```

该脚本为 SIM-01/02 的 A/B/C 各运行三次，为 SIM-03 的 A 运行三次，并将 SIM-03 已完成的一次 B/C 与两次新增运行合并。每个方法的一次 rollout 同时生成前 80% 内 7:1 重建与后 20% 开环预测指标，汇总器会严格校验数据集路径、总帧数、80/20 分界、留出帧序列和30点清单。

### 8.6 EndoGaussian baseline

EndoGaussian 使用与 8.5 相同的数据集、前 80% 内 7:1 留出帧、后 20% 时间外推帧和固定 30 个非夹持测评点。训练只读取前缀双目 RGB、当前 FoundationStereo 深度、公共组织 mask 和相机标定；checkpoint 冻结后，按 Shape of Motion 的特征栅格化方式从 frame 0 查询像素解码 EndoGaussian 的 3D Gaussian 轨迹。全流程不使用器械控制，不做位姿、尺度或时间对齐，因此后 20% 结果应标为零样本时间场外推。

```bash
bash scripts/setup_endogaussian_baseline_env.sh
SIM_GPU_ID=0 bash scripts/run_endogaussian_sim_baseline_three_repeats.sh \
  outputs/endogaussian_sim_unified_three_repeats_v1
```

固定版本、隔离输入、单位转换、单次运行与审计命令见 [EndoGaussian baseline 说明](baselines/endogaussian_sim/README.md)。

### 8.7 EH-SurGS baseline

[EH-SurGS](https://github.com/IRMVLab/EH-SurGS) 将手术场景表示为 canonical 3D Gaussians，并用时间条件形变模型预测每个 Gaussian 的位置、尺度和旋转变化。其自适应运动层级为不同位置分配不同复杂度的运动基函数，使缓慢组织区域和快速局部形变不必共享同一个时间模型；本次训练保持官方 RGB/深度损失及 3000 次迭代配置。

本适配固定官方提交 `73fa04e6f5c21cc1685f728eccb1332e81ce620c`。唯一上游兼容补丁让自适应运动分块读取当前相机的真实内参，避免沿用官方 640×512 EndoNeRF 常数；没有修改形变网络、损失、优化器或 rasterizer。训练输入严格限制为前 80% 的合法双目 RGB、FoundationStereo 深度、公共组织 mask 和相机标定，7:1 留出帧及最后 20% 的全部观测均不进入训练。

官方实现没有材料点轨迹接口，因此 checkpoint 冻结后采用固定 Shape of Motion 查询锚定位移解码器：在 frame 0 用 EH-SurGS 自身 alpha/depth 得到 30 个查询起点，把目标时刻相对查询时刻的 Gaussian 位移作为属性，在查询时刻几何上光栅化。GT 深度、GT 3D、尺度拟合、ICP 和刚体/时间对齐均不参与解码。Future 数值只表示 EH-SurGS 原生时间场在最后 20% 的外推能力。

```bash
bash scripts/setup_eh_surgs_baseline_env.sh

SIM_GPU_ID=0 bash scripts/run_eh_surgs_sim_baseline_once.sh \
  sim03 repeat_01 0 outputs/eh_surgs_sim03_once_v1/repeat_01/sim03

bash scripts/run_eh_surgs_sim_three_repeats_two_gpus.sh \
  outputs/eh_surgs_sim_unified_three_repeats_v1 \
  outputs/eh_surgs_sim03_once_v1/repeat_01/sim03
```

完整环境、固定协议、轨迹解码和审计说明见 [EH-SurGS baseline 文档](baselines/eh_surgs_sim/README.md)。

## 9. 目录结构

```text
embodied_gaussians_fixed_super_best_sim/
├── src/                         # Gaussian、PBD、Warp/Torch 与 GUI 核心代码
├── scripts/                     # 数据生成、重建、测评、诊断和可视化脚本
├── FoundationStereo/            # 固定版本的双目深度子模块
├── data/sim/                    # 本地三套原始仿真数据（Git 忽略）
├── outputs/
│   ├── sim_three_datasets_medium_h3w4_three_repeats_v1/
│   │   ├── comparison_mean_std.md    # 均值、标准差和54条原始结果
│   │   └── comparison_mean_std.json  # 机器可读聚合数据
│   ├── sim_three_datasets_alltracker_medium_h3w4_three_repeats_v3/
│   │   ├── comparison_mean_std.md    # AllTracker三次聚合结果
│   │   └── comparison_mean_std.json
│   └── sim_three_datasets_alltracker_unified_protocol_three_repeats_v1/
│       ├── comparison_mean_std.md    # 最新同一rollout协议三次结果
│       └── comparison_mean_std.json
├── 刚度优化方案.md
├── 当前刚度优化方法与运行流程_小白公式版.md
└── PROGRESS.md                  # 数据、实现与实验演进记录
```

## 10. 当前限制与下一步

- 当前 C 组只辨识全局 distance stiffness 与阻尼；恢复局部柔软度需要可观测性约束下的连续或低维局部刚度场，不能简单地给每个粒子独立自由度。
- FoundationStereo 的无效像素和左右一致性误差会直接影响 3D 观测与刚度梯度；仿真 GT 深度仅用于诊断，不能作为正式算法输入。
- 当前已知夹持区域轨迹作为三组方法共同边界条件，实验评估的是夹持之外的组织传播和预测能力，不包含器械—组织接触反演。
- 三套数据都是可控仿真数据。后续迁移到真实 `super` 数据时，2D GT 需要人工标注，3D GT 则由双目深度反投影获得。
- 外观属性在线学习与物理优化已解耦；本次 CoTracker 三次复测未让外观优化污染物理 rollout 快照。更复杂的外观自适应仍需单独验证其可重复性。
- 最新统一协议下，AllTracker + 刚度更新改善了 SIM-03 抬升任务的未来几何预测，但 SIM-01/02 平面牵拉略有退化，说明当前全局材料梯度仍依赖形变方向与可观测性。

## 11. 参考项目与论文

- [Physically Embodied Gaussian Splatting](https://embodied-gaussians.github.io/)
- [SuFIA-BC / Orbit Surgical](https://orbit-surgical.github.io/sufia-bc/)
- [Real-to-Sim Deformable Object Manipulation: Optimizing Physics Models with Residual Mappings for Robotic Surgery](https://arxiv.org/abs/2309.11656)
- [FoundationStereo](https://github.com/NVlabs/FoundationStereo)
- [EndoGaussian](https://github.com/CUHK-AIM-Group/EndoGaussian)
- [EH-SurGS](https://github.com/IRMVLab/EH-SurGS)
- [Shape of Motion](https://github.com/vye16/shape-of-motion)

如果使用本仓库，请同时引用上游 Embodied Gaussians：

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

本项目为研究原型，不是临床或生产级软件。
