# Embodied Gaussians SIM baseline

本目录把 [Physically Embodied Gaussian Splatting](https://github.com/rai-opensource/embodied_gaussians)
适配到本仓库三套双目软组织仿真数据。这里严格区分两个对象：

| 名称 | 含义 | 正式用途 |
|---|---|---|
| `EG-Soft (paper reconstruction)` | 在固定官方代码初始化器上补回论文公式 (4)--(5) 的软体 shape matching | 软体 baseline 主结果（PSM/FK collision-only） |
| `EG-Public (rigid-only)` | 官方公开仓库实际发布的单刚体动力学与 visual-force 代码 | 能力边界审计/附录，不能冒充论文软体结果 |

官方 README 明确说明公开参考实现只支持刚体，论文中的 shape matching 没有发布；
但[论文](https://arxiv.org/html/2406.10788)明确用“每个粒子及其邻居组成一个 shape”
表示柔性物体，并报告了仿真和真实 rope 实验。因此，本适配不能被称为“官方软体代码原样运行”，
准确名称是 **基于固定官方参考代码的论文软体部分重构**。

## 1. 版本与代码隔离

- 固定上游 commit：`c97ec671f97af25985e0af8844c0aac8d8119b97`。
- 环境：`/Media_HDD/jwshan/conda_envs/eg_codex`，Python 3.11、Torch 2.6、Warp 1.7、gsplat 1.5.3。
- `initialize.py` 在运行前校验上游 README、`SimpleBodyBuilder` 和数据模型文件的 SHA-256；
  不匹配即退出。
- 新版 gsplat 默认 packed projection 与上游逐相机 background 张量不兼容；初始化入口只固定
  `packed=False` 恢复原批量张量语义，不改变 RGB+D 损失和优化器。
- baseline 代码不导入本项目的四面体 projector、三角面接触、轨迹观察器或在线材料模块。
- 所有新代码都在 `baselines/embodied_gaussians_sim/`；没有修改上游算法文件，也没有修改
  FixedSuperBest 方法文件。

## 2. EG-Soft 实现边界

保留的论文机制：

1. 只用医疗数据实际提供的左右双目在 frame 0 的 RGB、相机标定、公共组织 mask，以及与
   EndoGaussian/EH-SurGS 相同的 RGB-only FoundationStereo 深度建立包围盒；不使用额外 canonical
   scan 视角；
2. 调用上游 `SimpleBodyBuilder` 的粒子优化与 Gaussian grow 实现；
3. 每个粒子和 Delaunay 几何邻接粒子组成一个重叠 shape；论文未给邻居数/距离阈值，因此不引入
   可调的 k-NN 参数；
4. 严格实现论文公式 (4) 的 oriented-particle 矩阵、极分解以及公式 (5) 的 Jacobi 位置修正；
5. 每个 Gaussian 只刚性绑定到最近的一个粒子，保存粒子局部位姿；
6. 每个视频帧执行一次 `1/30 s` PBD，内部 20 substeps、4 次 Jacobi，使用上游默认重力和 z=0
   地面约束，帧末速度乘 0.9；普通物体每粒子质量严格使用论文的 `0.1 kg`，不误用只为仿真
   rope 指定的 `0.3 kg`；
7. 视觉校正每帧重建 Adam，执行5次随机相机 RGB L1 优化；位置、旋转、颜色、透明度参与，
   scale 固定；随后恢复 Gaussian 位姿，只将位移按 opacity 和 `Kp=60` 累加为 parent particle force；
8. 小于 2 mm 的 Gaussian 位移不生成视觉力。

明确禁止：

- `tissue_fixedsuperbest.npz` 的四面体拓扑、distance/volume 约束；
- triangle/tetra barycentric Gaussian binding 和随形变尺度更新；
- CoTracker、AllTracker、在线 FoundationStereo 状态校正；
- 在线刚度、阻尼或材料区域估计；
- GT tissue state、GT 3D轨迹和 GT 材料参数参与算法。

论文没有给出软体 `k_S` 数值。为遵守“不添加数据集调参值”，本实现不拟合 stiffness，而令
`k_S=1`，即每次完整执行公式 (5) 的约束投影；这不是声称论文公布了数值1，而是缺失实现下唯一
不引入松弛超参数的完整约束解释。邻域同样由无超参数的 Delaunay 几何邻接确定。由于官方公开仓库
没有发布这两项软体实现细节，结果必须继续命名为 `paper reconstruction`，不能写成官方代码原样结果。

### 初始化实现差异

论文描述了 densification，但官方公开 `SimpleBodyBuilder._grow_gaussians` 没有实际调用 densification。
本 baseline 忠实调用公开初始化代码，不从 FixedSuperBest 借用 densification 或表面 Gaussian 资产；
因此 metadata 会保留这一“论文描述—公开实现”差异。粒子半径严格使用公开
`SimpleBodyBuilderSettings` 默认的 `6 mm`，它位于论文报告的4--7 mm区间；不再按本数据厚度另设
半径。初始化 outlier radius/neighbor count、cohesion distance 和 z=0 ground 也恢复公开默认值。
输入图像保持医疗数据的正方形宽高比并把宽度缩放到论文的 640 px；没有通过拉伸伪造 640×360
视场。初始化 metadata 记录左右相机、frame 0、FoundationStereo 版本、输入哈希、实际 OBB、候选数
和最终粒子数。

## 3. 抓取驱动的两个版本

`psm-fk-collision-only` 是正式默认值：只读取允许的 PSM link poses，直接使用官方 tip mesh 顶点作为
运动学 robot particles，并只通过论文球碰撞传播作用；不设置额外的采样间距，也不读取 prescribed
tissue trajectory。它最符合论文的 robot-FK/collision 语义。若因原算法没有持久抓取约束而滑脱，
滑脱就是应报告的 baseline 行为，不得用 FixedSuperBest 的 persistent grip 修补。

`shared-prescribed-boundary` 仅保留为与 A/B/C 控制边界完全相同的可选协议对照。它把
`task_inputs/known_grasp_region_boundary.npz` 映射到最近 EG 粒子；该文件由生成器 GT grasp core
构造，不是 Embodied Gaussians 原算法输入，因此不得作为忠实主结果。

## 4. `EG-Public (rigid-only)`

`run_public_rigid.py` 直接构建上游 `EmbodiedGaussiansEnvironment`：整块组织是一个 rigid body，
所有组织 Gaussian 聚合为单个 body force/moment；PSM 使用已知 FK mesh collision。视觉力保留公开代码
默认值：3次 MSE 优化、`lr_means=0.0015`、`lr_quats=0.001`、外观学习率为0、`Kp=4`。
只把物理时间修正为数据的每帧一个 `1/30 s` step、20 substeps、4 XPBD iterations。

该版本没有软体自由度，只能放在附录/消融中。局部点间距离保持不变不代表软体预测准确，只说明
刚体约束被满足。

## 5. 输入隔离与统一协议

初始化只读取 frame 0 左右双目的 RGB、固定标定、公共 mask 和 RGB-only FoundationStereo 深度；
不读取 `canonical_scan/`。算法 rollout 只读取：

- 前80%非留出帧的双目 RGB；
- 相机内外参；
- 所有外部 baseline 共用的无分割误差 tissue mask adapter；
- 选择的 `task_inputs` 执行器输入。

`ground_truth/` 和 `evaluation/evaluation_points_30_non_grasp.json` 在 rollout 完成、状态与外观冻结前
不会被打开。之后 `export_tracks.py` 才读取30个 node ID 及 frame-0 左相机查询像素，并用 EG 自己
splat 得到的 alpha/depth 将像素反投影到模型三维点；GT depth 和 GT 3D 均不读取。按论文评测定义，
每个查询点选择最近的 query-time Gaussian，再把该点固定到 Gaussian 的 parent particle 局部坐标，
直接由已保存的 PBD 粒子位置/方向恢复整条轨迹。EG 已原生提供持久物理 frame，因此这里不使用
Shape of Motion 解码器；不做 ICP、Procrustes、尺度、位姿或时间对齐。
若查询像素没有通过与 EndoGaussian/EH-SurGS 相同的 `alpha > 1/255` 覆盖门槛，则按论文的
“query-time 最近 Gaussian”定义选择图像平面最近的可见 Gaussian，并以该 Gaussian 自身的相机深度
反投影原查询像素；回退数量与像素距离写入导出 metadata，仍不读取 GT depth/GT 3D。

协议与 EndoGaussian/EH-SurGS 相同：

- 前80%：`frame % 8 != 7` 才允许视觉更新，`frame % 8 == 7` 做7:1当前状态重建；
- 后20%：关闭视觉力与颜色/透明度更新，只保留已冻结状态、物理参数和已知控制开环预测；
- SIM-01/02 为300帧，SIM-03 为360帧；
- 固定30个非夹持点，双目2D与3D不对齐评测；渲染为组织层黑底 RGB + 独立 alpha；
- seed 0/1/2全部进入均值和样本标准差，不挑最好的一次。

## 6. 运行

环境已经存在，不需要重新安装。不要使用会触发 Conda 环境目录写检查的 `conda run`，直接调用该
环境解释器或下面的 wrapper。

先做不训练、不读取评测真值的协议审计和 shape-matching 单元测试：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/test_embodied_gaussians_shape_matching.py

/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/audit_embodied_gaussians_sim_protocol.py \
  --dataset-key sim01 \
  --variant paper-soft \
  --actuation psm-fk-collision-only
```

运行一次正式 EG-Soft（初始化、连续 rollout、查询导出、两种能力评测）：

```bash
SIM_GPU_ID=0 \
  bash scripts/run_embodied_gaussians_sim_baseline_once.sh \
  sim01 repeat_01 0 outputs/embodied_gaussians_sim01_repeat_01
```

可选 shared-boundary 协议对照需显式切换：

```bash
SIM_GPU_ID=0 EG_ACTUATION=shared-prescribed-boundary \
  bash scripts/run_embodied_gaussians_sim_baseline_once.sh \
  sim01 shared_boundary_audit 0 outputs/embodied_gaussians_sim01_shared_boundary_audit
```

运行公开 rigid-only 代码能力审计：

```bash
SIM_GPU_ID=0 bash scripts/run_embodied_gaussians_public_rigid_sim_once.sh \
  sim01 repeat_01 0 outputs/embodied_gaussians_public_rigid_sim01_repeat_01
```

按正式请求运行 SIM-01 两次、SIM-02/03 各三次并自动取均值与样本标准差：

```bash
SIM_GPU_ID=0 bash scripts/run_embodied_gaussians_sim_requested_repeats.sh \
  outputs/embodied_gaussians_sim_unified_three_repeats_v1
```

该批处理复用已完成的 SIM-01 seed 0，新增 SIM-01 seed 1，并执行 SIM-02/03 的 seeds 0/1/2；
最终生成 `requested_repeats_mean_std.{json,csv,md}`。所有请求运行都进入汇总，不选择最优结果。

正式输出拒绝覆盖，包含初始化 body 与 metadata、协议审计、逐帧诊断、全部粒子位置/方向、
30点预测轨迹、组织 RGB/alpha、reconstruction/future 指标和全目录 SHA-256。

## 7. 结果表命名

论文表建议使用：

- `Embodied Gaussians–Soft (paper reconstruction; collision only)`：主表；
- `Embodied Gaussians–Soft (paper reconstruction; shared boundary)`：可选控制边界对照；
- `Embodied Gaussians–Public (rigid-only)`：附录/能力边界。

不得把前两项写成“官方软体代码”，也不得把第三项的失败解释为论文声明不支持柔性物体。

### 正式 SIM 结果

SIM-01 按要求汇总 seeds 0/1，SIM-02/03 汇总 seeds 0/1/2；表中为算术均值 ± 样本标准差，
没有挑选最佳运行。三套数据均使用 `psm-fk-collision-only` 和论文原始软体物理参数：

| 数据集 | 能力 | Runs | 3D mean↓ (mm) | 双目2D mean↓ (px) | 2D有效覆盖↑ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SIM-01 Planar-X-Pull | 前80%在线7:1重建 | 2 | 3502.309 ± 405.600 | N/A | 1.54% ± 2.18pp | 7.324 ± 0.031 | 0.4599 ± 0.0003 | 0.7266 ± 0.0015 |
| SIM-01 Planar-X-Pull | 后20%开环物理预测 | 2 | 8069.362 ± 653.143 | N/A | 0.00% ± 0.00pp | 7.305 ± 0.000 | 0.4648 ± 0.0000 | 0.7215 ± 0.0000 |
| SIM-02 Planar-Y-Pull | 前80%在线7:1重建 | 3 | 911.075 ± 519.565 | N/A | 11.35% ± 16.04pp | 7.846 ± 0.583 | 0.4811 ± 0.0023 | 0.6906 ± 0.0379 |
| SIM-02 Planar-Y-Pull | 后20%开环物理预测 | 3 | 1905.701 ± 925.738 | N/A | 0.00% ± 0.00pp | 7.477 ± 0.000 | 0.4788 ± 0.0000 | 0.7171 ± 0.0000 |
| SIM-03 Edge-Z-Lift | 前80%在线7:1重建 | 3 | 5437.496 ± 2698.804 | N/A | 1.66% ± 1.44pp | 5.835 ± 0.062 | 0.2195 ± 0.0067 | 0.9595 ± 0.0070 |
| SIM-03 Edge-Z-Lift | 后20%开环物理预测 | 3 | 12621.398 ± 6275.488 | N/A | 0.00% ± 0.00pp | 5.636 ± 0.000 | 0.1949 ± 0.0000 | 0.9845 ± 0.0000 |

只要某个请求运行没有有效双目投影，该数据集的聚合 2D mean 就保守记为 N/A，同时单独报告
所有运行的有效覆盖率；这避免只统计仍留在视野内的点而低估误差。所有运行的 3D coverage 均为
100%，查询点只使用 EG 自身渲染的 depth/alpha，未使用 GT depth/GT 3D。模型在开环预测段均已
完全离开双目视场，所以 future 2D 为 N/A/0% coverage。失败结果没有用额外夹持、数据集刚度或
阻尼参数修补。

机器可读原始聚合和逐次结果见
[`requested_repeats_mean_std.{json,csv,md}`](../../outputs/embodied_gaussians_sim_unified_three_repeats_v1/requested_repeats_mean_std.md)。
