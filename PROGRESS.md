# SuPer grasp5 Progress

## 完整适配指南（2026-07-11）

当前最终适配路径、坐标系定义、tissue/ground 构建算法、严格 LND+dVRK PSM 位姿链、剩余偏差归因和复现命令已汇总到 `SUPER_COMPLETE_ADAPTATION_GUIDE.md`。该指南以 `bodies_v4`、`bodies_v5_table` 和 `psm_lnd_pose_driver.npz` 为当前权威产物，并明确排除历史 `bodies_v3`、frame 0 base 拟合和 direct-URDF q7 驱动路径。

## 当前数据产物

| 文件/目录 | 作用 |
|---|---|
| `data/super/grasp5_native/rgb/*-left.png` | 左目 rectified 图像序列。用于 SAM2 分割、深度估计输入、overlay 验证。 |
| `data/super/grasp5_native/rgb/*-right.png` | 右目 rectified 图像序列。和左目组成 stereo pair，用于 RAFT-Stereo/Python-SuPer 估计深度。 |
| `data/super/grasp5_native/depth/*-depth.npy` | 深度图，单位米。用于把 2D mask 像素反投影成 3D 点云，构建 tissue/ground。 |
| `data/super/grasp5_native/depth/*-disparity.npy` | 视差图。主要用于调试深度质量，正常建模主要用 depth。 |
| `data/super/grasp5_native/depth/*-depth_preview.png` | 深度图可视化预览。方便肉眼快速检查深度是否合理。 |
| `data/super/grasp5_native/depth/depth_generation_summary.json` | 深度生成记录：使用的 checkpoint、baseline、fx、每帧深度统计等。用于复现实验和排查尺度。 |
| `data/super/grasp5_native/calib_rectified.json` | rectified 后的相机标定：`K_left_rect/K_right_rect/P1/P2/Q/baseline` 等。用于深度换算、投影、后续构建相机模型。 |
| `data/super/grasp5_native/joints.json` | 从 bag 提取的 PSM1 原始关节序列，包含 5458 条 q、q_d、effort、timestamp、joint names。用于检查和生成 `robots.json`。 |
| `data/super/grasp5_native/super_dataset_metadata.json` | native 数据元信息：bag 路径、使用帧范围、输出目录、帧数、关节数等。用于记录这次提取过程。 |
| `data/super/grasp5_offline_demo/robots.json` | 当前 Embodied Gaussians `DatasetManager` 读取的机器人数据。demo 用它按时间驱动 PSM1 关节。 |
| `data/super/grasp5_offline_demo/cameras.json` | 当前项目的相机 manifest。告诉 `DatasetManager` 有哪些相机、外参、视频路径和 metadata 路径。 |
| `data/super/grasp5_offline_demo/videos/stereo_left.mp4` | 左目视频。demo 离线回放时读取它作为真实观测图像。 |
| `data/super/grasp5_offline_demo/videos/stereo_right.mp4` | 右目视频。当前主要用于双目验证/备用相机，后续也可做多视角观测。 |
| `data/super/grasp5_offline_demo/videos/stereo_left.json` / `stereo_right.json` | 每个视频的 metadata：相机内参 `K`、分辨率、每帧 timestamp。`OfflineCamera` 依赖它按时间索引视频帧。 |

## 目录定位

`data/super/grasp5_native` 是处理中间数据区，方便做深度、分割、点云和 debug。

`data/super/grasp5_offline_demo` 是项目运行数据区，目标是直接喂给 `DatasetManager` 和 demo。

## GUI 相机画面对齐（2026-07-11）

| 项目 | 状态 | 结论/产出 |
|---|---|---|
| 左目视频内容核对 | ✅ | `stereo_left.mp4` 首帧与 `rgb/000000-left.png` 均为 `1920x1080`；相位相关测得位移约 `(-0.0020, -0.0022) px`，可排除视频编码造成的画面平移。 |
| 相机模型核对 | ✅ | GUI、场景构建和 PLY 均使用 rectified 内参 `fx=fy=1742.7886`、`cx=860.4195`、`cy=682.2899`。相机位姿采用表坐标系下的 Blender camera-to-world，渲染时转换为 OpenCV 相机约定。 |
| GUI 偏差根因 | ✅ | `marsoom.CameraWireframe` 在主点不位于图像中心时错误地向纹理四角加入非对称偏移，并把四角归一化到球面，导致视频纹理平面与正确的场景投影不一致。左目主点相对图像中心约偏移 `(-99.6, +142.3) px`，旧实现四角射线误差约 `2.3~4.7 deg`。 |
| GUI 纹理平面修复 | ✅ | 在 `src/embodied_gaussians/embodied_visualizer/embodied_viewer.py` 中按 `(0,0)`、`(w,0)`、`(w,h)`、`(0,h)` 和真实 `K^-1` 重建四角射线，并将四角放在同一个成像平面上；离线相机与虚拟相机均使用该逻辑。未修改 conda 环境中的第三方包。 |
| Go To Camera 视野 | ✅ | 原始内参按当前 GUI 视口等比缩放，默认 `--camera-go-zoom 0.9`，完整包含左目画面并留约 10% 边距。`1.0` 表示刚好完整包含，值越小视野越大。 |
| 数值回投影验证 | ✅ | 修正后的纹理四角回投影为 `(0,0)`、`(1920,0)`、`(1920,1080)`、`(0,1080)`，数值误差小于 `0.001 px`，四角深度一致。 |
| GUI 人工验证 | ✅ | 用户已在虚拟显示器 GUI 中确认投影没有问题。 |
| Gaussian 颜色通道 | ✅ | `tissue.json` 中颜色已验证为 RGB；修复 viewer 将 gsplat RGB 输出上传到 `GL_BGR` 导致的 R/B 交换，`gaussian_texture` 现使用 `GL_RGB`。 |

运行命令：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
bash scripts/run_demo_browser_12.sh
```

需要进一步缩小 Go 后的视角时：

```bash
bash scripts/run_demo_browser_12.sh --camera-go-zoom 0.8
```

## PSM 尖端表面 Gaussian（2026-07-11）

| 项目 | 状态 | 结论/产出 |
|---|---|---|
| 运动学路径 | ✅ | dVRK xacro 重建 URDF，fixed base 经 LND/hand-eye 对齐；运行时将 `robots.json` 的 q7 展开为含 mimic 的 q14，由 Warp FK 驱动各 link。PSM 继续关闭自身、组织和地面碰撞。 |
| 旧视觉路径问题 | ✅ | 旧实现从 Warp shape/collision 几何采样后，只保留约 20 个腕部 Gaussian，并强制设置为各向同性 `0.5 mm`；同时没有可靠应用 visual origin，因此过小、过稀且可能偏离 visual mesh。 |
| 确定的视觉路径 | ✅ | 从 URDF `<visual>` mesh 离线采样，在 link-local 坐标中应用 mesh scale 和 visual origin，按表面法线构造贴面椭球，再按 link body id 绑定并随 FK 更新。 |
| 采样范围 | ✅ | 只保留 `PSM1_tool_main_link` 及其 URDF 下游子树，即插入杆、腕部、末端执行器和夹爪；不采样 base、yaw/pitch 和上方连杆。9 个下游 link 中 7 个具有 visual mesh。 |
| Gaussian 资产 | ✅ | `data/super/psm_robot/psm_surface_gaussians.npz`：1508 个 Gaussian、7 个 link。大 mesh 密度 `30000/m^2`，小 mesh 至少 64 个。 |
| Gaussian 尺度 | ✅ | 按每个 mesh 的采样间距计算；切向约 `0.7–3.75 mm`，法向约 `0.2–0.75 mm`，形成贴合表面的薄椭球。 |
| 表面验证 | ✅ | `psm_surface_gaussians_report.json` 记录各 link 覆盖数和点到 visual mesh 距离，最坏单点约 `0.27 mm`。 |
| CUDA 接入验证 | ✅ | 环境成功加载 1508 个 PSM Gaussian，总场景 6498 个 Gaussian，绑定 18 个 robot body 后世界坐标全部 finite。 |
| PLY 检查 | ✅ | `data/super/psm_robot/psm_tip_gaussians_scene_frame0.ply`；橙色为 URDF visual mesh，青色为尖端 Gaussian 中心。 |

重新生成尖端 Gaussian：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python scripts/build_psm_surface_gaussians.py
```

## SUPER tissue 视觉力稳定性（2026-07-11，2026-07-12 更新）

| 项目 | 状态 | 结论/产出 |
|---|---|---|
| Interaction 1 拉飞根因 | ✅ | tissue 是质量约 `0.01484 kg` 的单刚体，绑定 2490 个 Gaussian。旧实现单步 Adam 令每个 Gaussian 位移约 `2.598 mm`，随后直接求和为约 `1.21 N`、`0.025 Nm`，对应约 `82 m/s^2` 加速度。 |
| 颜色损失通道 | ✅ | 离线观测为 BGR、gsplat 为 RGB；SUPER visual-force loss 现先将观测 BGR 转为 RGB，避免颜色误差被错误解释为几何位移。 |
| 优化器状态 | ✅ | 每个 physics step 都从当前 Gaussian 位姿开始独立求解，因此 SUPER 现同步重置 Adam 一、二阶动量，防止跨帧累计漂移。 |
| 力的密度归一化 | ✅ | SUPER 按每个 body 的 Gaussian 数量归一化聚合力和力矩，使受力不再随 Gaussian 采样密度线性增长。 |
| 安全参数 | ✅ | `lr_means=0.0001`、`lr_quats=0.0001`、`kp=1.0`、Smooth L1 `beta=0.05`、总力上限 `0.005 N`、总力矩上限 `5e-5 Nm`。 |
| 修复后单步 CUDA 验证 | ✅ | Interaction 1 下 Gaussian 位移 P50/P95/max 均约 `0.1732 mm`；实际 tissue 总力 `4.97e-6 N`、加速度 `3.35e-4 m/s^2`、力矩 `1.35e-7 Nm`。 |
| 120 步稳定性对照 | ✅ | Interaction 1 相对 Interaction 0 的附加平移仅约 `[-3.3e-6, -5.6e-6, 0] mm`，最终线速度和角速度均为 0。 |
| 当前参与范围 | ✅ | 仅 tissue 的 2490 个 Gaussian 和 tissue body 参与 visual force；1508 个 PSM Gaussian 参与数为 0、实测最大视觉位移为 `0 mm`。PSM 自动视觉平移/refiner 已删除。 |

直接以 Interaction 1 启动：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
bash scripts/run_demo_browser_12.sh --visual-force-iterations 1
```

## PSM 颜色、手动修正与精确回放（2026-07-12）

| 项目 | 状态 | 结论/产出 |
|---|---|---|
| 单色根因 | ✅ | 旧 `build_psm_surface_gaussians.py` 将全部 1508 个器械 Gaussian 固定为 `[0.65,0.67,0.70]`，没有使用视频颜色。 |
| 多帧颜色烘焙 | ✅ | 新增 `bake_super_psm_gaussian_colors.py`，使用 12 个左目时刻和 strict LND pose 投影；357 个 Gaussian 直接观测、1151 个在同 link 局部传播，并保留 25 个蓝色标记点。最终 RGB std 约 `0.069-0.078`。 |
| 颜色预览 | ✅ | `data/super/psm_robot/psm_color_bake_frame0_overlay.png`；银色亮暗和蓝色标记已不再是一片平灰。 |
| Runtime LND FK | ✅ | `psm_lnd_kinematics.py` 负责 `q_dataset + q_manual_roll + q_manual_jaw`；零 correction 相比原 pose-driver 最大位置差 `1.9e-5 mm`、姿态差 `6.1e-6 deg`。PSM 不参与视觉力。 |
| GUI 手动平移 | ✅ | `Manual image X/Y` 范围 `[-5,5] mm`，`Manual camera Z` 扩大为 `[-30,30] mm`；相机向量经 `R_CW^T` 转成 world 平移并统一施加到 7 个可见 PSM link。 |
| GUI jaw 对称开合 | ✅ | 删除 `Wrist group camera X/Y/Z deg` 及对应远端刚体旋转路径；新增 `PSM jaw offset deg`，范围 `[-30,30] deg`，叠加到数据集 q7 jaw。mimic 保持 `jaw_mimic_1=+0.5*jaw`、`jaw_mimic_2=-0.5*jaw`。offset `+10 deg` 实测两个夹爪分别旋转 `+5/-5 deg`，其余 link 和整体平移不变。 |
| 精确时间戳播放 | ✅ | 以左目 1441 条 metadata timestamps 逐帧驱动视频与机器人，FPS 只控制播放速度。首/末时间戳为 `0.029093239/48.161777496 s`；末帧实际解码 index 1440，并立即自动暂停，不再让机器人继续到 `54.9896 s`。 |
| 视频/机器人索引 | ✅ | 视频和机器人统一使用 `searchsorted(..., side="right")-1` 的零阶保持规则；末帧选中机器人状态 `48.159168243 s <= 48.161777496 s`。 |

重新烘焙颜色：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python scripts/bake_super_psm_gaussian_colors.py
```

## PSM 器械多帧位姿适配（2026-07-11）

| 项目 | 状态 | 结论/产出 |
|---|---|---|
| 误差归因 | ✅ | 主要是适配方法问题，不是 depth 精度。旧路径将 SuPer q7 直接送入 dVRK URDF，并只在 frame 0 拟合 base；误差会随关节运动增长。 |
| 旧路径跨帧误差 | ✅ | 400 个采样状态中，`tool_main` 位置误差 P95 `30.7 mm`、姿态误差 P95 `6.28 deg`；腕部/末端位置误差 P95 约 `14.9~15.9 mm`。 |
| LND feature 单位 | ✅ | 修复 `build_super_psm_lnd_intermediates.py` 的重复 `x0.001`。`LND.json` feature 本身使用米；同时按原注释将 `grip_far` 的 `0.09` 修正为 `0.0095 m`。已重建 5458 帧 LND motion。 |
| 时间同步 | ✅ | 视频约 `29.77 Hz`，关节约 `99.28 Hz`。验证和新 overlay 均按视频 timestamp 查询最近关节状态，最近邻误差 P95 `4.77 ms`。旧 `validate_psm_projection.py` 已修正，不再按相同帧号错误配对。 |
| 确定的位姿路径 | ✅ | SuPer `LND.json` Modified-DH FK 决定器械运动；原始 `data/dvrk_model` 决定 CAD 表面。每个尖端 link 的 `T_LND-link->dVRK-link` 只由两个源模型在标准 `q=0` 下解析，运行时由 `handeye x LND FK(q) x model-link-offset` 更新 body。 |
| 无绝对修正 | ✅ | pose-driver 明确记录 `uses_dataset_reference_frame=false`、`uses_image_based_correction=false`；不使用视频 mask、frame 0、手工 base/roll 修正或深度结果。 |
| Canonical 模型一致性 | ✅ | LND 与 dVRK 对应 link 原点在 `q=0` 下的最大残差为 `0.00079 mm`，说明两个原始模型可直接建立固定坐标关系。 |
| 位姿驱动资产 | ✅ | `data/super/psm_robot/psm_lnd_pose_driver.npz`，包含 5458 帧、7 个尖端 visual link；`psm_lnd_pose_driver_report.json` 保存 canonical 模型变换和残差；生成脚本为 `scripts/calibrate_psm_lnd_pose_driver.py`。 |
| Runtime 接入 | ✅ | `apply_psm_lnd_pose()` 在跳转时间、每次 physics step 后覆盖 7 个尖端 body，并立即更新 body-bound Gaussian。原 dVRK articulation 不再决定可见器械姿态。 |
| CUDA 验证 | ✅ | 第 0 帧和第 3000 个状态切换成功，7 个 body 位姿 finite；实际 GUI 连续运行稳定，PSM base 无漂移。 |
| 更新后投影画面 | ✅ | `data/super/psm_robot/lnd_pose_validation/frame*_left_before_after.png`。左侧橙色为旧 direct URDF，右侧青色为严格 canonical LND+dVRK pose；对比图仅验证，不参与参数计算。 |

关键对比图：

- `data/super/psm_robot/lnd_pose_validation/frame000000_left_before_after.png`
- `data/super/psm_robot/lnd_pose_validation/frame000500_left_before_after.png`
- `data/super/psm_robot/lnd_pose_validation/frame001000_left_before_after.png`
- `data/super/psm_robot/lnd_pose_validation/frame001400_left_before_after.png`

重新生成 LND 数据、位姿驱动资产和对比图：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python scripts/build_super_psm_lnd_intermediates.py
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python scripts/calibrate_psm_lnd_pose_driver.py
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python scripts/validate_psm_lnd_pose_projection.py
```

启动更新后的 GUI：

```bash
bash scripts/run_demo_browser_12.sh
```

## SUPER tissue PBD/XPBD 软组织改造（更新至 2026-07-13，阶段 A--E 与 F 接触预检已完成）

> **历史方案，当前已暂停：** 本节保留当时的 PBD 实验和门禁记录，其中“当前 runtime/已切换”等表述只代表 2026-07-13 的状态。自 2026-07-19 起，正式 demo 已按用户决定改用文末 `bodies_v6_dense_frozen_table/tissue.json` 的单刚体组织；软体资产不再加载。

### 当前状态与根因

| 项目 | 状态 | 结论 |
|---|---|---|
| 运行时 tissue 资产 | ✅ 已切换 | SUPER 默认场景现从 `data/super/grasp5_native/soft_tissue_v1/tissue_soft.npz` 加载 11024 个 particle、41865 个 tetra 和 2490 个 skinned Gaussian；旧 `objects/tissue.json` 只保留作历史刚体资产，不再构造 runtime tissue。 |
| 历史“粒子”路径 | ✅ 已替换 | 旧资产的 1050 个点只是同一刚体上的 sphere collision shapes，没有进入 Warp `particle_q/qd/f`；当前 runtime 已改为 11024 个真实 PBD 粒子。 |
| 历史刚体根因 | ✅ 已修复 | 旧 `add_rigid_body()` 令组织只能整体平移/旋转；当前 tissue 使用 `add_soft_body()`、41865 个 tetra 和 Neo-Hookean XPBD 材料约束。 |
| Gaussian 绑定 | ✅ 已软体化 | 2490 个 tissue Gaussian 不再共绑一个 body id，而是通过 tetra/barycentric/rest-offset 蒙皮跟随局部粒子形变。 |
| 单独修改半径的作用 | 限于接触 | 旧刚体路径中半径不会产生局部形变；当前 soft runtime 中 `TISSUE_PBD_RADIUS_SCALE` 会改变 particle 的夹爪/ground contact shell，但仍不改变 tetra 材料拓扑与刚度。 |
| Warp 拓扑/接触能力 | ✅ | 当前 Warp `1.7.0.dev20250225` 已提供 `add_particle`、`add_tetrahedron`、`add_soft_mesh/grid`，XPBD integrator 也有 particle contact 和 tetrahedral constraint 路径；阶段 A 无需更换物理引擎。 |
| Warp 材料参数门禁 | ✅ B0 已通过 | 当前环境原生 `warp.sim.integrator_xpbd.solve_tetrahedra` 确实忽略 `k_mu/k_lambda/k_damp`；B0 已新增项目内 compressible Neo-Hookean energy XPBD projector，并在 CPU 与 A800 CUDA 上通过 `E` 单调性、`nu` 体积响应、timestep/iteration 敏感性和完整 tissue rest smoke。该 projector 尚未与 contact loop 交错，不能直接视为阶段 B runtime 完成。 |
| 视觉力边界 | 固定要求 | PSM visual force 保持关闭；tissue visual force 保留，并由当前 rigid `body_f` 路径改为 soft `particle_f` 路径。 |

### 阶段 A 实施结果（2026-07-12）

新增 `scripts/build_super_soft_tissue.py`，只生成和验证 soft asset，没有修改当前 runtime tissue，也没有打开 PSM collision 或 soft visual force。默认输入继续使用当前权威 `depth_v2/000000-depth.npy`、第一帧 tissue mask、`table_frame.json` 和 `bodies_v5_table/tissue.json`。

产物位于 `data/super/grasp5_native/soft_tissue_v1/`：

| 产物 | 内容 |
|---|---|
| `tissue_soft.npz` | table-frame rest particles、tet、surface faces、体积质量、接触半径、固定/候选支撑 mask、2490 个 Gaussian 的 tetra/barycentric/rest-offset binding。 |
| `metadata.json` | 输入 SHA256、参数、拓扑、体积/质量、Gaussian binding、材料求解器门禁和输出路径。 |
| `tissue_soft_surface.ply` | 四面体外表面预览。 |
| `tissue_soft_gaussian_binding.ply` | Gaussian 到最近 tetra 的 binding 距离预览；绿/黄/红分别表示不超过接触半径、网格间距和超过网格间距。 |

实测结果：

| 检查 | 结果 |
|---|---|
| 网格参数 | `spacing=1.5 mm`，`particle contact radius=0.7 mm`，密度暂取 `1000 kg/m^3`。 |
| 规模 | 1648 个有效 XY cells、11024 个粒子、41865 个正向 tetra、10048 个 surface faces。 |
| 拓扑 | 单连通；inverted=0、degenerate=0、nonmanifold tet faces=0；surface edge incidence 全部为 2，封闭表面通过。高度层中 2 处对角体素接触已用最小修改消除。 |
| 尺寸 | table-frame bbox 约 `81.06 x 63.70 x 15.00 mm`；离地最低粒子中心 `0.7 mm`。离散顶面相对高度场误差 P50/P95 约 `0.055/0.700 mm`，最大约 `0.972 mm`。 |
| 体积/质量 | tetra 总体积约 `28.259 cm^3`；按 `1000 kg/m^3` 得 `28.259 g`。旧 `14.844 g` 是 1050 个半径 1.5 mm collision sphere 的体积和，不是实测 tissue 质量，后续只作兼容对照。 |
| Gaussian binding | 1866 个中心直接位于候选四面体内，624 个投影到最近 tetra 并保存 rest offset；binding distance P95/max 约 `1.179/3.025 mm`。重心点加 rest offset 的 float32 rest-pose 重建最大误差约 `7.5e-9 m`。 |
| runtime 状态 | 阶段 A 生成资产时 `metadata.runtime_enabled=false`、当时尚未切换 demo；阶段 C 门槛通过后当前 runtime 已正式使用该 soft asset。 |

重新生成：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python scripts/build_super_soft_tissue.py
```

### 阶段 B0 material solver gate（2026-07-12）

新增 `MaterialTetrahedronXPBDProjector` 和 `solve_material_tetrahedra_xpbd`。项目内约束使用 compressible Neo-Hookean energy density：

```text
Psi(F) = mu/2 * (I1 - 3 - 2*log(J)) + lambda/2 * log(J)^2
C(F)   = sqrt(2 * V_rest * Psi(F))
```

`C(F)` 作为 XPBD arbitrary-energy constraint，累计拉格朗日乘子只在一个 substep 的 iterations 内保留。B0 仍是无 ground/contact/rigid body 的独立材料投影器；阶段 B 需要把该投影与 particle-ground、particle-shape contact 放进同一个 substep loop。

验证入口：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
PYTHONPATH=src /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/test_super_soft_tissue_solver_gate.py --device cuda
```

最终 CUDA 报告为 `data/super/grasp5_native/soft_tissue_v1/solver_gate_report.json`，设备为 NVIDIA A800 80GB；同一门禁也已在 CPU 通过。

| 门禁 | 实测结果 |
|---|---|
| 原生 Warp 参数无效复现 | 单 tetra 在 `E=5/30 kPa, nu=0.45` 时 tip displacement 都是 `0.503258 mm`、volume ratio 都是 `0.949674`，relative difference=`0`。 |
| 项目内 `E` 单调性 | 同一载荷下 `E=5 kPa` 位移约 `6.162 um`，`E=30 kPa` 位移约 `1.020 um`；高/低刚度位移比约 `0.1655`。 |
| 项目内 `nu` 体积响应 | `E=15 kPa` 时，`nu=0.35` 的 volume error 约 `5.130e-4`，`nu=0.45` 约 `2.144e-4`；更接近不可压缩时体积误差更小。 |
| timestep | `dt=1/60` 与 `1/120` 的平衡位移相对差约 `9.52%`，通过当前 `<10%` B0 门槛；阶段 B 使用更小 substep dt 时继续记录。 |
| iterations | 5 与 15 iterations 的响应相对差约 `0.0503%`，通过 `<5%` 门槛。 |
| 完整 tissue rest smoke | 11024 粒子、41865 tetra，3 个无载荷投影步全部 finite，max/mean rest drift 都为 `0 m`。 |
| B0 结论 | 所有 gates 在 CPU/CUDA 均为 true；材料语义门禁通过，但 contact integration、重力静置和 1000-step 稳定性仍属于阶段 B。 |

### 阶段 B real particle/tetra + ground smoke（2026-07-12）

阶段 B 已将 stage-A 资产真正转换为 Warp `particle_q/particle_qd/particle_f` 和 `tet_indices/tet_poses/tet_materials`，不再把这些点当作同一个 rigid body 上的 sphere shapes。新增 `SoftBody/TetraMesh/GaussianSkinning` schema、`SoftBody.from_npz()`、`EmbodiedGaussiansBuilder.add_soft_body()` 和 `SoftBodyHandle`。阶段 B 验收时使用 `add_gaussians=False` 隔离渲染；阶段 C 现已实现并验证 `add_gaussians=True`。

material energy projection 已与 Warp particle-ground contact 在每个 iteration 内交错；每个 substep 末尾额外执行 2 次 ground-only projection，并对零恢复地面接触做法向速度投影。这样消除了最后一次材料修正把底层粒子重新拉入地面造成的约 `20-30 um` 穿透和法向反弹。该路径目前仍不含 particle-shape/PSM contact。

验证入口：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
PYTHONPATH=src /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/smoke_super_soft_tissue_physics.py --device cuda
```

最终报告为 `data/super/grasp5_native/soft_tissue_v1/physics_smoke_report.json`，A800 CUDA 实测：

| 项目 | 结果 |
|---|---|
| 实际 model | 11024 个 Warp particles、41865 个 Warp tetra；256 个 `support_candidate_mask` 底部边缘粒子质量置 0。 |
| 材料/求解参数 | `E=15 kPa`、`nu=0.45`、`dt=1/60 s`、12 substeps、20 material/contact iterations、material relaxation `0.15`、速度阻尼 `12/s`。 |
| 1000-step finite | 全部采样位置、速度和 tet volume finite。 |
| tetra/体积 | 1000 steps 内 inverted tetra 始终为 0；最大总体积误差约 `0.5893%`，末步 total volume ratio 约 `0.994121`。 |
| ground/support | ground penetration 最大值 `0 m`；固定支撑粒子最大漂移 `0 m`。 |
| 位移 | 末步最大/平均粒子位移约 `0.248/0.120 mm`，在前 50 步内进入稳定平台，之后无整体下沉。 |
| 速度 | 所有采样的最大速度约 `0.09385 m/s`，该最大值来自恰好位于 ground 的少数接触粒子；末步 P95/P99/mean 约 `0.04449/0.07158/0.00599 m/s`。位置、穿透和体积没有随该接触修正速度累积漂移。 |
| 性能 | CUDA graph 下 1000 frames 用时约 `7.18 s`，约 `7.18 ms/frame`；不含渲染、PSM、Gaussian skinning 和 visual force。 |
| gates | finite、inverted=0、volume error `<2%`、penetration `<0.1 mm`、anchor drift `<1 um`、速度分位/上限门禁全部 true。 |
| 当时的 demo 状态 | 阶段 B 报告生成时为 `runtime_demo_switched=false`；该历史字段不回写。阶段 C 与主 runtime 联合门禁通过后，当前默认 GUI 已切换，见下文。 |

### 阶段 C tetra-to-Gaussian 蒙皮（2026-07-12）

阶段 C 已把 stage-A 保存的 `tet_ids/particle_indices/barycentric_weights/rest_offset/rest_quat` 注册进 `GaussianModel`。每个 soft Gaussian 使用四个粒子的 barycentric position 更新中心；由当前 tetra 变形梯度 `F=Ds*Dm^-1` 的 SVD polar rotation 更新 surface rest offset 和 quaternion。第一版明确保持 Gaussian scale 不变；tetra 退化或倒置时保持上一帧渲染位姿，不向 renderer 写入无效旋转。

`EmbodiedGaussiansBuilder.add_builder()` 同时对 soft particle、tet 和 Gaussian 索引做偏移，支持 SUPER 的 per-environment builder 复制。`EmbodiedGaussiansSimulator.update_gaussian_transforms()` 和无逐帧 Gaussian 存档的 loader 回放路径均已接入 soft skinning；rigid Gaussian 路径保持不变。

验证入口：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
PYTHONPATH=src /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/test_super_soft_tissue_skinning.py --device cuda
```

最终 A800 CUDA 报告为 `data/super/grasp5_native/soft_tissue_v1/skinning_gate_report.json`；同一门禁也已在 CPU 通过。

| 门禁 | A800 CUDA 实测结果 |
|---|---|
| builder 索引偏移 | 在 soft body 前插入无关 particle 和静态 Gaussian 后，soft particle/Gaussian 起点均正确偏移到 1；静态 sentinel 未被 soft kernel 改写。 |
| rest pose | 2490 个 Gaussian 的最大 mean 重建误差约 `5.27e-9 m`；最大 quaternion angle 数值误差约 `9.77e-4 rad`（float32 `acos` 在 dot 接近 1 时的分辨率）。 |
| 统一刚体变换等变性 | 对所有粒子施加同一旋转和平移后，Gaussian mean 最大误差约 `1.12e-8 m`；quaternion angle 数值误差约 `9.77e-4 rad`。 |
| 局部压缩连续性 | 中心区域平滑压缩时最大 Gaussian 位移约 `0.788 mm`；近区/远区平均位移约 `0.326/0.0132 mm`；最近邻位移跳变 P95 约 `0.0474 mm`。 |
| finite/orientation/scale | mean/quaternion 全部 finite；最大 quaternion norm error 约 `1.19e-7`；Gaussian scale bitwise 未修改。 |
| 结论 | `builder_offsets/rest_pose/rigid_equivariance/local_compression/scale_unchanged` 全部通过，`passed=true`。 |

阶段 C 蒙皮门禁通过后，已继续完成主 runtime bridge：标准 `XPBDIntegrator` 仍负责 PSM 刚体、关节和 particle prediction，但在有 soft tetra 且 `use_project_material_tetrahedra=true` 时屏蔽其材料参数无效的原生 tetra launch，再对已积分粒子执行项目内 Neo-Hookean material/ground interleaved projection。projector 会在标准 XPBD 复用 state buffer 之前保存 substep 起点，并在 material projection 后重建粒子速度。

新增 `scripts/smoke_super_soft_tissue_runtime.py`，通过真实 `EmbodiedGaussiansSimulator.physics_step()`（不是独立 smoke loop）每帧同时执行 12 substeps、20 material iterations 和 2490 个 Gaussian skinning。A800 CUDA 1000-step 报告位于 `data/super/grasp5_native/soft_tissue_v1/runtime_smoke_report.json`：全部 finite、倒置 tetra=0、最大总体积误差约 `0.5796%`、ground penetration=0、anchor drift=0、Gaussian 最近邻位移跳变 P95 最大约 `0.0143 mm`、quaternion norm error 最大约 `1.19e-7`，Gaussian scale 未修改；预热后物理+蒙皮约 `8.25 ms/frame`，不含渲染。

联合门禁通过后，SUPER 默认场景已从 `add_rigid_body()` 切换为 `add_soft_body(add_gaussians=True)`，并记录每个环境的 `SoftBodyHandle`。实际 PSM+soft+ground 场景已在 A800 启动并运行 100 帧：6498 个总 Gaussian（PSM 1508、tissue 2490、其余 ground），particle/Gaussian finite、倒置 tetra=0、末步总体积比约 `0.994229`。阶段 C 仍明确设置 `enable_particle_shape_contacts=false`、`enable_particle_particle_contacts=false`，并将 visual-force gradient/physics masks 设为空；100 帧实测两个 visual-force mask 计数均为 0。因此当前 `runtime_demo_switched=true`，但视觉力和 PSM 接触仍未开启。

### 阶段 D soft Gaussian force scatter + masked visual wiring（2026-07-13，核心链路已通过）

已新增与渲染解耦的 `EmbodiedGaussiansSimulator.scatter_soft_gaussian_forces()` 和 3 个 Warp kernels。输入是完整 Gaussian 数组上的 means displacement target，但 kernel 只遍历 `soft_gaussian_ids`，因此 PSM/ground/static Gaussian 不可能进入 particle force 路径。每个 soft Gaussian 先计算 `f_g=opacity*kp*delta_g` 并执行 per-Gaussian clamp，再按四面体 barycentric weights 原子散射到 4 个粒子；每条贡献除以该粒子的静态累计 Gaussian 支持权重，避免表面 Gaussian 密度直接放大局部载荷。`inverse_mass=0` 的固定粒子跳过。散射后执行 per-particle clamp，再以所有粒子力模长之和作为 tissue force budget 做全局缩放，最后加到 `state_0.particle_f`，供下一 physics step 消耗。

新增 `scripts/test_super_soft_force_scatter.py`。CPU 功能门禁和 1-frame 非 CUDA material-projector 动力接线已通过：

| 门禁 | CPU 实测 |
|---|---|
| 非 soft 隔离 | 对 sentinel 非 soft Gaussian 施加 `[1,-2,3] m` 的巨大残差，最大 particle force 仍为 `0 N`。 |
| 原子散射参考 | Warp scatter 与独立 Torch `index_add` 参考的最大粒子力误差约 `6.67e-11 N`。 |
| 锚点 | 所有 `inverse_mass=0` 粒子最大力为 `0 N`。 |
| 局部性 | 中心 128 个 soft Gaussian 加载时，近区平均粒子力约 `2.62e-6 N`，距离中心超过 `25 mm` 的远区最大力为 `0 N`。 |
| clamp | 故意输入过大目标后，clamp 前预算约 `0.7306 N`，全局 scale 约 `0.006844`，最终粒子力模长总预算约 `0.005000 N`；per-particle limit 同时通过。 |
| 最小动力接线 | 1 frame 后 finite、倒置 tetra=0、总体积比约 `0.995493`、anchor drift=0。CPU 使用同一 material projector 的非 CUDA-graph 路径，因为项目主 `physics_step()` 的 wrapper 只支持 CUDA capture。 |

正式 A800 `--device cuda --physics-steps 1000` 已完成，报告为 `data/super/grasp5_native/soft_tissue_v1/soft_force_gate_report.json`。持续中心局部载荷下 1000 步全部 finite、倒置 tetra=0、anchor drift=0；末步总体积比约 `0.994252`、最大粒子速度约 `0.0950 m/s`。非 soft 隔离、Torch 参考、锚点、局部性、per-Gaussian/per-particle/total clamp 和 bounded dynamics gates 全部为 true。因此 scatter 长期门禁已通过。

视觉 optimizer 到 scatter 的实际接线也已完成：`VisualForces` 可按显式 Gaussian ids 只开放 2490 个 soft Gaussian 的 pose gradients；SUPER 设置 `lr_quats=0`，第一版只允许 means target。`Frames` 增加 per-camera per-pixel `loss_weights_gpu`；soft visual force 开启且没有有效权重时会在 rasterization 前拒绝执行。masked Smooth L1/MSE 按权重和有效像素数归一化；optimizer 得到的 `visual_forces.means-gaussian_state.means` 通过已验证 scatter 写入 `state_0.particle_f`，由下一 physics step 消耗。PSM/ground Gaussian 梯度关闭，rigid body force mask 为空。

新增 `scripts/test_super_soft_visual_force_wiring.py`，CPU/A800 均通过：只有 2490 个 soft Gaussian 获得梯度；synthetic local target 可产生粒子力，固定粒子为 0、总预算约 `0.004689 N`；缺失/错误形状 pixel weights 会被拒绝，关闭 soft-force switch 时不写力。A800 报告为 `data/super/grasp5_native/soft_tissue_v1/visual_force_wiring_report.json`。

### 逐帧 tissue mask 与真实 masked RGB 一步闭环（2026-07-13）

新增 `scripts/propagate_super_tissue_masks.py`，使用本机已缓存的 `facebook/sam2.1-hiera-large`，以唯一人工标注的第一帧 `000000-tissue.png` 作为 mask prompt，对 stereo-left 全部 1441 帧离线传播。无需逐帧人工分割。输出位于 `data/super/grasp5_native/visual_force_masks_v1/`：`tissue_masks_packbits.npy`、`timestamps.npy`、`areas.npy`、`temporal_iou.npy` 和 `report.json`。mask 使用 packbits，运行时按时间戳随机访问，不把 1441 张全分辨率 bool 图常驻内存。

传播报告：1441/1441 帧返回、空 mask=0；首帧 SAM2 prediction 与人工 mask IoU 约 `0.999039`；相邻帧 IoU min/P05/median 约 `0.988184/0.994502/0.996747`；相对首帧面积比 min/P05/median/P95/max 约 `0.910254/0.912962/0.947615/0.995740/1.0`。当前 SAM2 安装缺少可选 `_C` fill-holes 扩展，官方路径自动跳过该后处理；以上统计来自实际保存结果，后续关键帧可视 QC 仍需要做。

新增 `PackedTissueVisualForceWeights` 并接入 `DatasetManager.update_frames()`：用最近精确相机时间戳解包 tissue mask，向内腐蚀 `7 px` 作为组织外边界、器械遮挡孔洞边界的保守安全带；tissue 内接近白色的高光像素降权到 `0.1`；没有 mask 的相机权重为 0。该做法不会使用当前有误差的 PSM 投影生成工具 mask。它仍不等于完整的显式 `dilated_tool_mask`，也尚未加入逐帧 invalid disparity confidence，这两项保留为下一层权重。

新增 `scripts/smoke_super_masked_visual_force.py`，已在 A800 运行真实第一帧闭环：真实 RGB + propagated tissue mask -> masked gsplat loss -> tissue-only Gaussian mean target -> bounded particle force -> 1 frame XPBD。腐蚀后有效 tissue pixels=`948520`，其中 `250` 个高光 pixels 降权，weight sum=`948295`；最大粒子力约 `1.56e-6 N`，总预算约 `0.005000 N`；PSM body force=`0`。消费该力后一帧 particle/Gaussian finite、倒置 tetra=0、总体积比约 `0.995035`，所有 gates 通过。报告为 `data/super/grasp5_native/soft_tissue_v1/masked_visual_force_smoke.json`。

首次真实 rasterization 还暴露并修复了 Conda CUDA 布局问题：Torch JIT 默认寻找 `$CUDA_HOME/include/cuda_runtime_api.h`，而当前环境实际在 `targets/x86_64-linux/include`。项目现在在导入 gsplat 前自动补充该路径；`gsplat_cuda.so` 已成功构建并完成上述闭环。

### 阶段 E：100 帧 masked visual-force 安全门槛（2026-07-13）

新增 `scripts/evaluate_super_masked_visual_sequence.py`。它按真实 stereo-left 时间戳和严格 LND PSM pose 运行两条分支：无视觉力 baseline，以及 `visual_iterations=1` 的 masked RGB 视觉力分支。默认每个相机帧运行 2 个 physics steps、每 3 个 physics steps 更新一次视觉力；100 帧共执行 200 个物理步和 66 次真实 `1080p` gsplat 反向传播。正式 A800 报告为 `data/super/grasp5_native/soft_tissue_v1/visual_sequence_report.json`。

100 帧安全门槛全部通过：particle/Gaussian 全部 finite、倒置 tetra=`0`、最大总体积误差约 `0.7918%`、最大粒子速度约 `0.1027 m/s`、anchor drift=`0`；相对 baseline 的最大组织中心漂移约 `1.23e-6 m`（`0.00123 mm`），远低于 `1 mm` 门槛。每帧 mask 均非空，66 次更新的 masked RGB loss 位于约 `0.02556–0.02861`；每次 particle force 模长总预算都被限制在约 `0.005000 N`，PSM body force 始终为 `0`。预热并完成 baseline 后，单独计时的视觉分支循环约 `0.0410 s/camera frame`；该数字不包含环境/视频初始化和 baseline 分支，不能直接当作完整 GUI 帧时。

评估时额外发现并正式记录了 GPU 重放底噪：第 0 帧尚未有视觉力被物理步消费，但重复分支相对 baseline 的局部非刚性差异已约 `0.1039 mm`；施力后整段最大分支差异约 `0.1206 mm`，超过底噪的部分约 `0.0167 mm`。根因是材料 projector 使用 GPU atomic add 把多个 tetra 修正累加到共享粒子，加法顺序不保证逐次 bitwise 一致。因此这份报告证明的是“真实图像视觉力闭环有限、稳定、不污染器械”，尚不能把两分支的全部位置差异因果归于视觉力，也不宣称 RGB loss 已完成几何配准。报告中已写入 `causal_visual_deformation_isolated=false`，防止后续误读。

虽然短序列安全门槛已过，默认 `--visual-force-iterations` 暂时仍保持 `0`：目前 66 次更新都触及 `0.005 N` 总预算，且视觉形变信号尚未明显高于 GPU 重放底噪；显式 tool mask 和 invalid-disparity 权重也未完成。当前可用 `--visual-force-iterations 1` 进行受控开启，但在因果对照和观测权重完成前不作为默认行为。

### 暂缓的改进方向（2026-07-13 决策）

当前不实施 graph coloring/确定性 tetra reduction：约 `0.1039 mm` 的 GPU A/A 重放差异不会引起翻转、体积爆炸或肉眼可见漂移，现阶段为消除它而改变 41865 个 tetra 的求解顺序，工程收益不足。该误差继续作为实验底噪记录；只有以后需要对小于约 `0.1 mm` 的视觉形变做严格因果归因时再恢复此项。

当前也暂缓显式器械 image mask、invalid-disparity 权重和 stereo depth/point-cloud/SDF 约束，不向 runtime 接入半完整深度路径。原因是 tissue mask 已覆盖 1441 帧，但现有全分辨率高质量 RAFT-Stereo `depth_hires_i48` 只有前 20 帧；旧报告的 `valid_fraction=1.0` 仅表示深度被范围裁剪后 finite，不能替代左右一致性或遮挡置信度。直接接入会造成前 20 帧使用 depth、之后退回 RGB 或报错的不一致行为。

以后恢复 stereo 方向时按以下顺序实施，而不是直接增大视觉力：先生成完整 1441 帧 disparity/depth；保存原始有效性并加入 left-right consistency、正视差、深度范围和局部 disparity outlier confidence；然后让 gsplat 输出 expected depth，以组织区域内的截断 signed depth residual 作为沿相机射线的 projective-SDF loss，RGB 仅作辅助；最后再用 `0.005/0.01/0.02 N` 阶梯门槛决定是否提高力上限。当前没有修改 loss、force 参数或默认开关。

### GUI 软化观察档（2026-07-13）

用户观察原 `E=15 kPa`、阻尼 `12/s`、视觉力总上限 `0.005 N` 的组织在 GUI 中接近刚体。先测试中档 `E=10 kPa`、阻尼 `6/s`、总视觉力 `0.02 N`：真实 30 帧全部安全门槛通过，无倒置、最大体积误差约 `0.8156%`、最大速度约 `0.1136 m/s`、anchor drift/PSM force=`0`，但最大粒子位移只从原约 `0.247 mm` 墠至约 `0.267 mm`，肉眼仍可能不明显。报告为 `visual_sequence_softened_10kpa_20mn_report.json`。

当前进一步切到仅供 GUI 比较的明显观察档：`E=5 kPa`、`nu=0.45`、速度阻尼 `4/s`、soft total visual-force limit=`0.05 N`；仍保留 12 substeps、20 material iterations、relaxation `0.15` 和原 256 个 support anchors，不通过减少求解迭代制造“假软”。6 帧真实 masked-RGB 预检中，4 次视觉更新全部触及 `0.05 N` 预算；finite、倒置 tetra=`0`、最大体积误差约 `0.8125%`、最大速度约 `0.1081 m/s`、anchor drift/PSM force=`0`，最大粒子位移约 `0.332 mm`。报告为 `visual_sequence_demo_5kpa_50mn_preflight.json`。

该观察档尚未通过 30/100 帧长期门槛，不能替代原正式 `15 kPa + 0.005 N` 安全基线，也不能证明 RGB 视觉力方向准确。默认 visual iterations 仍为 0；GUI 通过显式 `--visual-force-iterations 1` 开启，供肉眼比较软硬与可见变化。

用户继续观察 `5 kPa, nu=0.45` 仍偏硬后，当前 GUI 再切到更软档：`E=2 kPa`、`nu=0.35`、阻尼 `2/s`、视觉力仍为 `0.05 N`。6 帧预检全部 finite、倒置 tetra=`0`、anchor drift/PSM force=`0`、最大速度约 `0.1004 m/s`；最大粒子位移约 `0.538 mm`，为原 `15 kPa` 档约 `0.247 mm` 的两倍以上。最大体积误差约 `1.578%`，仍通过 `<2%` 门槛但安全余量已明显变小，因此该档只用于 GUI 软硬观察，不再无门槛继续降低刚度。报告为 `visual_sequence_demo_2kpa_nu035_50mn_preflight.json`。

为匹配视频中组织被夹起后的大延展，继续测试“降低约束+增大视觉力”。极限档 `E=0.5 kPa, nu=0.45, damping=1/s, total force=0.1 N` 在 6 帧内虽无翻转、最大位移约 `1.78 mm`、最大速度约 `0.155 m/s`，但最大体积误差约 `2.23%`，超过 2% 门槛，已拒绝作为 GUI 参数；失败报告为 `visual_sequence_demo_0p5kpa_nu045_100mn_preflight.json`。

当前采用通过门槛的折中高延展档：`E=0.75 kPa`、`nu=0.45`、阻尼 `1/s`、per-Gaussian/per-particle force limit=`4e-5 N`、soft total force limit=`0.075 N`。6 帧预检中最大粒子位移约 `1.13 mm`、最大速度约 `0.1605 m/s`、最大体积误差约 `1.713%`，finite、倒置 tetra=`0`、anchor drift/PSM force=`0`；施力后相对重放底噪的最大局部差异约 `0.213 mm`。报告为 `visual_sequence_demo_0p75kpa_nu045_75mn_preflight.json`。这仍是短时 GUI 观察档，尚未通过 30/100 帧门槛。

`Show Forces` 已适配 soft particle 路径。旧控件只读取 rigid `visual_forces.forces`，而 soft Gaussian 的 `body_id=-1` 令该数组为零；现在改为读取 Gaussian-to-particle scatter 后、乘过 global clamp scale 的实际 particle force。可视化仅选动态、有 soft Gaussian 支持且受力非零的粒子，并保留最强的最多 1000 个，避免绘制 11024 根线。每个向量由主干和四个三维箭头翼组成；新增 `Soft Force Display Gain`（默认 10）只改变显示长度，不改变物理力，同时显示当前箭头数量。刚体 Gaussian 的旧显示路径保留。箭头几何门禁已验证一个非零向量生成 5 条线段、零向量不生成伪箭头。

### 研究依据

- Müller et al., Position Based Dynamics：直接投影位置约束，适合实时、稳定的可变形仿真。<https://matthias-research.github.io/pages/publications/posBasedDyn.pdf>
- Macklin et al., XPBD：用 compliance 和累计拉格朗日乘子减弱刚度对 timestep/iteration 的依赖，并提供约束力估计。<https://mmacklin.com/xpbd.pdf>
- Camara et al., Soft tissue deformation for surgical simulation：手术软组织可使用体粒子和 clustered shape matching，粒径/刚度应通过真实形变标记标定。<https://link.springer.com/article/10.1007/s11548-016-1373-8>
- Liu et al., Real-to-Sim Registration：与 SuPer/dVRK 最接近；沿重力方向把观测表面扩展成四面体体网格，并把视觉表面注册作为 PBD 动态约束。<https://arxiv.org/abs/2011.00800>

### 确定的技术路线

1. 使用 **裁剪后的四面体 particle+tets XPBD**，不使用规则长方体 `add_soft_grid`，也不使用无拓扑随机点弹簧网络。
2. 从现有 tissue mask、`depth_v2` 和 ground plane 构建平面局部高度场；只保留 mask footprint 内的有效体素。
3. 阶段 A 已采用网格间距 `1.5 mm`、particle contact radius `0.7 mm`；实际为 11024 粒子、41865 四面体。
4. 每个有效 hexahedral cell 拆成 5 个正定向 tetra；禁止当前 `sample_indices(max_particles)` 随机删点，因为会破坏拓扑。
5. 质量按 tetra 总体积分配；阶段 A 以软组织近水密度的数值起点 `1000 kg/m^3` 得到约 `28.259 g`。旧刚体 `14.844 g` 只作为兼容对照，不作为真实质量标定目标。
6. 材料扫描必须推迟到 solver gate 通过之后。计划仍可从 `E={5,15,30} kPa`、`nu={0.35,0.45}` 起步并转换为 Lamé 参数，但必须先证明改变 `E` 会单调改变受载位移、改变 `nu` 会改变体积响应，并检查 timestep/iteration 敏感性；这些仍只是调参起点，不是 SuPer 真值。
7. 阶段 B 稳定参数已收敛为 `dt=1/60 s`、12 substeps、20 material/contact iterations、material relaxation `0.15`；8 substeps/10 iterations 的短测曾出现倒置和约 8% 体积误差，不作为后续默认值。
8. 第一阶段使用 ground contact 加少量底面/边缘锚点；不要固定整个底面，否则组织无法被提起。当前为增加可提起性，已将 256 个底部周边候选固定点确定性均匀抽稀到 64 个。
9. 阶段 F 已只启用两片 PSM 夹爪 collision mesh 与 soft particles 的 particle-shape contact；整条机械臂、shaft、ground 和 rigid self-collision 仍关闭。

### 规则粒子与论文中非规则密集粒子的结论（2026-07-13）

论文示例中“大小球不规则、密集堆积”和当前 tissue “平面方向规则排列”属于两种不同的离散化路线，不能只根据球的视觉排列判断哪一个更软或更真实。论文中的球常直接作为 volume particles，并用邻域、弹簧或 clustered shape matching 建立约束；当前实现里球主要表示 Warp particle 的位置和接触半径，真正承载材料形变的是粒子之间的四面体及 Neo-Hookean XPBD energy constraint。

当前 tissue 也不是二维平面：它以 `1.5 mm` 规则 XY 网格采样第一帧 mask/depth 高度场，再沿 table-frame Z/重力方向填充至最多约 `15 mm` 厚，形成 11024 个三维粒子和 41865 个 tetra。顶视图中多个 Z 层重叠，加上 XY 坐标规则，因此视觉上像一张规则平面。

选择规则裁剪四面体作为第一版稳定基线的原因：现有观测只可靠给出第一帧顶面，底面和内部未知；结构网格最容易从高度场得到封闭体积、正向且可复现的 tetra、体积质量和稳定 Gaussian binding。当前已验证 inverted/degenerate/nonmanifold=0。直接随机堆球再做三维 Delaunay 会额外引入密度不均、空洞、邻域突变和 sliver tetra，必须先做质量清理，否则“看起来更自然”不等于材料响应更可信。

该选择不是最终真实性假设。规则网格可能带来轴向偏置，对斜向拉伸、剪切、弯曲边界和复杂内部形状的各向同性不如质量良好的非规则 tetra。后续在视觉力和接触链路稳定后，保留当前规则网格作为 control，新增同一封闭体积内的 Poisson-disk interior sampling + constrained Delaunay tetrahedralization，删除或改善 sliver tetra，重新计算体积质量和 Gaussian binding；用相同载荷比较位移、体积保持、方向偏置、倒置率和 frame time，而不是仅比较粒子截图。只有该 A/B 对照显示非规则网格有明确收益，才替换默认资产。

### Gaussian 与软体的绑定

物理粒子和 Gaussian 必须继续分开采样，通过四面体蒙皮关联：

1. 为每个 Gaussian 查找包含它或距离最近的 rest tetra。
2. 保存 4 个 particle indices、barycentric weights 和 `rest_offset`。直接包含时 offset 为零或极小；外部表面 Gaussian 使用到 tetra 投影点的 offset，避免第一步蒙皮把表面吸进体内。
3. 每个 physics step 更新 `mean_g = sum(w_i * particle_q[i]) + R_tet * rest_offset_g`，其中 `R_tet` 是 rest 到 current tetra 的 polar rotation。
4. 用同一个 polar rotation 更新 Gaussian quaternion；接近退化/倒置时回退到上一帧或 rest rotation。
5. 第一版保持 Gaussian scale 不变；稳定后再用变形后的 covariance 更新椭球尺度。
6. 不允许把 Gaussian 直接改成物理粒子，也不允许把所有 Gaussian 继续绑定到伪 soft rigid body。

### tissue 视觉力到 PBD 粒子的耦合

1. 保留 BGR->RGB、Smooth L1、Adam reset 和 tissue-only Gaussian gradient mask，但 photometric loss 必须进一步限制在逐帧 `visible_tissue_mask & ~dilated_tool_mask` 内；高光、无效视差和遮挡区域降权。第一帧人工 mask 用 SAM2 video propagation 离线传播，只在失败关键帧人工纠偏，不逐帧手工标注。
2. 每个 tissue Gaussian 产生位移目标 `delta_g`；第一版仅使用 means 位移，设置 tissue `lr_quats=0`。
3. 计算 `f_g = opacity * kp * delta_g`，不再按 body 聚合为一个 wrench。
4. 按 barycentric weights 反分配：`particle_f[i] += w_i * f_g`，使用 Warp atomic add。
5. 按每个 particle 的累计 Gaussian 权重或支持面积归一化；mass=0 的锚点不受视觉力。
6. 增加 per-particle force、total tissue force 和单步位移上限，避免局部 Gaussian 密度造成尖峰。
7. visual force 在 physics step 后写入 `state_0.particle_f`，由下一次 XPBD step 消耗；PSM Gaussian 和 PSM body 始终不参与。
8. RGB 只作为低权重辅助；专业版本以 stereo depth point-cloud/SDF surface registration constraint 为主。器械坐标未修正期间，器械膨胀 mask 内不产生 tissue 视觉力，遮挡区域由 XPBD 和邻域形变传播预测。

### 需要修改/新增的代码

| 文件 | 需要的修改 |
|---|---|
| `scripts/build_super_bodies_from_first_frame.py` | 不建议直接大改；复用高度场函数，停止随机粒子抽样。 |
| `scripts/build_super_soft_tissue.py` | ✅ **已新增**；生成并自检 rest particles、tet indices、封闭 surface faces、体积质量、fixed/support mask、radius，以及 Gaussian tetra/barycentric/rest-offset binding；输出 `soft_tissue_v1/tissue_soft.npz + metadata.json + PLY`。 |
| `scripts/test_super_soft_tissue_solver_gate.py` | ✅ **已新增**；复现原生 Warp 材料参数无效，并验证项目内 energy-XPBD 的 `E/nu/dt/iterations` 响应和完整 stage-A tissue rest smoke；输出 `solver_gate_report.json`。 |
| `src/embodied_gaussians/scene_builders/domain.py` | ✅ 已新增 `SoftBody/TetraMesh/GaussianSkinning` schema 和 `SoftBody.from_npz()`；不再用现有 `Particles` schema 表示 soft runtime。 |
| `src/embodied_gaussians/embodied_simulator/builder.py` | ✅ 已新增 `add_soft_body()` 和 `SoftBodyHandle`；调用 Warp `add_particle()`/`add_tetrahedron()`，验证 runtime/rest volume，并支持 `none/fixed/support_candidate` anchor mode。保留 `add_rigid_body()` 供其他场景使用，并已支持注册 soft Gaussian。 |
| `examples/embodied_environments/super_embodied/super_embodied.py` | ✅ tissue 已切换为 `add_soft_body(add_gaussians=True)`；当前使用 `E=0.75 kPa, nu=0.45`、阻尼 `1/s`、soft force `0.075 N`，support anchors 从 256 抽稀为 64。PSM rigid visual force 与 particle-particle contact 仍关闭；particle-shape 只开放两片夹爪 collision mesh。 |
| `src/embodied_gaussians/embodied_visualizer/embodied_viewer.py` | ✅ `Show Forces` 已支持实际 post-clamp soft particle force；最多显示 1000 个最强动态粒子，使用主干+四翼箭头和独立 display gain，且保留 rigid Gaussian force 显示。 |
| `src/embodied_gaussians/embodied_simulator/gaussians.py` | ✅ Gaussian model 已增加稀疏 soft binding arrays：soft Gaussian id、tet/particle indices、weights、rest offset 和 tet inverse rest pose。 |
| `src/embodied_gaussians/embodied_simulator/warp.py` | ✅ 已新增 particle->Gaussian skinning，以及 Gaussian force barycentric scatter、per-particle clamp、scaled apply kernels。 |
| `src/embodied_gaussians/embodied_simulator/simulator.py` | ✅ `update_gaussian_transforms()` 已处理 rigid/soft Gaussian；soft scatter 已由 masked RGB optimizer 自动调用；缺少 pixel weights 时 fail closed。另补 Conda CUDA header 路径供 gsplat JIT。 |
| `src/embodied_gaussians/embodied_simulator/visual_forces.py` | ✅ 已支持显式 Gaussian-id gradient participation 和 soft-force settings；SUPER 只开放 soft ids，PSM/ground gradients 与 rigid force 均关闭。 |
| `src/embodied_gaussians/embodied_simulator/frames.py` | ✅ 已增加 per-camera per-pixel loss weights、形状/finite/非负验证。 |
| `src/embodied_gaussians/embodied_simulator/visual_force_masks.py` | ✅ 已新增 packed timestamped tissue mask provider、7px 保守腐蚀、高光降权和无 mask 相机归零。 |
| `src/embodied_gaussians/dataset/dataset_manager.py` | ✅ 视频帧更新后同步更新对应时间戳的 visual-force pixel weights。 |
| `src/embodied_gaussians/physics_simulator/integrator.py` | ✅ project-local compressible Neo-Hookean energy XPBD projector 已与 particle-ground constraint 交错；阶段 F 新增 post-material particle-shape contact、单次修正上限和 tetra 体积安全线搜索。 |
| `src/embodied_gaussians/physics_simulator/simulator.py` | ✅ `PhysicsSettings` 已增加 material/contact 参数；particle-shape contact 可按 shape allow-list 过滤。material projector 开启时屏蔽 Warp 重复且无界的原生 particle-shape solve，只保留项目内有界接触路径。 |
| `scripts/smoke_super_soft_tissue_physics.py` | ✅ **已新增**；通过真实 `add_soft_body()` model 运行重力、材料和 ground contact 的 1000-step CUDA graph smoke，输出逐段 finite/tet volume/penetration/velocity/performance 报告。 |
| `scripts/test_super_soft_tissue_skinning.py` | ✅ **已新增**；验证 builder 索引偏移、rest pose、统一刚体变换等变性、局部压缩连续性、quaternion finite/norm 和 scale 不变；输出 `skinning_gate_report.json`。 |
| `scripts/smoke_super_soft_tissue_runtime.py` | ✅ **已新增**；通过主 `EmbodiedGaussiansSimulator.physics_step()` 联合运行 physics+skinning 1000 步，验证 finite/tet volume/ground/anchors/Gaussian continuity/orientation/scale/performance；输出 `runtime_smoke_report.json`。 |
| `scripts/test_super_soft_force_scatter.py` | ✅ **已新增**；CPU 功能门禁和 A800 1000-step 持续局部载荷正式门禁均通过，输出 `soft_force_gate_report.json`。 |
| `scripts/test_super_soft_visual_force_wiring.py` | ✅ **已新增**；验证 pixel-weight fail-closed、soft-only gradients、optimizer target scatter、锚点/预算和 disable switch；CPU/A800 均通过。 |
| `scripts/propagate_super_tissue_masks.py` | ✅ **已新增**；从第一帧人工 mask 用 SAM2.1-large 传播 1441 帧并输出 packbits mask/时间戳/QC statistics。 |
| `scripts/smoke_super_masked_visual_force.py` | ✅ **已新增**；真实第一帧 masked RGB 到 XPBD 的一步 A800 闭环全部通过。 |
| `scripts/evaluate_super_masked_visual_sequence.py` | ✅ **已新增**；运行精确时间戳 baseline/视觉力双分支，记录 100 帧 mask/loss/force/体积/速度/倒置/漂移和 GPU A/A 重放底噪；安全门槛通过，但因果形变尚未从底噪中完全隔离。 |
| `src/embodied_gaussians/physics_simulator/saver.py` | 已支持 `particle_q/qd/f`；补充 soft binding/拓扑资产版本信息即可。 |

### 分阶段实施与验收

| 阶段 | 产出 | 必须通过的检查 |
|---|---|---|
| A. soft 资产 | ✅ 已完成：`soft_tissue_v1/tissue_soft.npz`、2 个 PLY、metadata | inverted/degenerate/nonmanifold=0；封闭单连通；尺寸、高度、质量来源和 Gaussian rest binding 已数值验证。 |
| B0. solver gate | ✅ 已完成：project-local energy XPBD + CPU/CUDA report | 原生 Warp 参数无效已复现；`E`/`nu` 单调性、dt `<10%`、iterations `<5%`、完整 tissue rest smoke 全部通过。 |
| B. 无渲染 XPBD | ✅ 已完成：schema、`add_soft_body()`、material+ground projector、1000-step CUDA report | 1000 steps finite；倒置 tetra=0；最大总体积误差约 `0.5893%`；ground penetration/anchor drift=0；约 `7.18 ms/frame`。 |
| C. Gaussian 蒙皮与主 runtime | ✅ 已完成：soft binding model + tetra polar skinning + loader/runtime update + 主 Simulator material bridge + 默认 SUPER 切换 | 独立 rest/rigid/local gates 全过；主 runtime 1000 steps 全过，约 `8.25 ms/frame`；真实 PSM+soft 场景 100 帧 finite、倒置=0、视觉力/PSM 接触 mask=0。 |
| D. particle visual force | ✅ core 完成：scatter CPU/CUDA、1000-step load、soft-only optimizer wiring、1441-frame tissue masks、真实 masked RGB 一步闭环 | 非 soft/PSM force=0；scatter 参考约 `7.13e-11 N`；长期载荷无倒置；真实闭环预算 `0.005 N`、PSM force=0。 |
| E. 回放评估 | 🟡 安全门槛完成、准确性方向暂缓：精确 timestamp 100-frame baseline/visual 双分支报告 | 100 帧 finite、倒置=0、体积误差 `<0.8%`、PSM force=0、整体漂移约 `0.00123 mm`；GPU A/A 局部重放底噪约 `0.1039 mm`。graph coloring、显式 tool mask、invalid disparity 和 stereo depth/SDF 已记录但按当前决定暂不实施。 |
| F. PSM 接触 | 🟡 核心接触与 401 帧安全预检完成：仅两片夹爪 particle-shape contact | 前 401 帧 finite、倒置=0、最大总体积误差约 `1.422%`、最大速度约 `0.224 m/s`；已观察到实际接触候选。完整 1441 帧与肉眼夹取/拉伸真实性仍待 GUI 验收。 |

### 下一步首要任务

阶段 E 的原 `15 kPa + 0.005 N` 100 帧安全基线已通过并保留报告；当前代码为匹配大延展暂时切到 `0.75 kPa + nu 0.45 + 0.075 N + damping 1/s`，只通过 6 帧预检，最大体积误差约 `1.713%`。需要用户先在带真实 soft force 箭头的 GUI 判断拉伸是否达到预期，再决定保留、回退或做 30/100 帧正式门槛。默认 visual iterations 仍为 0。

阶段 F 已开启夹爪接触。当前接触参数为 margin `2 mm`、摩擦 `0.25`、relaxation `0.6`、每次投影最大修正 `0.15 mm`，并保留 tetra 最低体积比 `0.2` 的安全线搜索；401 帧预检报告为 `psm_tissue_contact_0p15mm_preflight.json`。下一步应在 GUI 中检查夹爪是否确实能带动并拉长组织；若只发生穿透/滑动，再调整接触代理或摩擦，不再通过降低材料刚度掩盖接触问题。非规则 tetra 仍只在视觉/接触链路稳定后做 A/B 对照。

### 阶段 F：PSM 夹爪—组织接触（2026-07-13）

已启用两片夹爪 `PSM1_tool_wrist_sca_ee_link_1/2` 的 invisible URDF collision mesh（单环境 shape ids `28/30`）与 11024 个 soft particles 的接触。Warp 的 soft collide 不读取 rigid collision flag，因此 Simulator 新增 shape allow-list：碰撞生成前临时把其余 shape 标为 `GEO_NONE`，生成后恢复；整条机械臂不会参与 tissue contact。PSM body 由 LND pose 外部驱动并设为无限 solver mass，记录 pose 的角速度/COM 速度只用于接触摩擦，组织反力不会反推器械。

为让夹爪接触与项目内材料求解兼容，material projector 开启时不再执行 Warp 原生、重复且无单步上限的 particle-shape projection；接触改在每个 substep 的材料投影后执行 2 次。用户指定每次 particle correction 上限为 `0.15 mm`。此外只在候选修正会把任一 tetra 压到 rest volume 的 `20%` 以下时，通过 12 次二分线搜索统一缩小该次接触；最终 401 帧预检中最小 scale 为 `1.0`，因此用户指定的 `0.15 mm` 没有被变相缩小。

失败记录必须保留：无 final contact 时 stored contact-plane residual 曾约 `1.4 mm`；把接触重复到 20 次 material iterations 会在约第 400 帧爆炸；保留 Warp 原生 contact 再叠加有界 final contact 时，即使 `E=0.75 kPa` 也会在第 318--321 帧出现 1 个 inverted tetra。根因是同一 particle-shape contact 被原生 XPBD 和项目内 projector 重复求解，而不是 `0.15 mm` 本身。去掉重复路径后，frame `0..400`、每相机帧 2 physics steps 的 A800 预检通过：最大 contact candidates/active proxy=`406/86`、最大总体积误差约 `1.4216%`、最大速度约 `0.2242 m/s`、最大位移约 `2.319 mm`、inverted tetra=`0`。stored contact-plane penetration proxy 最大约 `1.426 mm`，但它使用物理帧开头生成的旧 contact point/normal，经历切向滑动后不是 fresh mesh SDF；仍需 GUI 与后续 fresh-SDF 检查确认是否有可见穿透。

用户观察夹爪仍可能从离散粒子间穿过后，当前先采用增大 contact shell、暂不直接实现动态连续 triangle surface collision 的增量方案。规则网格 spacing=`1.5 mm`，原 particle radius=`0.7 mm`、直径 `1.4 mm`，轴向相邻粒子之间存在名义 `0.1 mm` 缝隙；当前 radius 已提高到 `0.8 mm`、直径 `1.6 mm`，大于网格间距并封住规则轴向名义缝隙。该半径目前由 Warp 同时用于夹爪和 ground contact，不等价于真正无缝、可变形的组织三角碰撞表面。

`0.8 mm` 半径的 frame `0..400` A800 contact-only 回放通过：最大 candidates/active proxy=`426/97`，相对 `0.7 mm` 的 `406/86` 均增加；最大总体积误差约 `1.3843%`、最大速度约 `0.2285 m/s`、最大位移约 `2.281 mm`、inverted tetra=`0`、contact safety scale 最小仍为 `1.0`。报告为 `psm_tissue_contact_radius0p8_preflight.json`。stored contact-plane penetration proxy 约 `1.542 mm`，其数值随 radius 定义本身增加且仍不是 fresh mesh SDF，不可用作“实际网格穿透深度”的最终结论。

视觉力总预算已从 `0.075 N` 提高到 `0.10 N`，per-Gaussian/per-particle cap 仍为 `4e-5 N`。6 帧真实 masked-RGB 短门禁中 4 次更新都达到约 `0.10 N`，finite、倒置 tetra=`0`、最大总体积误差约 `1.418%`、最大速度约 `0.1135 m/s`；post-force nonrigid change 超过 GPU replay floor 约 `0.328 mm`。报告为 `visual_sequence_demo_0p75kpa_100mn_radius0p8_preflight.json`。这只是短预检，尚未替代长期视觉力门禁。

### 物理粒子之间的绑定策略（2026-07-13）

物理粒子之间不是“每个球与周围所有球用硬弹簧绑住”，也不是彼此独立。11024 个粒子由 41865 个四面体构成固定拓扑；每个 tetra 同时关联 4 个 particle，并通过 compressible Neo-Hookean XPBD energy constraint 约束其拉伸、剪切和体积变化。两个粒子只有在共享至少一个 tetra 时才是直接材料邻居；更远粒子通过多个 tetra 的共享顶点逐级传播形变。因此夹爪直接接触少数 surface particles 后，力会沿 tetra 邻接图传播到周围，而不是瞬间平均分配给全体粒子。

该绑定是有柔度的材料约束，不是焊死：`E=0.75 kPa` 控制整体拉伸/剪切顺应性，`nu=0.45` 控制近不可压缩的体积响应，20 次 material iterations 与 relaxation `0.15` 负责数值投影。64 个 support particles 通过 inverse mass=`0` 固定，其余为动态粒子。particle-particle collision 当前关闭；相邻粒子不靠球体互撞维持组织形状，而靠共享 tetra 维持。

2490 个视觉 Gaussian 与物理拓扑分离：每个 tissue Gaussian 绑定到一个 tetra 的 4 个顶点，使用 barycentric weights 加 rest offset 跟随形变；视觉优化得到的 Gaussian force 再按相同 4 个权重反向 scatter 到这些 particle。随后 tetra material constraints 把影响传给更远的物理邻居。增大 particle contact radius 只扩展夹爪/ground 看到的碰撞壳，不改变 tetra 邻接、材料刚度或 Gaussian 的四点绑定关系。

### PSM 纯论文图像跟踪完整结果（2026-07-14）

已新增 `scripts/track_super_psm_paper_exact.py`，并从原始视频完整运行 frame `0..1440`，不再停在 frame 320。该路径严格使用 `online_dvrk_tracking` commit `cb2a264167aaf05b5a9c20da885d48568f78311f` 的估计链路：首帧 SurgicalSAM2 提示后原始视频传播、1500 个 BO 候选初始化、MixAngle、NvDiffRast 轮廓损失、每帧 CMA-ES 3 轮/70 population、对称夹爪、上一帧关节初始化、ContourTipNet 和论文 Kalman。估计器未加入光流、重新提示 SAM、缺失夹爪补全、LND/base/joint prior、速度限制或额外平滑；唯一输入修正是把人工点击的 `[paper-left, paper-right]` 顺序转换为论文实现要求的 `[right, left]`。

完整运行共 1441 帧，用时约 `102.33 s`，初始化 loss=`4.32134`，逐帧 mean loss=`0.41694`。原始状态保存为 `data/super/psm_tracking/tracking_states_paper_exact.npz`，shape 分别为 `cTr=(1441,6)`、`joints=(1441,4)`、packed SAM masks=`(1441,64800)`；论文位姿转换到当前 PSM 七个视觉 link 后为 `visual_poses_video_paper_exact.npz=(1441,7,7)`，再按数据时间戳重采样为 `psm_paper_exact_pose_driver.npz=(5458,7,7)`。这一 CAD-link 注册和重采样只发生在估计完成后的驱动适配阶段，不反馈到论文估计器。`--psm-pose-driver paper` 已指向这份 exact driver；此前增强结果保留为 `paper_robust`。

完整结果忠实暴露了论文方法在该视频上的失效，而不是通过后处理掩盖：frame 0 基本可对齐；frame 160 已有明显夹爪/姿态漂移；frame 320 的原始 SAM mask 漏掉第二片夹爪，轮廓优化只能对错误观测继续拟合，导致两片 CAD 夹爪方向错误或重叠；后段仍有跳变。相邻帧 cTr 旋转跳变的 p50/p95/max 约为 `3.93/19.93/42.76 deg`，平移约为 `2.09/6.26/10.48 mm`，关节角约为 `7.38/33.44/68.04 deg`。因此 exact driver 已可用于方法复现和投影对比，但未经补爪、遮挡处理及时序约束前，不应直接当作可靠的物理接触真值。

论文四部件原生 CAD 投影位于 `data/super/psm_tracking/paper_native_exact/`；当前 PSM tip-only 投影对比位于 `data/super/psm_tracking/projection_paper_exact/`。两者均覆盖 frame `0,160,320,480,640,800,960,1120,1280,1440`。`tracking_report_paper_exact.json` 记录代码版本、完整帧数、运行参数和 `non_paper_estimator_additions=[]`。

## PSM 三部件全序列位姿矫正、GUI 注册与夹爪修复（更新至 2026-07-17）

### 当前唯一推荐的 GUI 驱动方式

当前实际观察和后续组织接触实验统一使用 `corrected` 驱动；`strict`、`registered_lnd`、`paper`、`paper_robust` 和 `hybrid` 只保留为消融对照与历史结果，不在一次运行中混合：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
bash scripts/run_demo_browser_12.sh \
  --psm-pose-driver corrected \
  --psm-visual-mode tip
```

`tip` 只隐藏长器械杆的视觉 Gaussian，保留腕部和两片夹爪；夹爪的物理接触不受该显示选项影响。若只想检查整根器械外形，可临时改为 `--psm-visual-mode full`。

### 从纯论文结果到当前 corrected 驱动的处理链

```text
纯论文 1441 帧结果（只读基线）
  -> 141/142/160/320 三部件人工诊断
  -> body / jaw_left / jaw_right 全 1441 帧独立传播
  -> 时间戳对齐的 LND 运动作为稳定先验
  -> 每帧小范围相机姿态、腕关节和共同 jaw 残差优化
  -> 时序平滑，并强制保留四个人工关键帧
  -> 论文 CAD 到当前 GUI URDF 的一次固定 SE(3) 注册
  -> 两片夹爪按同一 wrist parent 和同一物理铰点重建
  -> 重采样成 GUI 使用的 5458 状态 pose driver
```

纯论文 `tracking_states_paper_exact.npz` 和 `psm_paper_exact_pose_driver.npz` 没有被覆盖；图像矫正只产生新的 corrected 资产。

### 已完成的工作与结果

| 项目 | 状态 | 结果/产物 |
|---|---|---|
| 三部件关键帧分割 | ✅ | `scripts/experiment_super_psm_part_segmentation.py` 在 frame `141,142,160,320` 分别提取 `body/jaw_left/jaw_right`，每帧保存 `frameXXXXXX_masks.npz` 和可视化；输出在 `data/super/psm_tracking/part_segmentation_experiment/`。该实验解决了原 union mask 把杆端当夹爪、或漏掉第二片夹爪后无法区分身份的问题。 |
| 关键帧位姿验证 | ✅ | `scripts/refine_super_psm_part_keyframes.py` 从 registered LND 先验出发，只优化有界的小残差；四个关键帧中两片夹爪尖端最终误差均小于 `0.78 px`。输出为 `part_pose_refinement/corrected_keyframe_states.npz`、`metrics.json` 和 `contact_sheet.png`。 |
| 三部件完整传播 | ✅ | `scripts/propagate_super_psm_part_masks.py` 已覆盖 frame `0..1440`，共 `1441` 帧，用时约 `72.43 s`；人工 anchor 为 `141,142,160,320`，每 25 帧注入 LND 投影提示，资产为 `part_masks_full_sequence/part_masks_full.npz`。 |
| 完整逐帧矫正 | ✅ | `scripts/track_super_psm_part_corrected.py` 已优化全部 `1441` 帧；所有帧的优化目标都比初值低。经修正范围 A/B 后，当前默认上限为整体平移 `3 mm`、整体旋转 `8 deg`、wrist `12 deg`、jaw `25 deg`。loss 的 P50 从 `1.2764` 降到 `0.3228`；位移修正 P50/P95 为 `0.874/1.120 mm`，旋转修正 P50/P95 为 `2.820/4.124 deg`。本次完整运行约 `238.44 s`。 |
| 固定 GUI 几何注册 | ✅ | `scripts/rebuild_super_psm_registered_drivers.py` 使用当前 tip Gaussian 与 registered paper CAD 做一次固定 SE(3) 拟合，得到约 `[-1.278,+6.734,-9.346] mm` 平移和 `3.518 deg` 旋转。registered LND 与 corrected 共用这一个不可随帧变化的注册，避免把首帧 CAD 坐标差误当作运动误差。 |
| registered/corrected 驱动重建 | ✅ | 已生成 `psm_registered_lnd_pose_driver.npz` 和 `psm_part_corrected_pose_driver.npz`，均为 GUI 的 `5458` 状态；视频帧级资产分别为 `visual_poses_registered_lnd.npz` 和 `visual_poses_part_corrected.npz`。 |
| GUI 驱动覆盖问题 | ✅ | GUI 现在直接加载所选 pose-driver，并在跳帧、播放和每次 physics step 后重新施加到两个 Warp state。roll/jaw 手动量已改为在所选 driver 上叠加的 tracked-parent 关节增量，不再通过 q7 重算 strict FK，因此不会悄悄覆盖 registered/corrected；相机方向的整体手动平移仍保留。 |
| GUI 自旋/夹爪手动修正 | ✅ | `SUPER Playback` 已对所有 pose driver 暴露 `PSM self-spin offset deg`（`[-180,180] deg`）和 `PSM jaw opening offset deg (+open)`（`[-30,30] deg`）。自旋默认值现为目视校正的 `-27 deg`。两个增量均以当前 selected driver 为基线，绕 URDF tracked-parent 的真实 `z` 轴做刚体旋转，不重算或覆盖 `corrected`。数值门禁确认 `-27 deg` roll 使全部远端 link 共同刚性旋转 `27 deg`，`17 deg` jaw 使左右各对称旋转 `8.5 deg`。验证脚本为 `scripts/validate_super_psm_gui_pose_offsets.py`。 |
| 夹爪共同铰点与平行铰轴 | ✅ | 旧适配先把左右夹爪当成独立刚体；第一轮共同铰点修复又错误地用 CAD 注册矩阵共轭 jaw 旋转，造成“共点但不同轴”，圆形铰链端在稳定闭合段仍有约 `4.14 deg` 的 P50 不平行误差。当前完全按 `psm.urdf` 重建：两 jaw 的 parent 相同、origin=`(0,0,0)`、axis=`(0,0,1)`，相对运动严格为 `Rz(+jaw/2)` 与 `Rz(-jaw/2)`；encoder jaw 是基线，图像只提供共同开合残差。 |
| URDF 完整远端链与 wrist 零点 | ✅ | jaw-only URDF 修复仍把“逐部件论文 wrist 绝对位姿”与 URDF jaw 混合，导致杆线离开夹爪开合平面 P50/P95 达 `16.28/19.36 deg`。根因是 paper wrist-pitch 首帧约 `-29.09 deg`，而同帧 encoder/URDF q4 约 `+1.64 deg`；首帧注册已吸收该约 `-30.7 deg` 零点差，不能再次把 paper 绝对角送入 URDF。当前以 tracked `tool_wrist_link` 为空间锚点，用 literal URDF wrist-pitch/yaw/jaw 父子链重建 links 2..6；q4 严格使用 encoder，q5 使用 encoder 加 paper 图像残差。 |
| 夹爪数值门禁 | ✅ | 稳定闭合段 frame `548..1246` 共 `699` 帧：旧独立注册 pivot gap P50 约 `5.0063 mm`，完整 URDF 远端链为数值 `0 mm`；两圆形铰链轴的不平行误差 P50 约 `1.9e-6 deg`；杆线离开夹爪开合平面的 P50/P95 从混合链全序列的 `16.28/19.36 deg` 降至完整 URDF 全序列的 `1.87/2.19 deg`，稳定闭合段为 `1.89/1.93 deg`。报告与图位于 `urdf_distal_chain_validation/`。 |
| 投影验证 | ✅ | 当前完整 URDF 远端链并叠加默认 `-27 deg` 自旋的实际 GUI tip Gaussian 投影位于 `gui_projection_corrected_urdf_chain_roll_minus27/`，覆盖 frame `0,480,560,800,1120,1280,1440`；三维父子链对照图位于 `urdf_distal_chain_validation/jaw_shared_pivot_comparison.png`。 |
| 修正范围 A/B | ✅ | 原范围 `2 mm / 5 deg / wrist 8 deg / jaw 20 deg`、当前范围 `3 mm / 8 deg / 12 deg / 25 deg` 和更宽的 `4 mm / 10 deg / 16 deg / 30 deg` 均完成全部 `1441` 帧实验。当前范围把 loss P50/P95 从原范围的 `0.3736/1.0239` 降到 `0.3228/0.9527`，且没有引入明显的新时序跳变；更宽范围会更强地拉动低可信帧，因此不作为默认。科学投影和实际 GUI cyan Gaussian 对照均位于 `correction_bound_comparison/`。 |

### registered LND prior、corrected 和 GUI 的关系

- `registered_lnd`：稳定的 LND 编码器运动，加上一次固定的 paper-CAD 到当前 GUI-CAD 注册；它不使用逐帧图像残差，是 corrected 的基准线。
- `corrected`：在同一 registered LND 基准上加入三部件图像提供的小范围时序矫正。当前 GUI 和后续接触实验使用这一份。
- GUI 不再自己重新求一套位姿；它只按视频时间戳读取 corrected driver，并把七个可见 link 的位姿写入仿真。
- 图像里看到的 registered LND prior 与 GUI 曾不一致，不是 prior 本身失效，而是旧 GUI 适配链存在 CAD 坐标注册差以及 tracked driver 被 q7/strict FK 路径覆盖的可能；上述固定注册和 offset 隔离已经修复这两个问题。

### 当前仍存在的限制

1. **夹爪 mask 的长序列置信度并不高。** body 的 P05/P50/P95 为约 `0.708/1/1`，但 jaw-left 有 `848` 帧、jaw-right 有 `902` 帧低于 `0.35`。因此 corrected 使用 LND 作为主干，只允许小残差，并对残差平滑；不能把传播 mask 当作逐帧真值。
2. **器械自转角仍可能有小偏差。** 当前优化已把整体旋转上限从 `5 deg` 放宽至 `8 deg`、wrist 残差从 `8 deg` 放宽至 `12 deg`，但仍优先保证稳定而不是强行追随低置信度轮廓。真实工具型号、工具零点和 roll 方向尚未用独立标定板或工具型号参数完成标定。
3. **放宽上限不能修复错误观测。** frame `800` 和 `1120` 的传播 jaw mask 明显不可靠；扩大范围后尖端误差没有改善，frame `800` 反而略增。因此这里需要补人工 anchor 或修复 mask，而不是继续增加位姿自由度。
4. **剩余夹爪外形差异主要不是共同铰点问题。** 当前 `psm.urdf` 的左右夹爪都使用 `meshes/tool_wrist_sca_link_2.stl`，第二片通过旋转翻转得到；这等于假设左右 CAD 镜像。若真实 LND 的齿形、厚度、弯曲或型号不同，姿态正确时轮廓仍会不同。
5. **GUI 的夹爪显示采样偏稀。** 当前每片夹爪只有 `64` 个 surface Gaussian，切向尺度约 `0.827 mm`，会把小尺寸金属夹爪显示得偏粗、偏圆；这与 CAD 型号误差会叠加。
6. **尚未完成完整 1441 帧的人工 GUI 验收。** 自动矫正、驱动生成、闭合段数值门禁及抽样投影已经完成；仍需从头到尾肉眼检查 corrected 模式，重点看快速自转、遮挡、再次张开以及组织接触片段。

### 下一步优先级

1. 先用上述唯一 `corrected + tip` 命令完成一次完整 GUI 人工验收，并记录仍有明显偏差的帧号；不再同时比较多种驱动。
2. 从实验记录或实物确认器械的具体型号与代际，再比较 JHU dVRK 官方 CAD、当前 CAD 和论文独立左右夹爪 CAD；在型号未确认前不盲目替换整套 URDF。
3. 保留当前运动学和共同铰点，只替换 wrist/jaw 的 visual mesh；碰撞 mesh 单独验证，避免外形更换破坏已经稳定的组织接触。
4. 将每片夹爪的 Gaussian 从 `64` 提高到至少 `512`，用相同帧和相机做轮廓 A/B；只有高密度显示仍不匹配时，才把主要误差归因到 CAD。
5. CAD 与显示层确定后，再针对剩余 roll/self-rotation 偏差做工具专用零点和角度标定。

关键复现命令：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

# 完整三部件传播
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/propagate_super_psm_part_masks.py

# 完整 1441 帧小范围位姿矫正
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/track_super_psm_part_corrected.py

# 用当前 GUI CAD 注册重建 registered 与 corrected 驱动，并强制共同夹爪铰点
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/rebuild_super_psm_registered_drivers.py

# 夹爪共同铰点数值验证
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/validate_super_psm_gui_jaws.py
```

## 时间戳同步 FoundationStereo 深度首批结果（2026-07-19，v3 已被稠密 v4 取代）

> **废止说明：** `depth_v3_foundation_timestamped` 把左右一致性和 RAFT 置信度直接变成 `NaN`，导致平面和组织点云不完整。该目录只保留作失败对照，后续建模不得再使用其中的 `depth_modeling.npy` 或 `depth_high_confidence.npy`；当前推荐输入见下方 v4 记录。

### 本轮范围与当前状态

本轮只生成和检查新深度，没有重建 `bodies_v4`、`bodies_v5_table` 或 `soft_tissue_v1`，也没有修改器械 pose driver。现有全局/table 坐标继续冻结；尤其不重新计算或覆盖已有 `X_table_camera`。深度与组织/平面构建使用 `eg_codex`，器械图像矫正继续使用 `online_dvrk`，两套环境不混用。

| 项目 | 状态 | 结果 |
|---|---|---|
| 主模型 | ✅ | 官方 FoundationStereo Vit-Large，仓库 commit `6e8806816b533e4d13ddbb95ffa907b797060a62`，checkpoint SHA256=`60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1`。 |
| 对照模型 | ✅ | 现有 RAFT-Stereo checkpoint，SHA256=`d22e84c0e431bf31d7cc66902c40601859eb40b35ef7f4399ea81276c2915819`；只做左右一致性、模型差异和冲突检查，不替代 Foundation 主深度。 |
| 运行环境 | ✅ | `/Media_HDD/jwshan/conda_envs/eg_codex/bin/python`，Python `3.11.15`、Torch `2.6.0+cu124`、CUDA `12.4`、`cuda:0`。为 FoundationStereo 补充 `timm 1.0.19`、`einops 0.8.1`、`imageio 2.37.0`；没有改 Torch/CUDA。 |
| 时间同步 | ✅ | 每个左目时间戳在右目时间轴上找前后两帧，两次估计视差后线性插值到左目时刻，不再把相同编号的左右图直接配对。frame 0 实际使用 right frame `1/2`，时间差 `-7.664/+38.881 ms`，插值系数约 `0.16466`。 |
| 推理参数 | ✅ | frame `0..4`，Foundation/RAFT 都为 `32` iterations；Foundation hierarchical 开启；左右一致性阈值 `1.5 px`；模型冲突阈值 `3 mm`；深度范围 `35–250 mm`。 |
| 输出 | ✅ | `data/super/grasp5_native/depth_v3_foundation_timestamped/`，共约 `378 MB`；每帧均包含 Foundation/RAFT disparity、depth、置信度 NPZ、mask 和 comparison PNG。完整配置与统计在 `depth_generation_summary.json`。 |

### 哪一帧用于平面和组织

- **组织仍以第 0 帧为建模帧。** 第 0 帧左目图、tissue mask、深度都处于同一套 `1920x1080` rectified 像素坐标，可直接反投影。不会把第 1–4 帧的组织深度平均进去，避免组织或器械轻微运动造成双层表面。
- **地面以第 0 帧建立候选平面，第 1–4 帧只做静态稳定性检查。** 在真正重建前先看各帧法向和 offset 是否一致；只有通过后才考虑用多帧稳健统计降低噪声。无论是否多帧统计，平面都先在现有相机坐标中求解，再用冻结的旧变换表达，不改变全局/table 坐标定义。
- **旧 SAM2 第 0 帧分割可以继续用。** `000000-tissue.png` 和 `000000-ground.png` 与第 0 帧左图同为 `1080x1920`，没有改变 rectification、裁剪、缩放或帧号。相邻帧报告里复用它们只表示固定像素 ROI 诊断，不冒充已经传播到这些帧的新语义分割。

### 三类深度文件的含义

| 文件 | 含义 | 后续用途 |
|---|---|---|
| `NNNNNN-depth.npy` | Foundation 主深度；已通过 Foundation flipped-pair 左右一致性、正视差和深度范围检查。 | 保留原始主结果和算法对照。 |
| `NNNNNN-depth_modeling.npy` | 推荐建模候选；保留 Foundation 有效点，只在 RAFT 也有效且两模型相差超过 `3 mm` 时剔除该点。 | 后续 frame 0 tissue/ground 候选输入。 |
| `NNNNNN-depth_high_confidence.npy` | 两个模型都有效且相差不超过 `3 mm` 的严格交集。 | 只用于审计、局部可信度和误差可视化，不直接独立拟合地面。 |

这里不能把 RAFT 当硬门槛覆盖全部地面：frame 0 ground mask 内 Foundation 有效率约 `71.14%`，RAFT 只有 `24.99%`，严格交集约 `20.83%`。严格交集存在明显空间选择偏差，虽然单帧点面残差更小，但 frame `1..4` 相对 frame 0 的拟合法向偏差达到约 `8.7–10.8 deg`，因此已拒绝“只用 strict high-confidence 深度建地面”的方案。改用 RAFT 冲突否决后，frame 0 总有效率约 `81.47%`，组织 mask 内约 `94.80%`，保留了 Foundation 主模型的空间覆盖。

### 首帧和多帧质量结果

| 检查 | 旧 `depth_v2` | 新 Foundation 主深度 | 新 modeling 深度 | 结论 |
|---|---:|---:|---:|---|
| frame 0 地面拟合点面绝对残差 P95 | `10.48 mm` | `2.88 mm` | `2.90 mm` | 从明显错误降到约 3 mm，达到本轮首要门槛。 |
| frame 0 地面拟合点面绝对残差 P50 | `4.09 mm` | `1.08 mm` | 约 `1.1 mm` | 中位误差明显下降。 |
| frame 0 组织 Foundation 有效率 | 未作为新门禁统计 | `94.89%` | `94.80%` | RAFT 冲突否决几乎不损失组织覆盖。 |
| frame 0 组织两模型绝对差 | - | P50=`0.168 mm`，P95=`1.145 mm` | 同一差异图用于否决 | 组织主体上两模型高度一致。 |

Foundation 主深度在 frame `1..4` 相对 frame 0 的平面法向差约 `0.54/1.69/1.10/1.58 deg`，plane `d` 差约 `+0.18/+2.68/+0.16/-0.92 mm`；modeling 深度对应法向差约 `0.49/1.58/1.10/1.66 deg`，没有 strict hard-intersection 的空间偏差。frame 2 的 offset 仍高约 `2.5 mm`，所以当前结论是“首帧深度已显著改善，可以进入候选重建”，还不能宣称多帧平面完全收敛。

### 坐标保护记录

本轮脚本只写入新 `depth_v3_foundation_timestamped` 目录；对既有坐标/场景资产的定向 `git status` 为空。当前保护哈希记录如下，后续重建前后必须再次核对：

- `data/super/table_frame.json`: `2f288cf3c9e65a8d86579e811524bb423e7b38dd063a028c014b715cfbaa2a7e`
- `bodies_v5_table/build_metadata.json`（含冻结 `X_table_camera`）: `a7a830a95d84cbc36bd27db40d02fba932d6ecec35bf358967d8a08a17cb391b`
- `bodies_v5_table/ground_plane.json`: `12b7362bd568d76d614d404acabb6afa26263ddd0292f6a9541cf2975b8748bd`
- `psm_part_corrected_pose_driver.npz`: `21df09849b08d6cef1ae47c694e7400b6df5228e0805c5c35ecf3d68e2ef648a`

### 复现命令与下一步

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
XFORMERS_DISABLED=1 /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/generate_super_depth_foundation_timestamped.py --frames 0,1,2,3,4
```

该阶段原计划使用 `000000-depth_modeling.npy`，现已因孔洞问题取消。后续统一改用下方 v4 的全稠密 `000000-depth.npy`，仍只构建新版本候选目录，不覆盖当前权威场景。

## FoundationStereo 全稠密深度修正（2026-07-19，当前推荐 v4）

### v3 为什么看起来不完整

FoundationStereo 实际已经为 `1920x1080` 的每个像素输出有限视差。v3 的问题不是模型没有预测，而是后处理错误地要求同一左目帧对时间戳前后两个右目配对都通过 flipped-pair 左右一致性，并进一步用 RAFT 交集删除像素。这把质量检查误当成了几何裁剪：frame 0 Foundation LR 有效率约 `83.25%`，strict Foundation+RAFT 交集约 `52.18%`，所以预览和反投影都会出现大片孔洞，平面与组织自然不完整。

v4 已把“深度值”和“是否高置信”彻底分开：

1. `NNNNNN-depth.npy` 保留 FoundationStereo 的完整时间戳插值视差，只做正分母和 `35–250 mm` 物理范围检查。本数据五帧全部像素都通过，因此没有补洞、最近邻扩散或跨边缘插值。
2. 左右一致性、RAFT 有效性、两模型差异仍然计算，但只写入 `confidence.npz`、`confidence_level.png` 和审计深度，不再把主深度置为 `NaN`。
3. 置信度等级 `1/2/3` 分别表示：Foundation 稠密值；额外通过 Foundation 左右一致性；再额外通过 RAFT 有效且两模型深度差不超过 `3 mm`。等级低表示使用时应降权或重点复查，不表示删除该点。

### 当前产物

当前推荐目录为 `data/super/grasp5_native/depth_v4_foundation_dense_timestamped/`，约 `379 MB`：

| 文件 | 用途 |
|---|---|
| `NNNNNN-depth.npy` | **当前唯一推荐的平面/组织稠密深度输入**，单位米。 |
| `NNNNNN-disparity.npy` | Foundation 时间戳插值后的完整视差。 |
| `NNNNNN-depth_lr_consistent.npy` | 仅供左右一致性审计，含 `NaN`，不得直接当完整组织输入。 |
| `NNNNNN-depth_high_confidence.npy` | Foundation+RAFT strict 交集，只供局部误差检查，含大量空洞。 |
| `NNNNNN-confidence.npz` | dense/LR/RAFT/high-confidence mask、LR error、前后帧 temporal span、模型差异和 1/2/3 等级。 |
| `NNNNNN-comparison.png` | 左图、完整 Foundation 深度、RAFT 对照、模型差异和灰度置信度等级；第二栏才是主深度。 |
| `depth_generation_summary.json` | 模型、环境、时间戳配对、参数、五帧全局/组织/地面统计。 |

### 完整性和几何验证

| 检查 | v4 结果 |
|---|---:|
| frame `0..4` 全图 finite fraction | 每帧均为 `1.0` |
| frame 0 tissue mask | `973869 / 973869` 个像素有深度，`100%` |
| frame 0 ground mask | `915391 / 915391` 个像素有深度，`100%` |
| frame 0 腐蚀后 ground 拟合点 | `882420` 个，和完整 mask 数量一致 |
| frame 0 ground 点面绝对残差 P50/P90/P95/P99 | `0.955 / 2.470 / 2.878 / 3.539 mm` |
| 旧 `depth_v2` ground P95 | 约 `10.48 mm` |

五帧稠密平面相对 frame 0 的法向差约 `0.90/1.08/1.44/1.41 deg`，plane `d` 差约 `-1.78/-0.002/+0.35/+0.39 mm`。这说明稠密化没有退回旧平面的十毫米级噪声。组织和地面可以直接在原 SAM2 mask 内完整反投影；置信度只用于后续稳健拟合时降权和边缘审计。

重新生成：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
XFORMERS_DISABLED=1 /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/generate_super_depth_foundation_timestamped.py --frames 0,1,2,3,4
```

后续候选重建必须显式使用：

```text
data/super/grasp5_native/depth_v4_foundation_dense_timestamped/000000-depth.npy
```

仍然保持原规则：组织使用第 0 帧及原第 0 帧 SAM2 tissue mask；地面使用第 0 帧完整 ground mask 建候选，并用第 1–4 帧检查静态稳定性。本轮未写入 `table_frame.json`、`bodies_v5_table`、`soft_tissue_v1` 或 corrected PSM pose driver，冻结的 `X_table_camera` 未改变。

## 稠密 v4 的 ground plane、ground 与刚体 tissue（2026-07-19，v6 迁移前中间方案）

> **已被 v7 坐标迁移取代：** 本节记录先把新平面放入旧 table 坐标进行审计的 v6 阶段。用户随后决定让该平面定义新的 `z=0`；当前权威坐标和 runtime 见下方 v7 章节。

### 当前结论

已用第 0 帧全稠密 FoundationStereo 深度和原 SAM2 mask 重建三类资产，并把 SUPER demo 切换到这套候选：

1. `ground_plane.json` 是物理碰撞平面。
2. `ground.json` 是地面的可视 Gaussian。
3. `tissue.json` 是一个刚体组织，使用小球作为紧密碰撞代理、Gaussian 作为外观；当前不再加载 PBD 四面体组织。

此前 `soft_tissue_v1`、`soft_tissue_v2_dense` 和 `soft_tissue_v2_tight` 保留为实验记录，但不是当前运行时输入。1 mm PBD 候选的几何拓扑正确，静置时却有分辨率相关速度抖动；用户随后决定先使用刚体，因此不再以软组织门禁阻塞当前场景。

### 重拟合平面为什么没有改变坐标系

新平面先在相机坐标中拟合，再只用已有 `data/super/table_frame.json` 中的 `X_table_camera` 表达到旧 table 坐标。整个过程没有重新估计、覆盖或把平面“转正”为 `z=0`，因此：

- table 原点、x/y/z 轴和所有既有器械位姿都没有改变。
- 新平面在旧 table 坐标中保留为一般平面，而不是强行写成 `[0,0,1,0]`。
- 物理重力/up 采用新平面的法向，但这只是场景物理参数，不是坐标变换。
- 新平面相对旧 table z 轴约倾斜 `11.7419 deg`，平面参数为 `[-0.06261821, 0.19362987, 0.97907427, 0.00031684]`，形式是 `ax+by+cz+d=0`。

坐标保护哈希在 1000 帧门禁前后完全一致：

| 文件 | SHA256 |
|---|---|
| `data/super/table_frame.json` | `2f288cf3c9e65a8d86579e811524bb423e7b38dd063a028c014b715cfbaa2a7e` |
| `data/super/grasp5_offline_demo/cameras.json` | `eed6d8bddef88dea576efd8a83457f562bb05859aae9a76c70222c394f7b6a6c` |

corrected PSM pose driver 及其器械坐标链没有在本阶段重建或覆盖。深度、平面、ground 和 tissue 继续使用 `eg_codex`；器械矫正仍使用 `online_dvrk`。

### 三类资产

正式目录：`data/super/grasp5_native/bodies_v6_dense_frozen_table/`

| 资产 | 当前结果 |
|---|---|
| ground plane | 使用 `875413` 个 ground 拟合点；点面绝对残差 P50/P90/P95/P99 约为 `0.952/2.463/2.871/3.531 mm`。 |
| ground | `5000` 个可视 Gaussian；它不重复承担物理平面碰撞。 |
| rigid tissue | 单一刚体，`4613` 个半径约 `1 mm` 的球形碰撞代理、`2863` 个可视 Gaussian。碰撞球最近邻 P50 约 `2 mm`，直径与邻距约为 `1.0`，即相邻小球基本相切。 |
| 输入覆盖 | frame 0 tissue/ground mask 内的深度均为 `100%` finite，不再因置信度筛选留下大片孔洞。 |
| 统一预览 | `super_bodies_frozen_table.ply`：红色为 tissue 碰撞球中心，橙色为 tissue Gaussian，绿色为 ground Gaussian，蓝色为 ground plane 采样。 |

中间相机坐标候选保存在 `bodies_v6_dense_camera/`，只供审计；demo 只加载 `bodies_v6_dense_frozen_table/`。

### 运行时切换

`examples/embodied_environments/super_embodied/super_embodied.py` 现直接从 v6 冻结 table 目录加载三类资产：

- tissue 通过标准 `Body + add_rigid_body()` 构造，只有一个刚体 body。
- 运行时 `particle_count=0`、soft Gaussian 数量为 `0`，不会进入 PBD、tetra material projector 或 soft Gaussian-to-particle force 路径。
- tissue 的 `2863` 个 Gaussian 重新绑定到该刚体 body，刚体视觉力路径仍可使用。
- PSM 继续由所选时间戳 pose driver 直接驱动；PSM 刚体碰撞仍关闭，不让器械碰撞扰动当前刚体重建。
- ground plane 在所有 child builder 合并后重新写入 Warp model，避免被 builder 默认地面覆盖。

### 1000 帧刚体门禁

新增 `scripts/smoke_super_rigid_scene.py`，报告位于 `bodies_v6_dense_frozen_table/rigid_scene_smoke_report.json`。A800 CUDA 的 1000 帧结果：

| 门禁 | 结果 |
|---|---:|
| 场景 body / shape / 总 Gaussian | `19 / 4646 / 9371` |
| tissue rigid body 数量 | `1` |
| soft particle / soft Gaussian | `0 / 0` |
| 新 ground plane 是否与资产一致 | 通过 |
| tissue 最低最终平面间隙 | 约 `0.0155 mm`，无穿透 |
| 1000 帧 tissue 平移漂移 | 约 `0.0218 mm` |
| 最终组织速度 | `0` |
| table/camera 哈希 | 前后不变 |
| 纯物理步进 | 约 `2.27 ms/frame`，不含初始化与渲染 |
| 最终结果 | 全部门禁通过 |

### 复现命令

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

# 从第 0 帧稠密深度构建相机坐标候选
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/build_super_bodies_from_first_frame.py \
  --depth data/super/grasp5_native/depth_v4_foundation_dense_timestamped/000000-depth.npy \
  --output-dir data/super/grasp5_native/bodies_v6_dense_camera \
  --particle-radius 0.001 --max-particles 5000 \
  --max-tissue-gaussians 3500 --max-ground-gaussians 5000 \
  --voxel-size 0.001 --gaussian-iters 600

# 只使用已有 X_table_camera 表达到冻结 table 坐标，不重拟合坐标系
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/transform_super_scene_to_frozen_table_frame.py

# 导出三类资产的统一预览
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/export_super_bodies_to_ply.py \
  --bodies-dir data/super/grasp5_native/bodies_v6_dense_frozen_table \
  --output data/super/grasp5_native/bodies_v6_dense_frozen_table/super_bodies_frozen_table.ply \
  --plane-resolution 60

# 1000 帧刚体、平面和坐标哈希门禁
PYTHONPATH=src:. /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/smoke_super_rigid_scene.py --device cuda --steps 1000
```

## 新 ground 定义 z=0 的全局坐标迁移（2026-07-19，当前权威 v7）

### 用户决定与迁移边界

用户决定不再把新拟合 ground 作为旧 table 坐标中的倾斜一般平面，而是让它定义新的 table `z=0`。这会改变所有 world/table 坐标数值，因此本阶段采用一个唯一的旧 table → 新 table SE(3)，统一迁移：

- `table_frame.json` 中的相机到 table 变换。
- `cameras.json` 中左右相机的 camera-to-world 位姿。
- ground plane、ground Gaussian 和刚体 tissue。
- 运行时 strict/registered/corrected 等 PSM table pose。

没有重新估计 hand-eye、LND、paper/GUI CAD 注册或 URDF 关节。所有 active PSM driver 的 NPZ 保存的是 `poses_rect_camera_xyz_xyzw`，因此原文件保持不变；运行时只通过新的 `X_table_camera` 自动得到新 table pose。

### 唯一坐标变换

迁移脚本为 `scripts/align_super_world_to_dense_ground_plane.py`。它选择最小旋转把 v6 平面法向转到 `+Z`，不额外绕 Z 改 yaw；原点只沿新 Z 平移，使平面 offset 变为 0：

```text
X_new_table_old_table =
[[ 0.9980187502,  0.0061264785,  0.0626182116, 0.0000000000],
 [ 0.0061264785,  0.9810555240, -0.1936298661, 0.0000000000],
 [-0.0626182116,  0.1936298661,  0.9790742743, 0.0003168443],
 [ 0.0000000000,  0.0000000000,  0.0000000000, 1.0000000000]]
```

- 旋转角：`11.7418937 deg`。
- 平移：新 Z 方向 `+0.316844 mm`。
- 新 `ground_plane.json`：严格 `[0,0,1,0]`。
- 新 ground Gaussian z 范围约 `-9.93e-9 .. +1.18e-8 m`，即只有浮点误差。
- 新坐标仍为右手系，旋转矩阵 determinant=`1`，正交误差约 `1.11e-16`。

### 权威文件、哈希与备份

| 文件 | 迁移前 SHA256 | 迁移后 SHA256 |
|---|---|---|
| `data/super/table_frame.json` | `2f288cf3c9e65a8d86579e811524bb423e7b38dd063a028c014b715cfbaa2a7e` | `6dddc2178cdf816f5dada5febdd528f80e42d52e631076e1f5f4a952297adecf` |
| `data/super/grasp5_offline_demo/cameras.json` | `eed6d8bddef88dea576efd8a83457f562bb05859aae9a76c70222c394f7b6a6c` | `e1e7b7e7e21ca8a9409c88a29409d2e9ad6b783a85b277e71cec0c84340ce4ef` |

迁移前原文件和迁移报告位于：

```text
data/super/coordinate_backups/pre_dense_v7_z0_20260719/
```

当前场景资产位于：

```text
data/super/grasp5_native/bodies_v7_dense_ground_z0/
```

v6 目录保持原样，只作为旧 table 坐标审计输入，不再由 demo 加载。

### 三类资产与运行时

`examples/embodied_environments/super_embodied/super_embodied.py` 已切换到 v7：

| 资产 | v7 结果 |
|---|---|
| ground plane | `[0,0,1,0]`，作为 Warp 物理地面。 |
| ground | `5000` 个 Gaussian，全部位于 z=0。 |
| rigid tissue | 单一刚体，`4613` 个半径约 `1 mm` 的 sphere shape，`2863` 个可视 Gaussian。 |
| soft 路径 | runtime particle=`0`、soft Gaussian=`0`，不加载任何旧 soft NPZ。 |
| PSM | corrected driver 原 NPZ 不变；运行时由新 `X_table_camera` 统一迁移；刚体碰撞仍关闭。 |

统一预览为 `bodies_v7_dense_ground_z0/super_bodies_ground_z0.ply`。

### 等变性验证

迁移不是重新拟合各对象。数值检查证明所有新结果都等于旧结果统一左乘同一个变换：

| 检查 | 最大误差 |
|---|---:|
| tissue collision sphere 世界坐标 | `2.11e-17 m` |
| tissue Gaussian 世界坐标 | `2.13e-17 m` |
| ground Gaussian 世界坐标 | `0` |
| 左右相机 `X_WC` | `0` |
| corrected PSM 全 5458 帧 × 7 link，共 38206 pose | 矩阵元素 `3.33e-16`，平移 `2.00e-16 m` |
| ground/tissue 左右目归一化投影 | `4.44e-16` |

因此 world 数值已改变，但相机画面中的投影、对象相对位置和 corrected PSM 的视觉矫正结果没有改变。

六个 active PSM driver 在迁移前后 SHA256 完全一致，包括 corrected driver：

```text
psm_part_corrected_pose_driver.npz
21df09849b08d6cef1ae47c694e7400b6df5228e0805c5c35ecf3d68e2ef648a
```

GUI offset 回归也通过：`-27 deg` roll 对六个 distal link 保持共同刚体旋转；`+17 deg` jaw opening 仍为左右各约 `8.5 deg`，共同铰点 gap=`0`。

### corrected PSM 的 1000 帧门禁

`scripts/smoke_super_rigid_scene.py --psm-pose-driver corrected` 已通过 A800 CUDA 1000 帧：

| 门禁 | 结果 |
|---|---:|
| runtime ground plane | `[0,0,1,-0]`，与 v7 资产一致 |
| tissue 最低最终间隙 | 约 `0.0155 mm` |
| tissue 1000 帧平移 | 约 `0.0214 mm` |
| 最终组织速度 | `0` |
| table/camera 新哈希 | 门禁前后不变 |
| 纯物理步进 | 约 `2.28 ms/frame` |
| 最终结果 | 全部门禁通过 |

报告：`data/super/grasp5_native/bodies_v7_dense_ground_z0/rigid_scene_smoke_report.json`。

### 复现命令

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

# 首次迁移前先预演；不会写文件
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/align_super_world_to_dense_ground_plane.py

# 首次正式迁移；会先备份旧 table_frame 和 cameras
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/align_super_world_to_dense_ground_plane.py --apply

# 当前权威坐标下的 corrected PSM 刚体门禁
PYTHONPATH=src:. /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/smoke_super_rigid_scene.py \
  --device cuda --steps 1000 --psm-pose-driver corrected
```

迁移脚本会拒绝非空 v7 输出目录和与 v6 不匹配的 canonical table frame，防止二次迁移。

## PSM 双目三维矫正仓库审计（2026-07-19）

- 当前主流程 `track_super_psm_paper_exact.py` 只读取左目视频。`PoseEstimationProblem` 虽然输出 6DoF 相机位姿，但损失仍是单目二维 silhouette、二维距离图、面积和投影后的二维夹爪点；`use_pts_loss` 不是三维点或深度损失。
- `BiManualPoseEstimationProblem` 中的 left/right 表示左右机械臂，不表示左右目相机；`share_depth_buffer` 只是两条机械臂渲染时共享 z-buffer。
- 仓库旧 `ParticleFilter` 路径会同时读取左右视频，并把机械臂彩色标记点分别投影到两台相机，用双目二维重投影联合约束 6DoF 位姿。这条路径依赖可靠的 painted-marker 检测、同帧左右视频和旧数据格式，不可直接用于当前无标记的 SAM2/CAD 流程。
- 当前 `diagnose_super_psm_stereo_depth.py` 已做时间戳插值双目深度与 GUI CAD 表面比较，但它是只读 RAFT-Stereo 诊断，不优化也不写回 pose driver。现有 `-5.7106 mm` 仍只是 4 个可靠 body 帧得到的 camera-z 候选值。
- 若实施真正稠密三维矫正，应使用时间戳同步 FoundationStereo 作为器械深度主观测、RAFT-Stereo 作为一致性检查，在 rectified 左目相机坐标中对观测深度与 CAD 渲染 z-buffer 建立稳健损失，同时保留左右目 silhouette、运动学和时间平滑约束。
- 三维矫正仍输出 rectified camera-frame pose；右目只通过冻结的双目标定参与投影，最终才用当前 v7 `X_table_camera` 转到 world。不得重估 table frame、hand-eye、LND、CAD registration 或 URDF 铰链关系。
- 上述仓库审计阶段只检查了旧流程；后续实现结果记录在下一节。整个实施过程仍未修改相机标定，且候选写为独立 NPZ，没有覆盖原 active PSM driver。

### 用户确认后的方案升级

用户明确要求不能把固定 `-5.7106 mm` 平移当作最终器械深度。方案改为：固定值只作为全局搜索初值 `D0`，随后在双目可靠关键帧上优化整件器械的有界 6DoF 残差，并用 FoundationStereo 稠密深度与 CAD 前表面采样的稳健损失直接约束 camera-z 和器械倾斜。wrist 和 jaw 不从噪声双目中重新估计；每个时刻都把同一个刚体修正左乘到 7 个源 link，从而严格保留已经验证的 LND/URDF 相对运动、共用夹爪铰点和开合角。低置信帧不独立拟合，而是按真实时间戳插值。

## PSM 时间戳同步稠密双目三维矫正（2026-07-19，v2 已拒绝）

> **拒绝原因：** 该实验没有沿用下文已经验证较好的 `online_dvrk_tracking -> corrected` 主流程，而是在最终 `corrected` driver 上另加公共 `D0 + 逐关键帧 6DoF`。Foundation/CAD 表面残差错误地把约 `-5.85 mm` 解释成全序列相机 z 初值，实际 GUI 验收显示器械明显过近、整体投影比原 `corrected` 更差。数值深度残差和运动学门禁不能替代真实图像投影验收。`stereo_3d` 已从运行参数撤下，相关 NPZ 和图片只保留作失败审计，任何正式回放继续使用 `corrected`。

### 纠正后的实施边界

- 恢复并冻结 `online_dvrk_tracking` 论文视频标定、LND 稳定主干、三部件 SAM2、有界 9 参数矫正和 URDF 重建组成的原 `corrected` 链路。
- 不再使用任何固定 `-5.7/-5.85 mm` 全局平移，也不把稀疏 Gaussian 最近邻深度当成绝对器械表面。
- 后续严格按用户指定顺序执行：时间戳 LND 运动学先生成 `T_kin`；Foundation/RAFT 只微调 camera-z 得到 `T_depth`；再把 `T_depth` 作为 `online_dvrk_tracking` 视频标定/跟踪的初始化，最后执行三部件矫正和 URDF 重建。原 `corrected` 用作必须不劣化的 A/B 基线。
- README 的 stock `--use_bo_initializer` 不能原样使用，因为它会重新大范围搜索 z、姿态和末端关节并覆盖深度先验。视觉阶段保留其 SAM2、可微 CAD 渲染和 CMA-ES/Gradient 方法，但 z 必须冻结或限制在 `T_depth` 周围最多 `+/-0.3-0.5 mm`，并保留 encoder/对称 jaw 约束。
- camera `+z` 在当前 table/world 中为 `[-0.064771,+0.698724,-0.712453]`；每 `+1 mm` camera-z 同时产生约 `0.702 mm` 平面内移动和 `-0.712 mm` world-z 变化。深度修正必须在相机坐标中做，不能当成 world-z 或 insertion。
- 深度阶段不再输出一个固定平移值。每个可靠关键帧从 `T_kin` 开始做 `2-5` 轮 CAD z-buffer 重渲染和小步更新，单步 camera-z 限制 `0.2-0.5 mm`；每轮都重新计算残差，满足 `|delta_z|<0.1 mm`、改善 `<0.05 mm` 且左右投影不劣化后停止。随后才进入受 z 先验约束的 online_dvrk 视觉矫正。
- 必须先在人工关键帧生成左右目原 `corrected`/候选 A/B；只要投影看起来更差或轮廓误差增加，即使深度数字下降也拒绝。
- 原 `psm_part_corrected_pose_driver.npz` SHA256 仍为 `21df09849b08d6cef1ae47c694e7400b6df5228e0805c5c35ecf3d68e2ef648a`，没有被覆盖。

### 已执行方法

- 主深度：FoundationStereo；RAFT-Stereo 只做左右一致性、模型一致性和独立残差复核。20 个请求关键帧的稠密深度保存在 `data/super/psm_tracking/psm_stereo_3d_corrected_v1/depth_observations/`。
- 时间同步：左目时刻使用前后右目视差做插值。frame `1440` 没有右目时间包围，禁止作为优化锚点；frame `1200` 只有 `33` 个初始 body 样本，低于正式门槛 `100`，同样禁止作为锚点。
- 全局初值：18 个可靠帧的 Foundation body 深度得到 `D0=-5.847134 mm`（rectified 左相机 `z` 方向）。它只负责把优化带到正确深度附近，不直接当作全序列最终平移。
- 逐关键帧：在 wrist 锚点周围优化公共 6DoF，边界为每轴旋转 `+-4 deg`、相机 x/y 每轴 `+-3 mm`、围绕 `D0` 的 z 残差 `+-3.5 mm`。损失包括 body/jaw 深度、分割轮廓越界惩罚和弱先验。
- 16 个关键帧通过全部质量门；frame `240`、`1120` 因优化达到最大计算次数而拒绝。所有其余帧使用真实左目时间戳上的 PCHIP 连续插值，锚点范围外保持最近可靠修正不变。
- 被拒绝的实验文件为 `data/super/psm_tracking/psm_stereo_3d_corrected_v2/psm_stereo_3d_corrected_pose_driver_candidate.npz`，SHA256=`fc15b0944c7d14cd7653210a11c0a5c2a49a66f3090d16dded6fe4b05c0b36f1`。原 `corrected` 驱动没有覆盖；`stereo_3d` 运行选项已经撤销。

### 深度与运动结果

| 检查 | 结果 |
|---|---:|
| 16 个接受锚点的 Foundation body 绝对中值残差，中位数 | `0.539 -> 0.168 mm` |
| Foundation body 绝对中值残差，95 分位 | `1.204 -> 0.429 mm` |
| RAFT 独立复核绝对中值残差，中位数 | `0.200 mm` |
| 深度绝对中值改善的锚点 | `13/16`；其余 3 帧仍小于 `0.666 mm` |
| 附加修正最大线速度 / 角速度 | `1.449 mm/s` / `6.577 deg/s` |
| 附加修正最大线加速度 / 角加速度 | `9.574 mm/s^2` / `36.908 deg/s^2` |
| 时间运动硬门 | 全部通过；上限分别为 `5 mm/s`、`10 deg/s`、`100 mm/s^2`、`300 deg/s^2` |

这里的时间硬门是必需的。左目实际相邻时间间隔最短约 `2 ms`，按帧号做普通高斯平滑会产生虚假高速；v2 已改为直接在实测时间戳上做不过冲的 PCHIP 插值。

### 夹爪、坐标和运行时门禁

| 检查 | v2 结果 |
|---|---:|
| 7 个 link 相对源驱动的最大相对变换误差 | `3.45e-8 m` / `6.10e-6 deg` |
| 两片 jaw 共用转轴最大距离 | `3.12e-17 m`（内部硬门）；独立 1441 帧验证为 `0 mm` |
| 稳定闭合区间 | frame `548..1246`，共 `699` 帧 |
| 闭合 jaw 表面距离中位数 | `0.0148 mm` |
| 闭合 jaw 铰轴平行误差中位数 / 最大值 | `4.09e-6 deg` / `0.00160 deg` |
| v7 table frame / cameras 哈希 | `6dddc2...adecf` / `e1e7b...ce4ef`，前后不变 |
| `stereo_3d` GPU 场景冒烟 | 100 步通过，报告 `runtime_smoke_100.json`，全部 gate=true |

候选没有给左右 jaw 分别增加位移，也没有重新拟合 jaw 角。它只改变整件器械在相机中的公共位姿，因此不会重新引入此前的左右偏移、转轴分离或闭合不了的问题。

### 对比图和报告

- 全关键帧对照：`data/super/psm_tracking/psm_stereo_3d_corrected_v2/comparisons/contact_sheet.png`。
- 单帧六栏对照：同目录下 `frameNNNNNN_stereo_3d_comparison.png`，包含左右目旧/新投影及 Foundation 修正前后残差。
- 夹爪旧独立 link 与当前共用转轴对照：`data/super/psm_tracking/psm_stereo_3d_corrected_v2/jaw_validation/jaw_shared_pivot_comparison.png`。
- 完整逐帧指标、输入哈希、边界和门禁：`data/super/psm_tracking/psm_stereo_3d_corrected_v2/report.json`。

### 失败审计说明

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

# 仅在复查失败原因时生成 v2；不要作为正式 driver
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/refine_super_psm_stereo_3d.py

# 正式回放恢复到此前已验证的 online_dvrk/LND/三部件链路
bash scripts/run_demo_browser_12.sh \
  --psm-pose-driver corrected \
  --psm-visual-mode tip \
  --psm-roll-offset-deg -27
```

## PSM 多轮深度后视觉矫正关键帧版（2026-07-19，v1）

本轮按用户确认的顺序实现，不再使用固定 `-5.8 mm`：

```text
时间戳匹配的 registered LND 运动学
  -> 每个关键帧独立、多轮、小步 camera-z 深度微调
  -> 以该深度状态初始化 online_dvrk_tracking 的可微 CAD 视觉优化
  -> 与原 corrected 做逐帧 A/B 门禁
  -> 用原固定 CAD-to-GUI registration 和字面 URDF 远端链重建夹爪
```

实现脚本为 `scripts/refine_super_psm_depth_then_visual_keyframes.py`。它使用 FoundationStereo 作为主深度，RAFT-Stereo 只做独立一致性检查；深度比较来自真实 paper CAD 三角面的透视 z-buffer，不再使用稀疏 Gaussian 最近邻。camera-z 每轮最多 `0.3 mm`、最多 4 轮、累计最多 `1.2 mm`。每轮都重新渲染；Foundation 残差不下降、RAFT 变差、body Dice 相对本帧初始值下降超过 `0.005` 或可靠样本不足时立即停止。

视觉阶段复用 `online_dvrk_tracking` 的 CtRNet/nvdiffrast 可微 CAD 渲染和 Gradient 优化路径。为保留深度结果，平移边界改成各向异性：camera x/y 各 `+/-3 mm`，用于恢复原方法已经较好的二维轮廓；camera-z 只允许在深度结果附近 `+/-0.4 mm`。旋转、wrist 和共同 jaw 残差边界分别为 `8 deg`、`12 deg`、`25 deg`。jaw 左右片不允许独立平移，最终仍由 URDF 父链和 `+/-jaw/2` 重建。

### 关键帧结果

正式评估帧为 `0,141,160,320,480,800,1120,1280,1360`。深度阶段得到的 camera-z 累计微调分别为：

```text
frame 0:       0.0 mm   （第一步使轮廓超过允许降幅，拒绝）
frame 141:    -0.3 mm
frame 160:    -0.3 mm
frame 320:    -1.2 mm
frame 480:    -1.2 mm
frame 800:    -0.3 mm
frame 1120:   -0.6 mm
frame 1280:   -1.2 mm
frame 1360:   -0.6 mm
```

这些值是各帧多轮重渲染的结果，不是一个套用全序列的平移常数。最终 A/B 接受新结果的帧为 `141,160,480,1120,1280`。其余帧回退原 corrected：frame 320 的 body 轮廓门禁未通过；frame 0/800 是周期性 CAD 合成身份锚而不是真实图像证据；frame 1360 的右 jaw 置信度只有约 `0.10`。

5 个新接受帧的指标中位数：

| 指标 | 原 corrected | 新接受结果 |
|---|---:|---:|
| body Dice | `0.757` | `0.791` |
| 左右 jaw 尖端平均误差 | `1.59 px` | `1.03 px` |
| Foundation body 深度绝对中值残差 | `3.30 mm` | `2.45 mm` |
| RAFT 独立 body 深度绝对中值残差 | `3.29 mm` | `2.42 mm` |

URDF 夹爪门禁通过：9 个最终选择状态的左右 jaw 共用铰点最大 gap=`0 m`，铰轴最大角误差约 `3.31e-6 deg`。table/cameras/active corrected driver 的哈希分别仍为 `6dddc2...adecf`、`e1e7b...ce4ef`、`21df09...648a`；脚本现在会在运行前硬检查这三个哈希，不匹配就拒绝运行。

### 输出与当前边界

- 总对比图：`data/super/psm_tracking/psm_depth_then_visual_v1/contact_sheet.png`。
- 单帧两行六栏图：同目录 `frameNNNNNN_comparison.png`；上行包含观测、LND、深度后、online_dvrk raw、旧 corrected、最终接受结果，下行包含四种 Foundation-CAD 深度残差和左右目旧/新投影。
- 数值报告：`data/super/psm_tracking/psm_depth_then_visual_v1/report.json`。
- 独立状态：`keyframe_states.npz`；URDF 重建结果：`keyframe_visual_poses_urdf.npz`。

这是供人工查看的关键帧版，不做时间插值、不生成 1441 帧状态、不生成 5458 状态 runtime driver，也没有加入 GUI 选项。因此当前正式回放仍必须使用 `corrected`；全序列时间速度/加速度和连续闭合检查留到关键帧图片人工认可之后。

复现命令：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/refine_super_psm_depth_then_visual_keyframes.py \
  --output-dir data/super/psm_tracking/psm_depth_then_visual_v1
```

### 双接口回放补充

按用户要求保留两个并列接口，不覆盖旧结果：

- `corrected`：深度优化之前的原器械位姿，原 driver SHA256 仍为 `21df0984...648a`。
- `depth_then_visual`：把 5 个通过门禁的关键帧相对 `corrected` 的残差按真实时间戳 PCHIP 传播；4 个拒绝关键帧和序列首尾是严格零残差锚点。独立 driver 为 `psm_depth_then_visual_pose_driver_candidate.npz`，SHA256=`72bf8aaa...b07f`。

新 driver 共 5458 状态。原 corrected 的视频 pose 和 runtime driver 重建误差均为 0；新修正最大线/角速度为 `0.662 mm/s` / `2.689 deg/s`，最大线/角加速度为 `2.26 mm/s^2` / `15.43 deg/s^2`。全 1441 帧 jaw 共用铰点 gap=`0`，铰轴最大误差约 `5.79e-6 deg`。

`depth_then_visual` 已通过 100 步 A800 刚体场景加载冒烟，全部 gate=true；报告为 `data/super/psm_tracking/psm_depth_then_visual_full_v1/runtime_smoke_100.json`。两个 GUI 参数现在可以直接 A/B 回放。

## 刚体组织改为 tblock 式挤密粒子（2026-07-19，v8）

### 为什么 v7 看起来有很大的粒子缝隙

v7 的组织虽然已经是一个刚体，但它的碰撞球按简单立方网格排列：球半径为 `1.0 mm`，相邻球心距离几乎严格等于 `2.0 mm`。也就是说，相邻球只在一个点上相切；渲染出球的轮廓后，四个球之间必然留下明显的菱形空隙。论文 demo 的 tblock 不是这种排列，它经过粒子位置优化，球心错层且有轻微重叠，所以看起来是不规则地挤在一起。

### v8 的修改

- 新资产：`data/super/grasp5_native/bodies_v8_packed_rigid_tissue/`；v7 保留不覆盖，可随时对照。
- 组织仍是**一个刚体**，没有启用 PBD、四面体或软组织求解。
- 球半径仍为 `1.0 mm`，球心改为 HCP/AB 错层紧密排列，再加入固定随机种子的微小扰动：平面内最多 `0.08 mm`，高度最多 `0.04 mm`。
- 名义球心间距改为 `1.85 mm`。实际最近邻距离中位数为 `1.768 mm`，实际重叠中位数为 `0.232 mm`；`100%` 的粒子至少与一个邻居轻微重叠。
- 碰撞球数量由 `4613` 增至 `6118`。组织 Gaussian 仍是原来的 `2863` 个，外观和相机投影不因本次重排改变。
- 为避免球数增加导致刚体质量增加，运行时球密度由 `1000` 调为 `754.0046 kg/m^3`。实测 v8 质量 `19.3228927 g`，v7 目标质量 `19.3228920 g`。

### 坐标不变保证

本次只更换 `tissue.json` 中的刚体局部碰撞球心。以下内容原样复制或由硬门禁保护：

- `tissue.X_WB` 完全不变；组织 Gaussian 的 means、quats、scales、颜色完全不变。
- `ground.json` 和 `ground_plane.json` 完全不变，权威平面仍是 world/table `z=0`。
- 最底层球心保持 `z=1.0 mm`，半径为 `1.0 mm`，因此底面仍与 `z=0` 相切。
- `table_frame.json` SHA256 仍为 `6dddc217...adecf`。
- `cameras.json` SHA256 仍为 `e1e7b7e...ce4ef`。
- PSM old/new 两套 pose driver 均未修改，器械运动、夹爪铰链和视觉矫正不受本次组织重排影响。

由于 Gaussian 和 `X_WB` 均未改变，真实相机画面里的组织外观与 v7 相同；变化只体现在物理碰撞球显示和碰撞体的内部填充。v8 相对 v7 的球心包围盒边界变化不超过约 `0.56 mm`，这是边缘错层排布造成的，不是整体坐标变换。

### 验证结果

`scripts/smoke_super_rigid_scene.py` 已在 A800 上运行 1000 步，全部 gate 通过：

| 检查 | v8 结果 |
|---|---:|
| 组织类型 | 1 个刚体，0 个软体粒子 |
| collision sphere | `6118` |
| 最低最终离地间隙 | `0.0152 mm` |
| 1000 步后组织平移 | `0.0230 mm` |
| 最终速度 | `0` |
| 运行时质量 | `19.3228927 g`，与 v7 一致 |
| 纯物理步进 | `2.388 ms/frame` |
| table/camera 哈希 | 前后完全不变 |

报告：`data/super/grasp5_native/bodies_v8_packed_rigid_tissue/rigid_scene_smoke_report.json`。

精确粒子俯视对比图：`data/super/grasp5_native/bodies_v8_packed_rigid_tissue/particle_packing_comparison.png`。左侧是 v7 规则相切网格，右侧是当前 v8 的错层挤密布局。

### 复现命令

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

# 从冻结的 v7 坐标资产重建一份独立核对副本；不覆盖当前 v8
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/repack_super_rigid_tissue.py \
  --output-dir /tmp/super_bodies_v8_rebuild

# 验证 z=0、质量、刚体稳定性和坐标哈希
PYTHONPATH=.:src /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/smoke_super_rigid_scene.py \
  --device cuda --steps 1000 --psm-pose-driver corrected
```

## 2026-07-24 交接入口

- 当前可运行基线仍为 `bodies_v9_dense_0p5mm_rigid_tissue`，没有被计划中的
  v10 候选替换。
- 下一阶段完整方案见本文的
  `多帧双目刚体组织重建计划（2026-07-24，阶段 A/B mask 已完成，阶段 C 待实施）`。
- 当前 ThinLinc/VirtualGL 启动命令和 `corrected`、`depth_then_visual` 两个位姿
  接口记录在该计划末尾的 `当前 v9 版本运行命令（继续保留）`。
- 保护哈希核对、阶段 A 候选筛选和阶段 B 双目独立 tissue/tool mask 已完成；
  下一步只对阶段 B 最终选择的 6 对双目帧生成阶段 C 的左右参考深度和置信度，
  不创建或切换 v10 scene。

## 多帧双目刚体组织重建计划（2026-07-24，阶段 A/B mask 已完成，阶段 C 待实施）

### 当前状态和实施边界

本节记录下一阶段计划和阶段 A/B 的执行结果；尚未生成多帧双目组织，也没有
替换当前运行资产。
当前默认场景仍使用：

```text
data/super/grasp5_native/bodies_v9_dense_0p5mm_rigid_tissue/
```

当前组织存在两个已确认的限制：

1. 几何只由第 0 帧左目 mask 和左目参考深度反投影。第一帧被器械遮挡、没有
   建出粒子或 Gaussian 的组织区域，后续视觉力不能凭空补出来。
2. v4 深度虽然由时间戳同步后的左右目共同估计，但输出坐标仍对齐到左目像素；
   SAM2 tissue mask 和 Gaussian 外观训练也只有左目。因此现在不是左右目共同
   构建的组织。

下一阶段只重建组织候选，必须遵守以下硬约束：

- 不重新拟合 ground plane，不改变当前 table/world 定义。
- `ground_plane.json` 继续严格为 `[0,0,1,0]`。
- 不修改 `table_frame.json`、`cameras.json`、双目标定、hand-eye、LND、URDF
  或任何 PSM pose driver。
- ground 和 ground plane 从 v9 原样复制，不随组织重建重新生成。
- 新结果只写入
  `data/super/grasp5_native/bodies_v10_multiview_rigid_tissue_candidate/`，
  未通过全部门禁前不修改 runtime 的 v9 路径。
- 新组织仍是一个刚体，继续使用 `0.5 mm` 半径、约 `0.82 mm` 节距的 HCP/AB
  挤密碰撞球，不启用 PBD、tetra 或软组织求解。
- 尽量原样保留 v9 的 `tissue.X_WB`。融合点先在当前 table 坐标中确定，再用
  v9 `X_WB` 的逆变换写入 body 局部坐标，避免重算组织刚体参考系。
- 初次 A/B 对照继续保持 v9 总质量约 `19.3229 g`，通过调整球密度隔离几何变化
  与质量变化；惯量按新的粒子分布重新累计。

当前坐标和运行资产保护哈希：

| 文件 | 当前 SHA256 |
|---|---|
| `data/super/table_frame.json` | `6dddc2178cdf816f5dada5febdd528f80e42d52e631076e1f5f4a952297adecf` |
| `data/super/grasp5_offline_demo/cameras.json` | `e1e7b7e7e21ca8a9409c88a29409d2e9ad6b783a85b277e71cec0c84340ce4ef` |
| v9 `tissue.json` | `21974bbb10fa4d756ba615f93b74b460952ca394fa8b679aab3cfc265785d872` |
| v9 `ground.json` | `8e769cd7fd92d2b5eab2f7ca306986d94206fcdf9726b02b3fc76030bc8f8ecb` |
| v9 `ground_plane.json` | `1a4674957ce063a47f1dcb16a44781b3df77ee4af711a6c105e4e5b8de0c5d96` |
| 原 `corrected` driver | `21df09849b08d6cef1ae47c694e7400b6df5228e0805c5c35ecf3d68e2ef648a` |
| `depth_then_visual` candidate driver | `72bf8aaade1e935445c6908e0519a8c63c9d202f31c21710e598cef91938b07f` |

### 阶段 A：选择可用于重建的多帧

不能把 1441 帧直接平均，因为真实组织在接触后可能已经移动或形变。先从组织
尚未被器械推动的时间段筛选约 `20-30` 个候选帧，再用可见覆盖增量选择约
`6-12` 个关键时间点：

1. 器械遮挡位置明显变化，新增可见组织区域。
2. 组织在 table 坐标中的表面和轮廓保持静止。
3. 图像没有严重运动模糊、烟雾、饱和高光或深度跳变。
4. 每加入一个关键帧，左右目联合覆盖率必须有可测量的增加。

帧选择采用按覆盖增量的 greedy set-cover，不按固定帧号随意抽样。若某一区域
在所有帧和左右目中始终不可见，它只能标记为 `inferred` 几何补全，不能记为
双目实测。

#### 阶段 A 实施结果（2026-07-24）

新增 `scripts/select_super_tissue_multiview_frames.py`。脚本在创建输出目录前硬
检查 table、camera、v9 tissue/ground/ground-plane 和两个 active PSM driver
的保护哈希；本轮 7 个哈希全部与本计划记录一致，ground plane 仍严格为
`[0,0,1,0]`。没有创建 v10 scene candidate，也没有修改 runtime 的 v9 路径。

原始 PNG 实际为左目 `1441` 张、右目 `1443` 张；冻结的离线回放 metadata 是
早期 finalize 时按共同帧截成的 `1441/1441`。本轮从原始 bag 只读恢复了
`1441/1443` 条真实消息时间戳，已有 metadata 覆盖的前 `1441/1441` 条与 bag
时间戳最大绝对误差均为 `0 s`，且没有回写
`calib_rectified.json`、`stereo_left.json` 或 `stereo_right.json`。恢复后的
float64 时间戳 SHA256 为：

- left：`5a6f7d620b75476cd62c0a9f63270846dc664b9cd2efff6c973c6f6cc5b14d93`
- right：`5804853d35482943961852d27d6c9052bbc891b16d92dea6d320067e6c7e1321`

静止区间没有用任意帧号截断，而是由 dataset jaw command 决定：初始 jaw
中值为 `0.9999924 rad`，首次下降超过 `0.05 rad` 是左目 frame `544`
（`18.182519832 s`）；再向前留 `1.5 s` 安全余量，候选只允许来自 frame
`0..498`（末时刻 `16.658915154 s`）。该区间每 3 帧检查一次，共检查
`167` 帧；左右目 Laplacian 清晰度、饱和比例、frame-0 非器械区灰度相关性门禁
均通过。最终 25 个候选左目帧为：

```text
0, 27, 57, 69, 96, 105, 126, 141, 168, 180, 216, 234, 243,
261, 291, 303, 321, 339, 360, 381, 408, 420, 444, 477, 483
```

候选对应的右目帧全部按真实时间戳最近邻查询，不按相同帧号配对；本段时间内
对应为 left frame `n` 到 right frame `n+1`。候选质量分数范围
`0.7102..0.9864`，frame-0 灰度相关性范围 `0.9659..1.0`，左/右 Laplacian
variance 最低分别为 `98.79/174.08`，没有发现严重模糊或饱和帧。

阶段 A 还生成了一组供阶段 B 标注使用的 provisional keyframes。因为独立右目
tissue/tool mask 尚不存在，这一步明确只做左目 proxy set-cover：覆盖域固定为
人工核验的 frame-0 tissue support，再减去每帧图像器械 mask 的 `30 px` 膨胀；
逐帧 SAM2 tissue mask 只用于质量审计，不能把其长期边界漂移算成新增覆盖。
greedy 选择顺序和增量为：

| 顺序 | left/right frame | 单步新增覆盖 | 累计左目 proxy 覆盖 |
|---:|---:|---:|---:|
| 0 | `27 / 28` | `98.11721%` | `98.11721%` |
| 1 | `444 / 445` | `1.47216%` | `99.58937%` |
| 2 | `141 / 142` | `0.20635%` | `99.79572%` |
| 3 | `0 / 1` | `0.11910%` | `99.91481%` |
| 4 | `57 / 58` | `0.04301%` | `99.95782%` |
| 5 | `180 / 181` | `0.02316%` | `99.98098%` |

首轮内部试跑曾直接把逐帧 SAM2 mask 用作 set-cover domain，导致 mask 缓慢收缩
被错误计为新增覆盖，并选中几乎重复的 frame `0/3`。该结果已拒绝并覆盖；
当前脚本固定 frame-0 support 后重新运行，预览中 6 个选择均对应可见的器械
遮挡变化。右目 proxy coverage 和 combined stereo coverage 仍有意记录为
`null`；必须完成阶段 B 的独立右目 mask 后重新做最终双目 set-cover，不能把
上述 `99.98098%` 写成双目覆盖率。

输入哈希除前述保护资产外：

| 输入 | SHA256 |
|---|---|
| `stereo_left.json` | `4ab90c0babf1f38f0528de75e0a4bfd301e8cc2c2ca88134faa941e1c9ce1187` |
| `stereo_right.json` | `ca30d70421807c26eae5db11199cfe9225af73bc2472e78d6e8e190ea042d80a` |
| `robots.json` | `d8724d7fae001d5e9dece3a3a8e950ba026e7d064f9215f763125f38c9b75012` |
| 左目 tissue masks | `445b0130691c61159c229d1c2a77d8545b35e14f7369a0128b2cbe1353096b1e` |
| 三部件 tool masks | `4bdfa684e4451ea8b85b01d1ab49144f0aaf3a75cd65f605a878ba1a9820724d` |

输出位于 `data/super/grasp5_native/tissue_multiview_v1/`：

| 输出 | 作用 / SHA256 |
|---|---|
| `observations.json` | 全部输入、门禁、候选指标、真实时间戳配对和下一步限制；`697d1df52cfc8e28861db26270bd10c4b06aea57bf00f16abf757efd287de088` |
| `timestamps_left_native.npy` / `timestamps_right_native.npy` | 从 bag 恢复的完整 native 时间戳；其数组内容哈希见上文 |
| `coverage/left_proxy_candidate_visibility.npz` | 25 帧左目 proxy 可见性与 set-cover 顺序；`1a5a7e8be292439b42b886cfc7daaa8ebadfc4362904835ce1c905daa975db5f` |
| `previews/phase_a_candidates.png` | 25 个候选的左右真实时间戳配对总览；`8838d74f80b134c595555fb2f0be45633619f49dff35355460457c68a0cfe4d1` |
| `previews/phase_a_provisional_keyframes.png` | 6 个 provisional set-cover 关键帧；`4cd45452887d420191aa4d80c9fef0283ce81bf71fbc04520e35e3bd64005abc` |

复现命令：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/select_super_tissue_multiview_frames.py
```

若只重建阶段 A 自己的 manifest/coverage/preview，可加 `--overwrite`；脚本会复用
已经按 bag identity 和数组哈希验证的 native timestamp cache。上述候选已经
在阶段 B 中完成左右目独立 tissue/tool mask 和最终 stereo set-cover；阶段 A
的 6 个 provisional keyframes 已被阶段 B 的最终 6 对选择替代。

### 阶段 B：左右目分别生成可见组织 mask

现有 `visual_force_masks_v1` 可作为左目候选，但不能直接作为多帧重建真值。
需要把 `scripts/propagate_super_tissue_masks.py` 泛化为左右相机参数化版本：

1. 左目独立传播 tissue mask。
2. 右目用右目首帧种子独立传播。左目深度重投影只能生成右目初值，必须检查
   右目独有边界和 disocclusion，不能简单平移左目 mask。
3. 左右目各自生成器械遮挡 mask。优先使用图像器械分割与 corrected CAD
   投影轮廓的并集，再进行保守膨胀。
4. 每帧最终用于重建的可见 mask 为：

```text
visible_tissue
  = SAM2 tissue
  & valid dense depth
  & acceptable confidence
  & ~dilated_tool_mask
  & ~highlight_or_blur_rejection
```

SAM2 mask 表示图像中的语义区域，不等于模型覆盖区域。后续视觉力还必须额外
与当前组织渲染覆盖 mask 相交。

#### 阶段 B 实施结果（2026-07-24）

本阶段新增 `scripts/build_super_tissue_right_seed.py` 和
`scripts/build_super_multiview_candidate_masks.py`，并把已有
`scripts/propagate_super_tissue_masks.py` 泛化为 `left/right` 相机参数化版本。
所有输出仍只写入
`data/super/grasp5_native/tissue_multiview_v1/`；没有创建 v10 scene candidate，
没有切换 runtime。

右目组织没有简单复制左目 mask。先用左目 frame 0 的稠密深度和 rectified
baseline 只生成投影初值，再在真实右目 frame 0 上用 SAM2.1-large、24 个分散
组织正点、corrected-CAD 器械负点和外部负点独立细化。首轮内部诊断的投影
recall 只有 `88.24%`，漏掉左上组织叶片，已拒绝且未用于传播；增加分散正点并
收紧 CAD 半径后，最终选择的右目 mask 面积为 `943837 px`，相对投影
IoU=`0.954652`、recall=`0.966266`，CAD core overlap=`1.4535%`，正/负提示
违反均为 0。`000000-right-seed-overlay.png` 已人工检查并在报告中记录
`visual_review_accepted=true`。

接受后的右目种子独立传播到术前 `0..498` 共 499 帧。frame 0 预测与人工接受
种子的 IoU=`0.996077`；相对种子面积比 min/p05/median/p95/max 为
`0.91809/0.91868/0.95630/1.00174/1.00344`，相邻帧 IoU
min/p05/median=`0.99037/0.99425/0.99663`。全部帧返回、首帧一致、无空 mask、
统计量 finite 四项门禁均通过。

25 对候选分别生成以下 mask：

1. 左目 tissue 使用已有独立左目 SAM2 传播，右目 tissue 使用上述独立右目传播。
2. 左目器械使用三部件图像分割；右目器械在每个真实右目图像上使用
   SurgicalSAM2-small，并由 corrected-CAD 主可见分量提供提示。
3. 两目最终器械遮挡均取“图像分割与 corrected-CAD 投影”的并集，再在全分辨率
   膨胀 `30 px`。CAD 后期存在一些位于暗背景的宽松离散轮廓；总预览逐格核验
   后确认它们没有漏掉真实器械，代价是保守少取像素。
4. 高光判定与当前 visual-force 权重一致：归一化最大通道 `>=0.90` 且通道差
   `<=0.18`。25 对 tissue 内共剔除左/右 `8614/18337` 个高光像素。
5. 真实时间戳配对最大绝对差为 `15.998 ms`，全部处于 `20 ms` 的单帧周期
   门禁内；仍使用各自真实时间戳，没有把相同帧号当成同步。

每一目继续固定使用人工核验的 frame-0 tissue support 做 set-cover domain，只
让逐帧器械遮挡和高光改变可见性，避免把 SAM2 边界缓慢漂移误算成新增组织。
左右 coverage 各自归一化后严格按 `50%/50%` 计算联合增益。最终 greedy 选择：

| 顺序 | left/right frame | 等权单步新增覆盖 | 等权累计覆盖 |
|---:|---:|---:|---:|
| 0 | `0 / 1` | `98.22770%` | `98.22770%` |
| 1 | `381 / 382` | `1.64841%` | `99.87611%` |
| 2 | `57 / 58` | `0.07980%` | `99.95592%` |
| 3 | `141 / 142` | `0.02275%` | `99.97866%` |
| 4 | `126 / 127` | `0.01313%` | `99.99179%` |
| 5 | `420 / 421` | `0.00341%` | `99.99519%` |

最终左目 proxy coverage=`99.991674%`、右目 proxy coverage=`99.998713%`、
等权 stereo proxy coverage=`99.995194%`。时间顺序为
`0/1, 57/58, 126/127, 141/142, 381/382, 420/421`。这三项明确是
**Stage-B image-space proxy coverage**，不是最终可用于融合的重建覆盖：
右目参考深度和逐目 stereo confidence 尚未生成，因此报告中
`valid_dense_depth_applied=false`、`stereo_confidence_applied=false`。
阶段 C 必须再计算：

```text
final reconstruction visibility
  = stage_b_visibility_proxy
  & valid_dense_depth
  & acceptable_stereo_confidence
```

每个方向输出 25 张 tissue、image-tool、CAD-tool、dilated-tool 和
`stage_b_visibility_proxy`，共 10 组、每组 25 张。主要审计文件：

| 输出 | SHA256 |
|---|---|
| `masks_right/seed/report.json` | `82788b79afb04b57c743765d8df496734ed41654aef4986178b869dcfb864776` |
| `masks_right/propagated/report.json` | `60e6efa413ed92cabece8c8894899bf05e9e084faaef521938ab602f8d9f2265` |
| `stage_b_report.json` | `0bbc1f2e0ab117a1c98b12d987c2bfe531dc35d66dea0540f5e1c4fd485d84c1` |
| `coverage/stereo_candidate_visibility.npz` | `b492f25800ac345ae37cb38c24dc6ea259faad6e3f35dedd849ffd4d5c9a829b` |
| `previews/stage_b_candidate_masks.png` | `0c456669256f2e8a23ae28f3415470087e1de2bb40e32efb4b13e3e5661dc530` |
| `previews/stage_b_selected_keyframes.png` | `42f49e466794723303512f6db8535f27b90d555756d089e135d88857b41ef42e` |

Stage-B 报告的 9 项 gate 全为 true，包括 25 对完整处理、双目时间差、右目图像
器械质量、最少 6 对、`>=99.9%` 等权 proxy coverage 和人工总览验收。阶段结束
后再次核验 table、camera、v9 tissue/ground/ground-plane 与两个 PSM driver，
7 个保护哈希全部不变，ground plane 仍严格为 `[0,0,1,0]`。

复现命令：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

# 生成并人工检查右目独立种子
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/build_super_tissue_right_seed.py \
  --overwrite --accept-visual-review

# 只传播 jaw 接触安全余量以前的右目 499 帧
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/propagate_super_tissue_masks.py \
  --camera-side right \
  --camera-metadata data/super/grasp5_offline_demo/videos/stereo_right.json \
  --timestamps-npy \
    data/super/grasp5_native/tissue_multiview_v1/timestamps_right_native.npy \
  --manual-mask \
    data/super/grasp5_native/tissue_multiview_v1/masks_right/seed/000000-tissue-refined.png \
  --output-dir \
    data/super/grasp5_native/tissue_multiview_v1/masks_right/propagated \
  --max-frames 499 --overwrite

# 首次生成 25 对 mask 和预览；检查总览后只复用本脚本自己的右目图像 mask
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/build_super_multiview_candidate_masks.py --overwrite
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/build_super_multiview_candidate_masks.py \
  --reuse-right-image-masks --overwrite --accept-visual-review
```

### 阶段 C：生成左右目参考的时间戳同步深度

保留 FoundationStereo 为主模型、RAFT-Stereo 为对照和置信度检查，不使用
RAFT strict 交集删除 Foundation 的稠密像素。

左目参考深度沿用当前方法：

```text
目标左目真实帧
  + 时间戳前后的两张右目帧
  -> 两次 Foundation/RAFT 推理
  -> 视差插值到左目时间戳
  -> 左目像素坐标中的稠密深度和置信度
```

新增对称的右目参考深度：

```text
目标右目真实帧
  + 时间戳前后的两张左目帧
  -> 交换左右并水平翻转后推理
  -> 结果翻回右目像素坐标
  -> 视差插值到右目时间戳
  -> 右目像素坐标中的稠密深度和置信度
```

左右视频分别有 `1441/1443` 帧，必须使用各自真实时间戳，不允许用相同帧号
冒充同步。右目深度公式必须从现有 `P1/P2/Q` 验证，并通过左右重投影检查符号
和 baseline 方向，不能手写一个未经验证的平移。

建议中间产物写到独立目录：

```text
data/super/grasp5_native/tissue_multiview_v1/
  observations.json
  masks_left/
  masks_right/
  depth_left/
  depth_right/
  coverage/
  previews/
```

### 阶段 D：只使用现有变换融合到 table 坐标

左目点使用：

```text
left pixel + left depth
  -> rectified-left OpenCV camera
  -> frozen X_table_camera
  -> current table frame
```

右目点使用：

```text
right pixel + right depth
  -> rectified-right OpenCV camera
  -> existing rectified stereo transform
  -> rectified-left OpenCV camera
  -> frozen X_table_camera
  -> current table frame
```

OpenCV 反投影不能直接混用 `cameras.json` 中面向渲染器的 Blender 相机约定。
融合前先单独验证左右点云和 ground 在 table 坐标中的重合情况；全过程不生成
新的 table transform。

几何融合使用约 `0.5 mm` 的 table XY 网格。每个网格单独保存左目支持次数、
右目支持次数、Foundation/RAFT 置信度、时间跨度和高度分布，再用加权中位数或
稳健估计得到上表面高度，不直接平均全部深度。

每个表面/footprint 单元必须分类为：

- `dual-view`：左右目均有直接观测；
- `left-only`：只有左目直接观测；
- `right-only`：只有右目直接观测；
- `inferred`：没有直接观测，只通过小孔插值或显式先验补全。

小于约 `1-2 mm` 的孔洞可以局部插值；更大的缺失区域必须优先寻找新增关键帧，
不能静默平滑填满。最终从 `z=0` 到融合上表面填充 HCP 刚体碰撞球。

### 阶段 E：左右目等权训练 Gaussian 外观

几何和外观分两步训练，避免 RGB 优化把已经融合好的组织形状拉歪：

1. 从融合表面初始化新 tissue Gaussian。
2. 第一阶段冻结 means，只优化颜色、透明度和有限尺度。
3. 左右目观测按 `50%/50%` 等权采样，不能因为左目帧数或 mask 面积更大而
   继续形成左目模型。
4. 每次只训练一个相机的小批量图像或 patch，避免把十几张 `1920x1080`
   图片同时送入 gsplat。
5. RGB、alpha/silhouette 和 depth loss 只在 `visible_tissue` 中计算。
6. 第二阶段如确有必要，只允许 means 在融合表面附近移动不超过
   `0.3-0.5 mm`，并加入 table-frame 几何约束。
7. 左目和右目必须分别报告 RGB、轮廓、深度误差与渲染对比图，不能只给总平均。

当前 Gaussian 只有一个 view-independent RGB，不能同时精确表达左右目不同的
镜面高光。第一版应降低高光权重，学习两目共同的稳定纹理。若以后必须保留
视角相关高光，需要单独扩展 SH 或 camera-conditioned appearance，不能在本轮
暗中改变渲染表示。

### 阶段 F：候选资产和验收门禁

候选目录：

```text
data/super/grasp5_native/bodies_v10_multiview_rigid_tissue_candidate/
```

至少通过以下门禁后，才允许提供 runtime 切换接口：

1. 上述 table、camera、ground、ground plane 和两个 PSM driver 哈希不变。
2. ground plane 仍严格为 `[0,0,1,0]`，组织最低碰撞球不穿过 `z=0`。
3. 左右点云重合误差分别报告；Foundation 与 RAFT 的差异只用于降权和审计。
4. 每个重建单元都有 `dual/left/right/inferred` 来源，明确报告四类面积比例。
5. 左右目可见 tissue mask 的模型投影召回率目标为 `>=98%`，mask 外泄漏目标
   为 `<=2%`；达不到时保留报告，不用平滑或扩大 Gaussian 掩盖问题。
6. 组织仍为一个刚体、半径 `0.5 mm`、紧密重叠填充，总质量初版与 v9 一致。
7. 先在视觉力关闭、PSM-tissue 碰撞关闭时运行左右目渲染和 1000-step 刚体门禁。
8. 通过后才增加独立的 v9/v10 A/B 接口，不覆盖 v9。
9. 双目视觉力属于后续独立门禁：有效 mask 必须包含
   `visible tissue & rendered model coverage & ~tool mask`；左右相机力预算各占
   总预算的一半，不能直接相加为两倍。

每完成一个阶段都要把命令、输入哈希、输出目录、覆盖率、失败项和对比图路径
追加到 `PROGRESS.md`。不能只写“已完成”而没有可复现记录。

### 当前 v9 版本运行命令（继续保留）

以下命令继续运行当前 v9 组织，不会调用尚未实现的 v10。请在 ThinLinc 桌面
终端中执行，脚本会自动使用 `eg_codex` 和 VirtualGL。

先检查 VirtualGL：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
bash scripts/run_demo_thinlinc.sh --check-virtualgl
```

运行深度优化前、当前较稳定的原 `corrected` 器械位姿：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
bash scripts/run_demo_thinlinc.sh --psm-pose-driver corrected
```

运行多轮双目深度微调后再视觉矫正的候选器械位姿：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
bash scripts/run_demo_thinlinc.sh --psm-pose-driver depth_then_visual
```

这部分操作记录形成时，启动脚本默认还是
`--visual-force-iterations 0 --cameras stereo_left`。后续同日已经接入右目并将
默认相机改为双目，见文末“右目视觉力接入”记录；视觉力迭代仍保守地默认为零。

## ThinLinc NVIDIA OpenGL / VirtualGL（2026-07-20）

ThinLinc `:14` 原先的 OpenGL renderer 为 Mesa `llvmpipe`，GUI 中约 8 万个
组织碰撞球全部由 CPU 软件绘制，是当前交互迟缓的主要原因。CUDA 物理本身仍在
A800 上运行，此问题只属于可视化渲染链路。

本轮采用项目本地安装，未覆盖系统 NVIDIA 驱动：

- VirtualGL `3.1.4`；
- 与当前内核模块严格匹配的 NVIDIA OpenGL/EGL `535.104.12`；
- 默认通过无头 EGL 设备 `egl0` 使用第一张 A800；`egl1` 对应第二张；
- 没有创建新的世界坐标、相机坐标或物理坐标变换；深度和 PSM 位姿资产均未改动。

实际验证结果：

```text
EGL device ID: egl0 or egl
EGL device ID: egl1
EGL vendor string: NVIDIA
OpenGL vendor string: NVIDIA Corporation
OpenGL renderer string: NVIDIA A800 80GB PCIe/PCIe/SSE2
OpenGL version string: 4.6.0 NVIDIA 535.104.12
ThinLinc display: :14
```

另外使用与正式程序相同的 Conda、Marsoom 和 Pyglet 创建了隐藏上下文，结果为：

```text
pyglet_vendor=NVIDIA Corporation
pyglet_renderer=NVIDIA A800 80GB PCIe/PCIe/SSE2
pyglet_version=(3, 3)
```

完整程序的 120 秒限时启动没有出现 `NoSuchDisplayException`、GLX/EGL 或 CUDA
错误，但超时发生时仍在为 `79567` 个 sphere 逐个累计刚体质量和惯量，尚未进入
GUI 事件循环。该启动耗时属于 v9 小粒子刚体构建成本，不是 VirtualGL 失败；
VirtualGL 已分别通过 GLX 检查和实际 Marsoom/Pyglet 上下文检查。

`scripts/run_demo_thinlinc.sh` 现在默认自动通过 VirtualGL 启动，并保留所有原有
Python 参数和 `--psm-pose-driver` 接口。快速检查：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
bash scripts/run_demo_thinlinc.sh --check-virtualgl
```

运行当前深度优化后位姿：

```bash
bash scripts/run_demo_thinlinc.sh --psm-pose-driver depth_then_visual
```

切换第二张 A800：

```bash
VGL_DEVICE=egl1 bash scripts/run_demo_thinlinc.sh \
  --psm-pose-driver depth_then_visual
```

仅在诊断软件渲染时使用 `USE_VIRTUALGL=0`。如果系统 NVIDIA 内核驱动以后升级，
脚本会因版本不匹配主动停止，必须同步更新本地 OpenGL/EGL 运行库，不能混用版本。

## 0.5 mm 无规则间隙刚体组织与分阶段夹爪碰撞（2026-07-19，v9）

v8 的 `1.0 mm` 球虽然已经错层重叠，但用户确认仍偏大。当前默认场景已切换到：

```text
data/super/grasp5_native/bodies_v9_dense_0p5mm_rigid_tissue/
```

### 小粒子密集填充

- 碰撞球半径固定为 `0.5 mm`。
- HCP/AB 错层名义球心节距为 `0.82 mm`，小于球直径 `1.0 mm`，名义重叠为 `0.18 mm`。
- 实际最近邻距离 P05/P50/P95 为 `0.785/0.799/0.811 mm`；实际重叠 P05/P50/P95 为 `0.189/0.201/0.215 mm`。
- `100%` 粒子至少与一个邻居重叠。平面三角孔隙覆盖阈值为 `sqrt(3)*r=0.866 mm`，当前 `0.82 mm` 节距低于该值，因此不会再出现 v7 那种规则白色孔隙。
- 粒子数由 v8 的 `6118` 增至 `79567`。组织仍是一个刚体，不是 PBD 软体。
- 球密度按源球总体积重新计算为 `463.8104 kg/m^3`；运行时质量 `19.3228945 g`，与 v7 目标 `19.3228920 g` 一致。

对比图：`data/super/grasp5_native/bodies_v9_dense_0p5mm_rigid_tissue/particle_packing_comparison.png`。

### 为什么增加了碰撞分组优化

Warp 默认会为同一刚体的每一对 sphere 写入“不自碰撞”过滤对。`79567` 个球会产生约 31 亿对，模型会卡在平方级初始化。当前 builder 为每个组织球分配独立 collision group，表达相同的“组织内部不自碰撞”规则，但只使用线性规模的存储；完整 `body_shapes` 列表仍保留，刚体质量和惯量照常累计。

最终场景使用 `separate_collision_group=False` 复制这些组，避免多环境 builder 把独立组重新合并。此修改只作用于明确传入 `individual_collision_groups=True` 的 SUPER 组织，不改变 PushT 等其他刚体的默认行为。

### 先关闭碰撞，之后再打开

当前严格按用户要求分成两个状态：

1. 初始化、位姿对齐和普通回放时，PSM-组织有效碰撞对数为 `0`。
2. 构建时预先分配 shaft 和左右 jaw 共 3 个末端碰撞形状与组织的候选对，共 `3 * 79567 = 238701` 对。
3. GUI 的 `PSM-tissue collision` 复选框可在运行时恢复这些候选对；切换时先同步 GPU，再清空旧接触并重新捕获 CUDA graph，不重建组织资产。
4. `Reset` 会再次关闭 PSM-组织碰撞。
5. 组织-ground 碰撞始终开启，共 `79567` 对，不受这个开关影响。
6. 机械臂其他 link 没有预分配组织接触，之后打开时也只有末端 shaft 和两片 jaw 能碰组织。

本轮只验证了候选对预分配和 `开 -> 关` 切换本身；1000 步稳定性测试在用户要求的初始关闭状态运行。实际开启后夹取还需要单独验收接触时刻、初始穿插、摩擦、接触刚度以及闭合是否能抬起组织，不能把“开关可用”等同于“抓取参数已经完成”。

### 1000 步 A800 门禁

| 检查 | v9 结果 |
|---|---:|
| collision sphere | `79567`，半径 `0.5 mm` |
| 当前 PSM-组织碰撞对 | `0` |
| 预分配末端候选对 | `238701` |
| ground contact pairs | `79567` |
| 组织类型 | 1 个刚体，0 个软体粒子 |
| 最低最终离地间隙 | `0.0155 mm` |
| 1000 步后组织平移 | `0.0217 mm` |
| 最终速度 | `0` |
| 运行时质量 | `19.3228945 g` |
| 纯物理步进 | `5.683 ms/frame` |
| table/camera 哈希 | 前后完全不变 |
| 最终结果 | 全部门禁通过 |

报告：`data/super/grasp5_native/bodies_v9_dense_0p5mm_rigid_tissue/rigid_scene_smoke_report.json`。

### 复现命令

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/repack_super_rigid_tissue.py \
  --output-dir /tmp/super_bodies_v9_0p5mm_rebuild \
  --particle-radius-mm 0.5 \
  --nearest-pitch-mm 0.82 \
  --footprint-radius-mm 1.05 \
  --xy-jitter-mm 0.02 \
  --z-jitter-mm 0.01 \
  --seed 19

PYTHONPATH=.:src /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/smoke_super_rigid_scene.py \
  --device cuda --steps 1000 --psm-pose-driver corrected
```

## 2026-07-24 最新交接状态

当前可运行基线仍为 `bodies_v9_dense_0p5mm_rigid_tissue`，没有创建或切换 v10
scene。阶段 A/B 已在保护哈希全部一致的前提下完成：jaw 闭合前 frame
`0..498` 的 25 对候选已有左右目独立 tissue mask，以及图像分割与
corrected-CAD 并集后膨胀 `30 px` 的 tool mask；总览已人工验收。最终等权
stereo set-cover 选择顺序为
`0/1, 381/382, 57/58, 141/142, 126/127, 420/421`，左/右/等权 proxy
coverage 分别为 `99.991674%/99.998713%/99.995194%`，完整记录在
`data/super/grasp5_native/tissue_multiview_v1/stage_b_report.json`。下一步是
阶段 C，只为最终 6 对帧生成时间戳同步的左右参考 FoundationStereo 深度和
RAFT 对照/置信度，再与 Stage-B proxy 相交；当前 coverage 还不能称为最终深度
有效覆盖。后续 table-frame 融合、双目外观训练和 v10 候选门禁继续按上文计划
执行，当前 v9 的 `corrected` 和 `depth_then_visual` 两条 ThinLinc 接口保持
不变。

## 右目视觉力接入与接触方案冻结（2026-07-24）

用户已冻结下一版组织—器械表示：

- 小粒子只作为带质量、位置、速度和弹性邻接关系的力学节点；
- 器械不直接接触粒子球；
- 粒子外边界生成连续三角 collision skin，器械只和 skin 接触；
- Gaussian 绑定多个粒子，粒子集合通过材料、体积和 rest-curvature 约束产生
  弹性形变；
- rest shape 的原始凹凸必须保留，不能重新引入位置平滑；
- collision skin、单侧不可穿透、CCD、摩擦和夹持抬起的完整设计记录在
  `接触处理.md`。

本轮按“先加右目视觉力”的顺序只完成双目视觉力阶段，没有把 v9 刚体组织切换为
弹性粒子，也没有实现 triangle-skin 接触。

### 已完成的双目链路

1. `example_embodied_super_offline.py`、`run_demo_thinlinc.sh` 和
   `run_demo_browser_12.sh` 默认相机均改为
   `stereo_left,stereo_right`。
2. `DatasetManager` 把每个 `Frames` timestamp 设为各相机实际解码帧的原生
   timestamp，而不是把左目请求时间复制给右目。
3. 新增 `MultiCameraPackedTissueVisualForceWeights`，左右目分别查找自己的
   packed mask；超出某一 mask 资产时间范围时，该相机置为 inactive，不复用
   末帧。
4. pixel loss 改为“每目在自身有效区域内归一化，再对 active cameras 取
   mean”。因此两目都有效时严格为
   `0.5 * L_left + 0.5 * L_right`，mask 面积不同不会改变相机权重。
5. 原 v9 线性总力 `0.005 N`、力矩 `0.00005 Nm` clamp 没有改变；增加右目
   不会把总力预算翻倍。

右目完整 mask 位于：

```text
data/super/grasp5_native/visual_force_masks_right_v1/
```

资产包含全部 `1441` 帧、分辨率 `1920 x 1080`。传播门禁结果：

```text
first prediction IoU with manual = 0.9960768
area ratio min / median / max    = 0.9175133 / 0.9533934 / 1.0034381
temporal IoU min / median        = 0.9877375 / 0.9965020
all gates passed                 = true
```

### 验证结果

`scripts/test_super_stereo_visual_force_weights.py` 的合成门禁验证：

- mask 面积不同仍得到独立相机均值；
- `L_left=1`、`L_right=3` 时总 loss 精确为 `2`；
- 部分右目资产超过覆盖时间后只保留左目，不夹到右目末帧。

`scripts/smoke_super_stereo_visual_force_inputs.py` 在真实视频的首、中、末帧验证：

- 左右目始终 active；
- 两目都使用各自实际解码 timestamp；
- 两目权重均非空且 finite；
- 两份 mask 资产均覆盖 `1441` 帧。

`scripts/smoke_super_stereo_visual_force_backward.py` 使用真实组织 Gaussian、真实
双目帧和真实 mask 完成一次 CUDA 反向传播，结果为：

```text
requested left timestamp = 24.0939159393
decoded right timestamp  = 24.0850334167
left loss                = 0.0351833776
right loss               = 0.0579198822
equal-camera mean        = 0.0465516299
linear force clamp       = passed
moment clamp             = passed
```

报告：

- `data/super/grasp5_native/visual_force_masks_right_v1/report.json`
- `data/super/grasp5_native/visual_force_masks_right_v1/stereo_input_gate_report.json`
- `data/super/grasp5_native/visual_force_masks_right_v1/stereo_backward_gate_report.json`

### 当前启动方式

双目输入现在是默认值，但两份启动脚本仍保守地保留
`--visual-force-iterations 0`。这避免在 triangle skin 和弹性组织尚未实现前，
让视觉力直接拖动当前 v9 整体刚体。只验证双目视觉力时显式执行：

```bash
bash scripts/run_demo_thinlinc.sh \
  --cameras stereo_left,stereo_right \
  --psm-pose-driver corrected \
  --visual-force-iterations 1
```

下一步才是 `接触处理.md` 的阶段 A：在不平滑 rest surface 的前提下构建弹性
粒子、四面体/邻接约束、四粒子 Gaussian binding 和外表面三角 collision skin。

## SUPER 默认器械位姿恢复为 corrected（2026-07-25）

双目视觉力示例此前没有显式传入 `--psm-pose-driver`，而 Python 入口默认值仍为
`strict`，因此直接执行示例命令时会退回未矫正的 LND 位姿。现在已经统一为：

- Python 参数默认值：`corrected`；
- `run_demo_thinlinc.sh`：显式传入 `--psm-pose-driver corrected`；
- `run_demo_browser_12.sh`：显式传入 `--psm-pose-driver corrected`。

脚本把 `"$@"` 放在默认参数之后，所以仍可用
`--psm-pose-driver depth_then_visual` 等显式参数覆盖默认值。当前正式默认使用稳定
的 `psm_part_corrected_pose_driver.npz`，没有切换到仍作为候选保留的
`depth_then_visual`。

## 接触组织自适应粒子密度冻结（2026-07-25）

`接触处理.md` 已按用户要求增加分层密度约束。新弹性资产不把当前 v9 的
`79567` 个刚体球全部照搬为力学节点，而是只继承其表层尺度：

```text
contact skin 与向内 0--3 mm：0.82 mm 节距，0.5 mm 显示尺度
向内 3--5 mm：节点间距从 0.82 mm 连续增加到 1.50 mm
向内 >5 mm：1.50 mm 深层间距
```

当前组织有效厚度约 `3--15 mm`，因此最薄区域会整层保持现有密度；只有较厚组织
的非接触核心被稀疏。深度按无平滑 rest contact skin 的内向几何距离定义，不按
table-frame 绝对高度定义。

表面 skin 目标边长仍约为 `0.82 mm`，不随深层变粗。器械只接触 skin；粒子
`0.5 mm` 只作为调试显示尺度，不创建粒子碰撞球。节点质量由 incident tetra
rest volume 和统一材料密度积分，避免自适应密度改变总质量。

第一版目标为 `25k--40k` 粒子、`100k--180k` tetra、`15k--30k` skin triangles；
硬上限分别为 `45k/200k/30k`。超限时只允许稀疏 `5 mm` 以下深层（最多到
`1.80 mm`）和简化器械 collision proxy，禁止稀疏 skin 或前 `3 mm` 高密层。
详细的 tetra 质量、表面距离、A800 时间和 contact candidate 门禁均已写入
`接触处理.md`。

## 无平滑双目自适应软组织重建（2026-07-25）

上述 `0--3/3--5/>5 mm` 是生成前的估算参数，已被本节真实网格和 A800 实测
取代。正式 demo 仍使用 v9 刚体，本节只产出并验证 v10 候选资产，没有打开
triangle-skin 接触。

### 双目 rest surface

- Stage-B 选出的左右各 6 帧已完成原生参考目 FoundationStereo 深度：
  left=`0,57,126,141,381,420`，
  right=`1,58,127,142,382,421`。
- 左右目先分别做逐帧格点中值和时间中值；双目一致格点再等权融合。没有对
  已观测高度做 Gaussian、均值、中值或 Laplacian 空间平滑。
- 4 个封闭遮挡缺口共 45 个格点只在未观测区做离散调和拓扑补洞；6885 个
  实际参与三角面的源表面顶点保持 bit-identical。
- rest surface 共 6888 个格点、13280 个三角；双目格点占 `92.20%`，推断格点
  仅占 `0.653%`。PLY：
  `data/super/grasp5_native/tissue_multiview_v1/rest_surface_v1/rest_surface.ply`。

生成脚本：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/generate_super_tissue_multiview_depth.py

/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/fuse_super_tissue_multiview_surface.py --overwrite
```

### 最终自适应四面体与 collision skin 候选

`scripts/build_super_adaptive_tissue.py` 使用 TetGen `0.8.3`。顶面 PLC 完全
冻结；TetGen 只允许在原 piecewise-linear facet 内加入 conforming Steiner
点。侧面预先按不超过深层尺度分段，底面使用粗闭合三角，避免复制一份高密顶面。

实测后采用：

```text
d = 到无平滑顶面三角的欧氏距离
0--1.5 mm：0.82 mm
1.5--8 mm：0.82 -> 1.80 mm 几何连续增长
>8 mm：1.80 mm
visual radius：0.50 mm
mechanics collision radius：0
```

该设置保留表面以下约两层当前密度节点，同时让相邻目标尺寸比最大值为
`1.3371 < 1.35`。最终资产：

| 项目 | 结果 |
|---|---:|
| mechanics particles | `37570` |
| tetrahedra | `193845` |
| closed skin triangles | `24196` |
| top / side / bottom skin | `14018 / 6222 / 3956` |
| frozen source top vertices | `6885`，最大位移 `0 mm` |
| dense / transition / deep nodes | `14620 / 21239 / 1711` |
| Gaussian tetra bindings | `2863` |
| connected / inverted / degenerate | `1 / 0 / 0` |
| integrated volume / mass | `42.3859 cm^3 / 42.3859 g`，密度 `1000 kg/m^3` |

文件：

- `data/super/grasp5_native/tissue_multiview_v1/soft_tissue_adaptive_v1/tissue_soft_adaptive.npz`
- `data/super/grasp5_native/tissue_multiview_v1/soft_tissue_adaptive_v1/metadata.json`
- `data/super/grasp5_native/tissue_multiview_v1/soft_tissue_adaptive_v1/tissue_collision_skin.ply`
- `data/super/grasp5_native/tissue_multiview_v1/soft_tissue_adaptive_v1/tissue_mechanics_nodes.ply`
- `data/super/grasp5_native/tissue_multiview_v1/soft_tissue_adaptive_v1/tissue_gaussian_binding.ply`

重新生成：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/build_super_adaptive_tissue.py --overwrite
```

### A800 静置与性能结论

最终资产用 `E=15 kPa`、`nu=0.45`、`12` substeps、`20` material iterations
运行 1000 步：

```text
all finite                    = true
inverted tetrahedra           = 0
maximum total-volume error    = 0.6098%
maximum ground penetration    = 0
maximum anchor drift          = 0
maximum sampled speed         = 0.08220 m/s
stability gate                = passed
pure physics                  = 30.497 ms/frame
Warp mempool high             = 39,178,202 bytes
20 ms performance gate        = failed
```

报告：
`data/super/grasp5_native/tissue_multiview_v1/soft_tissue_adaptive_v1/physics_smoke_report.json`。

因此 A800 显存远够，瓶颈是每帧 `12 x 20` 次 tetra 材料投影；把 iterations
降到 12 可到约 `18.25 ms/frame`，但会产生翻转，不能采用。下一步先优化材料
求解器的 coloring/分块和迭代调度，保持当前已通过的网格与 20 iterations
稳定基线；性能门禁通过后才接入正式 runtime 和实现器械—triangle-skin CCD。

## 自适应 tissue runtime 与第一版 triangle-skin 接触（2026-07-25）

用户已确认接受自适应资产的 `30.497 ms/frame` 纯物理稳定基线，因此本节取代
上一节“先通过 20 ms 门禁再接 runtime”的旧决定。没有降低 `20` 次材料迭代，
也没有稀疏 tissue skin。

### Runtime 切换

- `examples/example_embodied_super_offline.py` 新增
  `--tissue-mode {adaptive_soft,rigid_v9}`，默认使用 `adaptive_soft`；
- adaptive 模式加载
  `tissue_multiview_v1/soft_tissue_adaptive_v1/tissue_soft_adaptive.npz`，
  使用 `37570` 个力学节点、`193845` 个 tetra、`2863` 个 skinned Gaussian；
- 旧 v9 刚体组织保留为 `--tissue-mode rigid_v9` 回退；
- 该节记录第一版 runtime 的 `E=15 kPa`、`nu=0.45`、12 substeps、20
  material iterations；后续先更新为 `E=5 kPa`，当前默认值再由文末
  “厘米级形变、局部正体积屏障与器械防穿越”更新为 `E=2 kPa`；
- 左右目视觉力数据通路保持启用能力；soft Gaussian 只散射到四面体节点，
  PSM 和 ground 不进入 soft force；
- 本节实现时接触默认关闭；文末“默认接触与视觉力保持修正”已改为从安全首帧
  默认开启，并保留 GUI/CLI A/B 开关。

### Triangle-skin 接触实现

新增
`src/embodied_gaussians/physics_simulator/triangle_skin_contact.py`：

- 器械只接触 `24196` 个三角的闭合 collision skin；力学粒子 collision
  radius 全为 0；
- 三个 PSM 接触 mesh 以 `0.50 mm` 尺度生成 `4779` 个确定性表面样本：
  shaft=`277`、jaw1=`2251`、jaw2=`2251`；
- 每次接触更新只让器械样本查询变形 skin BVH，不建立工具三角与 skin 三角
  的笛卡尔积；
- signed gap 小于 margin 时生成单边法向修正，并用 `mu=0.35` 的位移级
  Coulomb 摩擦项跟随器械切向速度；
- 修正按 closest skin triangle 的 barycentric weights 分发到三个节点；
- `12100` 个 skin 节点对应 `63794` 个 incident tetra，接触只在这些 tetra
  上做最低体积比 `0.05` 的全局安全线搜索；
- 每两个 material substeps 做一次接触投影，速度扩张时间同步乘 2；组织材料
  仍保持每个 substep 的完整 20 次投影；
- 深初始穿透预检门槛为 `-1 mm`。状态 `3922` 的 gap=`-6.916 mm`，GUI
  接触开关会拒绝启用，不会把 rest tissue 瞬间推出。

合成 CPU 门禁脚本为 `scripts/test_triangle_skin_contact.py`；验证了接触候选、
barycentric 节点移动、固定点不动、penetration 单调下降和无翻转。

真实 A800 smoke 脚本为
`scripts/smoke_super_triangle_skin_contact.py`，报告：

`data/super/grasp5_native/tissue_multiview_v1/soft_tissue_adaptive_v1/triangle_skin_contact_smoke_report.json`

轻微接触状态 `2557` 的结果：

| 项目 | 结果 |
|---|---:|
| 初始 minimum signed gap | `0.252289 mm` |
| 初始 candidates | `10`，只来自 jaw2 |
| 10 帧后 minimum signed gap | `0.425402 mm` |
| contact margin | `0.400000 mm` |
| inverted tetra | `0` |
| minimum tet volume ratio | `0.601704` |
| maximum speed | `0.080063 m/s` |
| maximum anchor drift | `0` |
| contact disabled | `41.206 ms/frame` |
| contact enabled | `48.437 ms/frame` |
| contact overhead | `7.231 ms/frame` |

报告全部 smoke gates 通过。`4779` 个工具样本保证候选硬上限小于 `5000`；
但这只是轻微接触的 10 帧功能门禁。原阶段 C 的完整时序 signed-gap/CCD
crossing、候选 P95、单 jaw 下压滑动、双 jaw 闭合，以及阶段 D 的夹持抬起尚未
完成。下一步应运行 corrected pose 的连续时序接触门禁，再做“双 jaw 闭合—
保持—上提—松开”实验；不要直接在 rest tissue 的深穿透历史帧打开接触。

## Show Physics 恢复与默认深度位姿修正（2026-07-25）

切换到 adaptive tissue 后，粒子 collision radius 按 triangle-skin 接触方案
设为 `0`。Warp 调试渲染原本直接复用 collision radius，因此 `Show Physics`
虽然默认已勾选，实际画出的是零尺寸粒子；collision skin 也只存在于接触模块，
没有进入调试 renderer。

现在已将调试显示和碰撞彻底分离：

- adaptive asset 的 `particle_visual_radius=0.5 mm` 被保留到 runtime model；
- 物理求解继续使用 `particle_radius=0`，不会恢复粒子球接触；
- physics renderer 使用 model 的浅拷贝，只在拷贝上替换显示半径并挂入
  `24196` 个变形 collision-skin 三角，不修改 live simulation model；
- rigid PSM、材料 tetra、triangle-skin contact 和视觉力计算均不受显示设置
  影响；
- 完整 adaptive asset 的 CPU builder/finalize 门禁验证了 `37570` 粒子、
  `193845` tetra、`24196` skin faces、非零 collision radius 数量为 `0`、
  display radius 全为 `0.5 mm`，耗时 `4.917 s`；
- 重复参数解析门禁确认用户命令最终得到
  `visual_force_iterations=1`、双目相机和 `depth_then_visual`；源码编译、
  shell 语法和 diff whitespace 检查通过。

器械位姿命名也已澄清并统一：

- `corrected` 对应深度优化前的
  `psm_part_corrected_pose_driver.npz`；
- `depth_then_visual` 对应在 corrected 基础上完成双目深度微调与视觉矫正的
  `psm_depth_then_visual_pose_driver_candidate.npz`；
- 两份 driver 均存在且 shape 为 `(5458, 7, 7)`；
- Python 入口、`run_demo_thinlinc.sh` 和 `run_demo_browser_12.sh` 的默认值
  均改为 `depth_then_visual`；
- ThinLinc 脚本本身已默认传
  `--cameras stereo_left,stereo_right`，所以用户省略该参数不会关闭右目；
  脚本末尾的 `"$@"` 仍允许显式覆盖全部默认值。

当前推荐命令：

```bash
bash scripts/run_demo_thinlinc.sh \
  --visual-force-iterations 1
```

启动日志必须打印
`psm_pose_driver=depth_then_visual: ...psm_depth_then_visual_pose_driver_candidate.npz`
和 `enabled_cameras=['stereo_left', 'stereo_right']`。GUI 中 `Show Physics`
应能同时看到 `0.5 mm` 力学节点和橙色 triangle skin。

## 默认接触与视觉力保持修正（2026-07-25）

用户实际回放发现器械仍会进入组织下方、视觉力变化不明显。根因核对结果：

1. `--visual-force-iterations 1` 只开启视觉优化，不会改变原先硬编码为 false
   的 `enable_triangle_skin_contacts`；必须另行勾选 GUI 接触开关。
2. 如果先播放到器械已经穿入、再勾选开关，`-1 mm` 深穿透保护会拒绝开启；
   器械甚至可能完全穿过闭合 tissue 后从底面离开，此时 signed distance 会
   再次变正，不能用“当前正 gap”追回已经错过的 crossing。
3. 视觉优化每 3 个物理步更新一次，但 post-clamp soft force 只被下一个物理
   步消费，后两步没有保持力，有效平均载荷约为设计值三分之一。

### Depth-then-visual rest-skin 扫描

在 CPU 上用正式 adaptive skin、`4779` 个工具样本、
`depth_then_visual + -27 deg` 对 `0..5457` 每 5 个状态扫描一次：

```text
samples                         = 1092
initial minimum gap             = +10.000 mm（查询上限）
minimum sampled gap             = -2.945727 mm @ state 1065
states below contact margin     = 323
states below 1 mm               = 359
states with negative gap        = 295
scan time                       = 5.519 s
```

因此首帧可以安全自动开启接触；关闭接触后整段轨迹确实有大量穿越状态。旧的
strict/corrected smoke 状态编号不能直接拿来解释 depth-then-visual：例如
depth-then-visual 的 state 2557 rest gap 为 `+9.618 mm`，所以新的全序列扫描
取代了单个旧状态的判断。

### Runtime 修正

- Python 增加 BooleanOptionalAction：
  `--psm-tissue-contact/--no-psm-tissue-contact`，默认 true；
- ThinLinc 和 browser 启动脚本显式传 `--psm-tissue-contact`；
- Reset 先恢复首帧 tissue/PSM，再做 signed-gap 预检并开启 triangle skin
  contact；GUI 显示真实 ON/OFF；
- 接触面板每 `0.25 s` 更新 minimum gap、candidate count、maximum
  penetration 和 inversion-safe scale；
- 最新视觉粒子力在图像优化间隔内做零阶保持，Reset 时清空，安全上限不变：
  per Gaussian/particle=`4e-5 N`、total=`0.075 N`；
- `--visual-force-iterations > 0` 时默认显示实际 post-clamp soft-particle
  arrows；只画箭头时跳过额外的 viewer-camera gsplat pass；
- GUI 显示双目 loss、最大视觉目标位移、pre-clamp/applied force budget 和
  active particle count。

CPU force-hold 门禁使用 `3653` 个受力节点、`0.005 N` 测试预算，验证连续重施
加结果 bit-exact，Reset cache 后不再写力。参数默认/重复覆盖、Python 编译、
shell 语法和 whitespace 检查均通过。该轮尚未修改 triangle-skin 求解器；随后
发现并修正的全局冻结缺陷见下一节。视觉力仍没有在缺少联合长序列门禁时超过
`0.075 N` 安全上限。

## 接触全局冻结修正与组织软化（2026-07-25）

### 根因

`depth_then_visual + -27 deg` 的深接触状态 `1065` 可稳定复现“有候选但无接触
痕迹”：初始 gap=`-2.945726 mm`、候选数 `277`。旧算法第一次接触投影的全局
防翻转缩放为 `0.608886`，节点最大只移动 `0.243555 mm`；第二次开始缩放变为
`0`，后续投影全部冻结。

原因是 `63794` 个 skin-incident tetra 共用一个全局安全缩放。旧 kernel 即使
某个 tetra 的四个顶点没有收到本次接触增量也会检查它；任一历史压缩到最低
体积比 `0.05` 附近的无关单元，都可能把整张组织当前接触的缩放降为零。

### 修正

- 本次四节点接触增量能量 `<=1e-20` 的 tetra 跳过安全缩放；
- 当前已低于最低体积比的 tetra，只拒绝令体积继续下降的候选，允许体积保持
  或恢复；
- 接触从每 `2` 个 substeps 一次改为每个 substep 一次；
- 单次最大修正从 `0.40 mm` 降为 `0.10 mm`，避免大修正压坏薄表层 tetra；
- 组织 `E` 从 `15 kPa` 降为 `5 kPa`，速度阻尼从 `12/s` 降为 `4/s`；
- `nu=0.45`、12 substeps、20 material iterations、原始 rest surface、节点/
  tetra/skin 拓扑、anchor 和视觉力 `0.075 N` 总上限均不变；
- GUI 和启动日志显示实际 `E`、阻尼、contact stride 与最大修正，便于确认
  运行的是新配置。

这里没有重新生成 tissue，没有添加任何空间平滑，表面凹凸与纹理 rest
geometry 未被改写。

### CPU 门禁

扩展后的 `scripts/test_triangle_skin_contact.py` 构造两个互不相干的 tetra，
其中一个预先压到 rest volume 的约 `0.02` 且不接触器械。结果：

```text
all gates                              passed
contact candidates                     71
global inversion-safe scale            1.0
contacted maximum displacement         0.400000 mm
unrelated compressed tet displacement  0
contacted tet volume ratio              0.988453
```

新增门禁 `unrelated_tet_does_not_freeze_contact=true` 和
`unrelated_compressed_tet_unchanged=true` 均通过。

在正式 adaptive asset、`depth_then_visual + -27 deg`、state `895` 上做单步
CPU 定点门禁：投影前 gap=`-0.454 mm`，一步后 gap=`+0.048506 mm`，最大节点
位移 `0.892506 mm`，inverted tetra=`0`，最低体积比 `0.058962`。短连续状态
`855,875,895,915,935` 也保持无翻转，最低体积比 `0.486`。

`scripts/smoke_super_triangle_skin_contact.py` 已同步默认使用
`depth_then_visual`，轻接触状态 `895`、深穿透预检状态 `1065`，并在报告中
记录材料阻尼、contact stride 和最大修正。当前环境没有 CUDA 设备，所以旧的
A800 `48.437 ms/frame` 只能作为第一版历史数据；更新后的 A800 十帧性能、
完整连续时序 crossing 和双 jaw 夹持抬起仍待运行，不能声称已经完成。

## 厘米级形变、局部正体积屏障与器械防穿越（2026-07-25）

### 为什么不能允许 tetra 翻转

厘米级形变本身不要求负体积：拉伸和剪切可以在 `J>0` 下达到。负体积代表四个
节点的朝向已经反转；项目的 Neo-Hookean `log(J)`、Gaussian barycentric
绑定、skin 法向和 signed-distance 均不再成立。

正式 adaptive asset 的重力单步扫描：

| E | inverted tetra | 最低体积比 | 总体积误差 |
|---:|---:|---:|---:|
| `0.5 kPa` | `2` | `-0.8059` | `4.407%` |
| `0.75 kPa` | `1` | `-0.5068` | `3.161%` |
| `1.0 kPa` | `1` | `-0.2805` | `2.459%` |
| `1.5 kPa` | `0` | `0.1260` | `1.703%` |
| `2.0 kPa` | `0` | `0.3170` | `1.318%` |

所以默认采用 `E=2 kPa、nu=0.45、damping=2/s`，没有保留
`0.5--1.0 kPa` 失败档。

### 大形变视觉力

当前 soft visual-force 配置：

```text
lr_means                         0.001 m（每次 target 约 1 mm）
kp                               5
per Gaussian force cap           2e-4 N
per particle force cap           2e-4 N
total force cap                  0.30 N
```

`scripts/test_super_soft_force_scatter.py` 已参数化，可对正式 adaptive asset
运行局部大形变门禁。中央半径 `8 mm` 的 `156` 个 Gaussian 使用 `20 mm`
合成目标、连续施力12帧：

- 未加一般 substep 防翻转时：frame 8 位移 `10.358 mm`、frame 12
  `17.473 mm`，但 frame 10 开始翻转，frame 12 为 `15` 个；
- 单个全局步长屏障：翻转为0，但 frame 12 位移仅 `2.008 mm`，因一个危险
  tetra 限制全体节点，已拒绝；
- 最终局部屏障：只回退将要跨零的 tetra 四节点邻域并做四轮传播，极端情况
  才全局兜底。frame 8 位移 `10.414 mm`、frame 12 位移 `16.929 mm`，
  inverted tetra=`0`、最低体积比 `0.001004`、总体积比 `0.992429`，全部
  dynamics gates 通过。

一般材料/视觉力屏障仅要求 `J>=1e-4`，不是用5%体积比限制大形变；接触局部
修正仍采用 `J>=0.01`，因为薄表层被夹爪直接压到接近零没有力学意义。

### 可见表面与运动学器械

组织 Gaussian 最大轴中位数约 `1.35 mm`、上限 `2.0 mm`，旧 contact margin
只有 `0.4 mm`。即使 collision skin 数学上没有负 gap，画面仍可能看到毫米级
Gaussian 外观重叠。当前 contact margin 改为 `1.50 mm`、每 substep 投影、
单次最大修正 `0.25 mm`。

PSM 是 pose driver 直接写入的运动学物体，接触投影只能移动 tissue，不能阻止
器械继续执行一个穿透位姿。现在每次应用 `depth_then_visual` 后，若 fresh
skin gap 小于 `1.50 mm`，no-crossing guard 会把全部 distal PSM link 沿世界
`+Z` 回退到 margin。该方向建立在 table-frame 高度场和器械从上方接近的场景
假设上；GUI 显示 guard gap 和最近一次回退量。

state `895` 的最终联合 CPU smoke：

```text
gap before                          -0.453918 mm
gap after                           +1.499949 mm
contact candidates before/after     126 / 1
inverted tetrahedra                 0
minimum tetra volume ratio          0.252849
contact safe scale                  1.0
all gates                           passed
```

GUI 还新增 `Tissue maximum displacement from reset`，可以直接检查是否达到
`10--20 mm`，不再凭视觉缩放猜测。CPU 接触回归、Python 编译、shell 语法和
whitespace 门禁均通过。由于当前执行环境没有 CUDA，新增局部屏障和运动学
fresh-gap 查询的 A800 连续回放耗时、真实双目 RGB 是否持续给出正确方向的
厘米级目标，以及完整双 jaw 夹持抬起仍待实际 A800 验收。

## 特别柔软但表面不可穿透：定向 top-skin 屏障（2026-07-25）

### 本轮纠正

上一版 `E=2 kPa`、`+Z` kinematic no-crossing guard 的组合虽然能消除数学
穿透，却导致 tissue 太硬且器械位姿偏离。该方案已被当前实现取代：

- 材料改为 `E=250 Pa、nu=0.40、gravity=0、damping=1/s`；
- `ADAPTIVE_TISSUE_ENABLE_KINEMATIC_GAP_GUARD=False`，不再回退器械；
- contact margin 保持 `1.50 mm`，每 substep 最大修正改为 `1.00 mm`；
- 接触全局体积缩放关闭，材料局部正体积下限保留为 `J=0.001`；
- PSM 的 `depth_then_visual` 目标位姿在每帧 `12` 个物理 substep 内做 SE(3)
  插值，避免运动学瞬移直接跨过薄 skin。

### 穿透根因与修复

旧封闭 skin 查询在夹爪从上表面穿到组织下方后会再次返回 outside，因此接触
消失。现在对 `14,018` 个原始上表面三角增加定向单边 barrier，始终沿当前
变形三角法向判断夹爪是在表面上方还是下方，不依赖封闭体 parity。

为避免只压三个表面顶点，每个 top face 还预计算半径 `4 mm`、向内深度
`12 mm` 的平滑支撑柱。正式资产共有 `5,070,014` 个 face-to-particle 映射，
GPU ids+weights 约 `40.56 MB`；通用 contact spread 改为 `0` 层，避免重复
扩散。三角 skin 仍是唯一器械碰撞表面，粒子仍只作为力学节点。

### 回归结果

真实器械轨迹 state `895 -> 1065` 的三帧 CPU 回归通过：

```text
undeformed deep gap              -2.945727 mm
final oriented top gap           +0.714156 mm
top barrier contacts             74
maximum tissue displacement      5.193470 mm
P99 tissue displacement          4.067498 mm
minimum tet volume ratio         0.001888
inverted tets / anchor drift     0 / 0
PSM target-pose error            0
```

同一 `E=250 Pa` 材料的正式 `37,571` 节点视觉力门禁运行 `12` 帧后最大位移
`6.115 mm`，无翻转、anchor 固定，全部 gates 通过。当前环境无 CUDA；A800
连续 GUI 回放的帧耗时和完整夹持抬起仍是下一项验收，不能以 CPU 时间外推。

## 论文固定参数基础软体：保留 Gaussian 与视觉力，移除旧接触主路径（2026-07-28）

用户决定放弃此前效果不理想的 triangle-skin/top-skin 接触方案，先实现
arXiv:2309.11656 在 residual mapping 和在线刚度更新之前的基础软体状态。当前
新增 `paper_soft` 模式并设为 Python、ThinLinc 和 browser 启动默认值：

- 继续加载无平滑双目自适应资产的 `37570` 个力学粒子、`193845` 个 tetra 和
  `2863` 个 Gaussian；
- Gaussian 的 tetra id、四粒子 barycentric weights 和 rest offset 原样保留，
  每个物理步后继续由局部变形梯度更新；
- 材料使用固定参数 Neo-Hookean XPBD：
  `E=250 Pa`、`nu=0.40`、`gravity=0`、`damping=1/s`、`12` substeps、
  `20` material iterations、`min J=0.001`；
- 当前没有深度点云 residual mapping，也没有任何全局/局部刚度优化；运行时
  显式记录 `residual_mapping_enabled=false`、
  `stiffness_optimization_enabled=false` 和 fixed material；
- `paper_soft` 在 builder 层不注册 collision-skin faces，也不构建
  triangle-skin projector，不执行 particle-shape、particle-particle 或
  PSM-tissue contact；即使外部请求打开旧接触，运行时也保持关闭；
- masked stereo RGB visual force 保留并默认使用
  `--visual-force-iterations 1`；只有 tissue soft Gaussian 获得 pose gradient，
  PSM 和 ground 仍被隔离；
- soft Gaussian target 继续按四点权重散射到粒子，并保留 per-Gaussian、
  per-particle 和 tissue-total clamp。

旧实验没有删除：`--tissue-mode adaptive_soft --psm-tissue-contact` 仍可显式
运行 triangle-skin 接触，`--tissue-mode rigid_v9` 仍为刚体回退；两者不再是
默认主路径。

新增 `scripts/test_super_paper_soft_mode.py`。CPU 真实环境装配门禁结果：

```text
particles / tetrahedra / soft Gaussians = 37570 / 193845 / 2863
Gaussian four-particle binding          = passed
only soft Gaussians receive gradients   = passed
fixed uniform mu/lambda                 = passed
depth residual / stiffness optimizer    = OFF / OFF
triangle/particle contact               = OFF
all gates                               = passed
```

同一正式自适应资产的 soft visual-force wiring 门禁通过：`2863` 个组织
Gaussian 有梯度、固定粒子受力为 0、总力 clamp 为 `0.0050000008 N`、全部
finite。使用 `paper_soft` 固定材料参数和当前视觉力上限的三帧 CPU 完整动力学
短门禁结果：

```text
maximum tissue displacement             = 6.1581 mm
inverted tetrahedra                      = 0
minimum tetra volume ratio               = 0.001011
maximum total volume drift               = 1.9376%
anchor drift                             = 0
all dynamics gates                       = passed
```

当前默认启动：

```bash
bash scripts/run_demo_thinlinc.sh
```

等价关键参数：

```bash
python examples/example_embodied_super_offline.py \
  --tissue-mode paper_soft \
  --no-psm-tissue-contact \
  --visual-force-iterations 1 \
  --cameras stereo_left,stereo_right
```

该阶段只确认“组织在任何深度残差和刚度优化之前已经是固定材料参数软体”，以及
Gaussian/视觉力链路仍完整；不把 RGB visual force 解释成论文的深度 residual
mapping，也不宣称已经得到论文的在线局部刚度辨识结果。

## paper_soft 接触恢复与局部视觉位置修正（2026-07-28）

### 用户反馈与根因

用户要求重新开启器械/组织接触，并指出即使把
`--visual-force-iterations` 调到 `8`，组织仍没有明显局部变化。

`iterations` 只控制临时 Gaussian 图像优化次数，不是物理位移增益。旧视觉序列
报告还暴露了更关键的问题：一次 Adam 更新后，最大 Gaussian target 固定为
`0.173206 mm`，正好接近三个坐标轴各 `0.1 mm` 的步长；微小非零梯度和真正
显著的局部图像梯度被 Adam 归一成近似相同的更新。再经过两层 tetra 邻域扩散，
局部残差变成了大范围低幅度作用。旧 100 帧报告相对 replay floor 的额外非刚性
变化只有约 `0.0167 mm`，因此画面上基本不可见。

### 当前实现

- `paper_soft` 重新注册 `24196` 个 collision-skin triangles，并构建包含
  `4779` 个器械表面采样点的 triangle-skin projector；
- Python、ThinLinc 和 browser 默认启动参数均改为
  `--psm-tissue-contact`，GUI 仍可随时开关；
- 接触只做器械/组织几何约束，不重新启用深度 residual 或刚度优化；
- 视觉力比例增益由 `kp=5` 提高到 `kp=10`；
- per-Gaussian / per-particle / tissue-total 上限由
  `0.2 mN / 0.2 mN / 0.30 N` 提高到
  `0.4 mN / 0.4 mN / 0.60 N`；
- soft force 的 tetra 邻域扩散由 `2` 层改为 `0` 层，避免把局部目标摊到
  周围组织；
- 每次视觉求解的第一次反传按 mean-gradient norm 排序，只保留最强的
  `10%` soft Gaussian。后续 `iterations` 固定使用这一局部集合，避免 8 次
  Adam 更新逐轮蔓延到整个组织；
- GUI 将 `Visual target iterations`、`Visual physical gain kp` 和
  `Visual local Gaussians %` 分成三个独立滑块，避免再把 iterations 当成
  物理作用强度；
- GUI 新增 `selected Gaussians / active particles`，可直接确认当前视觉力是否
  真正落在局部区域。

真实环境 CPU 装配门禁：

```text
particles / tetrahedra / soft Gaussians = 37570 / 193845 / 2863
collision skin / tool samples           = 24196 / 4779
paper_soft contact enable / disable     = passed / passed
local gradient gate / zero spread       = passed / passed
depth residual / stiffness optimizer    = OFF / OFF
```

正式 adaptive 资产的局部合成门禁使用中心半径 `8 mm` 的 `156` 个 Gaussian、
`8 mm` target、当前 `E=250 Pa` 材料和零层扩散，三步结果：

```text
near-region mean force                  = 0.04305 mN
far-region maximum force                = 0
maximum local particle displacement     = 7.5494 mm
inverted tetrahedra                     = 0
minimum tetra volume ratio              = 0.001001
maximum total volume drift              = 1.2037%
anchor drift                            = 0
all gates                               = passed
```

这项门禁证明局部 target 到局部粒子形变的力学链路可用；当前无 CUDA，真实双目
RGB 梯度在 GUI 中会选择哪一组 Gaussian、方向是否符合视频，仍需在 A800 回放
时结合 `selected Gaussians`、`active particles` 和局部形变观察确认。

同一 `paper_soft` 模式的真实器械轨迹三步 CPU 接触回归也通过：

```text
gentle contact enable                    = passed
deep top-skin barrier final gap          = +0.7142 mm
deep top-barrier contacts                = 74
maximum contact displacement             = 5.1935 mm
minimum tet volume ratio                 = 0.001888
inverted tets / anchor drift             = 0 / 0
PSM target-pose component error          = 0
```

## 回滚局部视觉力实验并修正预览倍率（2026-07-28）

用户反馈在 `iterations=1` 时开启 `Visual Forces Meshes` 会看到大量 Gaussian
预览飞离组织，要求撤回上一轮视觉力实验。

已回滚上一节中的全部视觉物理改动：

- `kp: 10 -> 5`；
- per-Gaussian / per-particle 上限：`0.4 mN -> 0.2 mN`；
- tissue-total 上限：`0.60 N -> 0.30 N`；
- soft force spread：`0 -> 2` 层；
- 删除 strongest-10% Gaussian 梯度选择及相关配置、指标和 GUI 滑块。

飞离画面的直接显示原因是通用 viewer 默认把优化后的 Gaussian target pose
外推 `25x`，并把 soft particle force 再乘 `10x` 后绘制。该预览倍率不代表真实
Gaussian 或力学粒子已经移动。SUPER 入口现在显式使用：

```text
visual force target-pose display scale  = 1
force vector scale                      = 1
soft force display gain                 = 1
```

因此回滚后仍可观察视觉力，但 mesh 和箭头按实际量级显示。`paper_soft` 固定材料、
Gaussian 四点绑定、深度 residual=OFF、刚度优化=OFF，以及用户要求重新开启的
PSM/tissue contact 均保持不变。

## 当前默认轨迹“软体像刚体”的真实根因与修复（2026-07-29）

用户在 ThinLinc 默认
`raw_paper_lnd_sam2_dense_contact_closedjaw + roll=0 deg` 轨迹中观察到器械
落到组织平面下方、组织完全不变形。

针对该准确启动配置运行接触回归后确认，问题不是材料太硬：

```text
state 895 initial gap                    = -3.1457 mm
state 1065 initial gap                   = -5.4965 mm
contact candidates                       = 2310 / 12276
old set_psm_tissue_collisions result      = false
old maximum tissue displacement          = 0
```

旧开关在初始穿透超过 `1 mm` 时直接拒绝启用接触。当前轨迹到达相关帧时已经在
组织内部 `3--5.5 mm`，所以虽然 GUI 加载了 `paper_soft`，真正的 contact solver
从未工作；继续降低 Young's modulus 不可能产生变形。

当前修复：

- 删除 `1 mm` 初始穿透拒绝条件；
- 有有效 candidates 时允许从 overlap 状态启用 triangle-skin contact；
- 由现有 `10 mm` query range、定向 top-skin barrier、每 substep `1 mm`
  correction 和局部 `J>=0.001` 材料屏障逐步恢复；
- 仍然严格保持 PSM 离线目标位姿，不重新启用旧 `+Z` 器械回退；
- `E=250 Pa` 保持不变。试验过 `100 Pa`，它没有增加可见位移，反而使深接触面
  更容易整体让开，已撤销；
- 试验过 `2 mm/substep` correction，多采样点修正互相干扰、恢复变差，也已撤销。

同一默认轨迹、`E=250 Pa`、三物理步 CPU 结果：

```text
state 895 contact enable                 = true
gap before / after                       = -3.1457 / +1.3905 mm
maximum tissue displacement              = 5.0261 mm
state 1065 contact enable                = true
gap before / after 3 steps               = -5.4965 / -1.4695 mm
maximum tissue displacement              = 5.1771 mm
inverted tetrahedra / anchor drift       = 0 / 0
PSM pose error                           = 0
```

深帧是从已穿透 `5.5 mm` 的状态直接跳入，只跑三步仍有 `1.47 mm` 待逐步恢复；
顺序回放会在进入该深度前持续执行接触。该接触仍是非粘附的单向几何接触：
它能压动组织，但器械离开表面后不会自动“粘住并提起”组织；夹持提拉需要独立的
jaw grasp/attachment 约束，不能用进一步降低组织刚度代替。

## 改为简单顶部表面约束并关闭全部体积安全层（2026-07-29）

用户要求放弃器械回退和体积安全裁剪，只保留原理直接、能看见局部压陷的表面
约束。当前 `paper_soft` 路径已经改为：

1. 对器械表面采样点查询组织顶部定向三角面；
2. 使用单侧约束
   `g = dot(x_tool - x_surface, n_top) >= contact_margin`；
3. `g` 违反时不再平均稀释所有采样，而是给命中的表面粒子保留“最深穿透”
   所要求的向下位移；
4. 同一位移按权重传播到命中三角形周围 `4 mm`、向下 `12 mm` 的粒子柱；
5. 每个物理子步只保留 `0.1 mm` 位移上限；
6. PSM 位姿完全不回退，Gaussian 四粒子绑定继续随物理粒子更新。

已显式关闭：

```text
triangle_skin_contact_min_volume_ratio = 0
material_min_volume_ratio              = 0
kinematic_gap_guard                    = OFF
depth residual                         = OFF
stiffness optimization                 = OFF
```

实际默认 dense closed-jaw 轨迹、三物理步 CPU 回归：

```text
state 895 gap before / after            = -3.1457 / +1.4144 mm
state 895 maximum / p99 displacement    = 9.9486 / 4.3889 mm
state 1065 gap before / after           = -5.4965 / +1.3170 mm
state 1065 maximum / p99 displacement   = 14.8567 / 10.2382 mm
PSM pose error                          = 0
surface-crossing gates                  = PASS
```

因为用户明确要求删除体积安全层，压力跳变测试不再保证四面体正体积；上述两个
直接跳入既有穿透的测试分别出现 `70` 和 `370` 个翻转单元。这是关闭安全层的
直接后果，不再由求解器回退或隐藏。

视觉力同时改为小目标位移和直接局部散射：

```text
lr_means / kp                           = 0.0001 / 25
per Gaussian / particle force cap       = 0.03 / 0.03 mN
total force cap                         = 0.02 N
spread layers                           = 0
```

无体积安全层的一物理步局部回归得到 `0.887 mm` 最大位移、远场力为 `0`、
翻转单元为 `0`，因此 `interaction=1` 已能产生局部、可见且不经二次扩散的组织
位置修正。

## GUI 一直显示刚体的根因：render_state 未同步软粒子（2026-07-29）

用户在实际 ThinLinc GUI 中确认 Physics Particles 完全不动，尽管接触 smoke test
中的 `state_0.particle_q` 已有约 `10--15 mm` 位移。定位到
`SimulationViewer` 使用一份独立的 `render_state`：

- 原实现每帧只从物理状态同步 `body_q`；
- 从未同步 `particle_q`；
- 因此 Physics Particles 和由粒子构造的组织 collision-skin 三角面永远显示
  rest pose；
- 器械移动正常，但屏幕上的组织表面是假静态表面，于是视觉上始终表现为器械穿透
  且组织像刚体。

当前 `_refresh_body_q()` 在每次 render 前同时执行：

```text
state_0.body_q     -> render_state.body_q
state_0.particle_q -> render_state.particle_q
```

headless CPU 回归使用两组非零测试粒子，确认 render state 得到逐元素完全相同的
位置。ThinLinc 启动脚本同时把当前仓库的 `src`、`examples` 放在
`PYTHONPATH` 最前面；GUI 启动时检查 `embodied_gaussians.__file__`，若加载到
另一个 checkout 会直接报错而不是继续运行。

## 更软、更局部，并增强视觉力（2026-07-29）

用户在修复 GUI 粒子同步后确认能够看到接触变形，但反馈组织仍偏硬、接触压陷
范围太大、视觉力不可见且效果弱。本轮将三种影响分开调整：

```text
Young's modulus                         250 -> 100 Pa
top-contact support radius              4.0 -> 2.5 mm
top-contact support depth               12.0 -> 6.0 mm
top support entries                     5,070,014 -> 1,769,802
visual lr_means                         0.0001 -> 0.0002
visual Gaussian / particle force cap    0.03 -> 0.10 mN
visual total force cap                  0.02 -> 0.05 N
visual force update interval            3 -> 1 frame
force vector scale / soft display gain  1/1 -> 5/5
```

视觉力仍使用 `spread_layers=0`，所以增强后的力直接进入绑定粒子，不通过邻接
四面体做二次扩散。Gaussian target mesh 的 pose scale 仍为 `1`，避免再次出现
目标 mesh 飞离组织。

固定参数、Gaussian 四粒子绑定、体积安全 OFF、深度 residual OFF、刚度优化 OFF
均保持不变。

实际 dense closed-jaw 轨迹三步 CPU 回归：

```text
state 895 gap before / after             -3.1457 / +1.4896 mm
state 895 maximum / p99 displacement     8.8990 / 4.3339 mm
state 1065 gap before / after            -5.4965 / +1.4632 mm
state 1065 maximum / p99 displacement    15.3296 / 10.1485 mm
PSM pose error                           0
surface-crossing gates                   PASS
```

增强视觉力的一物理步局部测试产生 `5.142 mm` 最大位移，远场力为 `0`。由于按
用户要求保持全部体积安全关闭，该强视觉力测试出现 `9` 个翻转单元；该统计只记录
而不触发裁剪或回退。

## 进一步缩小接触范围并增强箭头显示（2026-07-29）

按用户反馈继续将表面约束支持柱从 `2.5 × 6.0 mm` 缩到
`1.5 × 4.0 mm`，支持映射从 `1,769,802` 项降到 `531,223` 项。为了让更小
范围仍能跟上深接触，单子步表面位移上限从 `0.1` 调到 `0.2 mm`；这不扩大粒子
支持范围。

视觉物理力保持上一节数值不变，只把 `Force Vector Scale` 和
`Soft Force Display Gain` 从 `5/5` 提高到 `8/8`。Gaussian target pose scale
仍为 `1`，因此只是箭头更明显，不会再次放大 target mesh。

实际 dense closed-jaw 三步 CPU 回归：

```text
state 895 gap before / after             -3.1457 / +1.4530 mm
state 895 maximum / p99 displacement     8.6282 / 3.7453 mm
state 1065 gap before / after            -5.4965 / +1.4069 mm
state 1065 maximum / p99 displacement    14.2506 / 9.1946 mm
PSM pose error                           0
surface-crossing gates                   PASS
```

## 组织继续软化并缩小接触范围（2026-07-29）

按用户要求继续降低固定 `paper_soft` 材料刚度，并收缩顶部表面约束的局部支持柱：

```text
Young's modulus                         100 -> 50 Pa
top-contact support radius              1.50 -> 1.25 mm
top-contact support depth               4.00 -> 3.50 mm
top support entries                     531,223 -> 348,674
contact correction cap                  0.20 -> 0.30 mm/substep
```

纠正上限只控制已命中粒子的单子步最大位移，不扩大接触半径或深度。视觉力、
Gaussian 四粒子绑定、深度 residual=OFF、刚度优化=OFF、体积安全=OFF 均保持
不变。

测试过更小的 `1.0 × 3.0 mm` 支持柱：浅接触可通过，但默认轨迹直接跳到
state 1065 的深接触位姿后仍残留约 `2.26 mm` 穿透；把纠正上限加到
`0.4 mm/substep` 也无效。因此没有保留这个过窄、呈针状变形的失败组合。

最终 `1.25 × 3.5 mm`、`0.3 mm/substep` 配置在实际 dense closed-jaw 轨迹、
三物理步 CPU 回归中通过全部接触门禁：

```text
state 895 gap before / after             -3.1457 / +1.4483 mm
state 895 maximum / p99 displacement     9.7360 / 4.4616 mm
state 1065 gap before / after            -5.4965 / +1.1983 mm
state 1065 maximum / p99 displacement    15.7934 / 8.1512 mm
PSM pose error                           0
surface-crossing gates                   PASS
```

由于按用户要求保持体积安全关闭，两个直接跳入既有穿透的压力测试分别记录到
`94` 和 `200` 个翻转四面体；它们不触发隐藏裁剪或器械位姿回退。

## 改为极软材料、视觉力主导（2026-07-29）

用户明确当前目标不是继续依赖器械接触制造主要形变，而是让材料只提供很弱的
连续性，组织位置主要由 masked RGB 视觉力修正。本轮把 `paper_soft` 调整为：

```text
Young's modulus                         50 -> 35 Pa
visual lr_means                         0.0002（不变）
visual kp                               25 -> 0.05
per Gaussian / particle force cap       0.10 -> 0.03 mN
visual total force cap                  0.05 -> 0.02 N
ThinLinc visual-force iterations        1 -> 3
force spread layers                     0（不变）
```

`kp` 和力上限降低不是把视觉修正关小。旧配置在第一次 Adam mean update 后，
`kp × displacement` 已远超 `0.1 mN`，所以 interaction=1 就进入限幅；
interaction 从 1 调到 8 基本不会继续增加实际粒子力，只会继续移动临时 target
mesh。现在一次 interaction 保持在限幅以下，1--3 次迭代能够真实分级改变物理
力；极软材料则把较小的力转换为可见位移。默认采用 3 次迭代。

测试过 `E=5/10/20 Pa`。在关闭全部体积安全的前提下，这些档位面对现有视觉力或
深接触会产生针状折叠：`5 Pa` 局部一步约 `14.22 mm` 位移并翻转 362 个
tet；`10/20 Pa` 的深接触最终仍分别残留约 `1.68/1.88 mm` 穿透。因此最终采用
当前 `1.25 × 3.50 mm` 接触支持范围下最低通过的 `35 Pa`，没有靠扩大接触范围
通过门禁。

35 Pa 的实际 dense closed-jaw 三物理步 CPU 接触回归：

```text
state 895 gap before / after             -3.1457 / +1.5040 mm
state 895 maximum / p99 displacement     10.8602 / 4.8522 mm
state 1065 gap before / after            -5.4965 / +1.3036 mm
state 1065 maximum / p99 displacement    15.6974 / 7.6611 mm
PSM pose error                           0
surface-crossing gates                   PASS
```

用 `0.6 mm × kp=0.05` 的等效局部视觉目标测试 156 个 Gaussian，一物理步最大
粒子位移约 `1.2766 mm`、最大速度约 `0.0942 m/s`、远场粒子力为 `0`，说明
视觉路径已可产生局部毫米级位置修正。该极限测试记录到 5 个翻转 tet；体积安全
仍按用户要求关闭，不做隐藏裁剪。

当前 CPU 环境没有 CUDA，无法在本轮重跑真实双目 1080p gsplat backward；
固定材料、Gaussian 四粒子绑定、视觉力 scatter 和器械表面接触门禁均已通过。
深度 residual 与在线刚度优化继续关闭。

## 简化为双夹爪三角面接触与高摩擦（2026-07-29）

根据最新要求，放弃此前试验的“粒子同时靠近两片夹爪后捕获、保存双锚点、
张开后按距离释放”方案。当前运行路径没有持久夹持状态，也不再根据粒子相对
两片夹爪的距离决定是否随夹爪运动。夹取只来自接触期间的库仑摩擦；失去法向
接触后摩擦自然消失。

旧的粒子—夹爪距离查询、双锚点捕获和持续夹持 Warp 内核也已从源文件删除，
不是仅靠配置关闭；当前代码中没有可被误触发的旧夹持路径。

视频中只有两片夹爪接触组织，因此轴杆也已从接触形状列表彻底删除：

```text
active contact shapes                   jaw_1 + jaw_2 only
shaft contact / top barrier             OFF / OFF
persistent grip anchors                 OFF
particle-to-jaw vertex query            OFF
jaw surface samples                     4,304 + 4,298
tissue collision triangles              24,196
```

每个夹爪表面采样点查询连续组织三角面，命中后的法向与切向修正按重心权重分配
给该三角形三个物理节点。因此三角形内部不再是只有顶点才响应的碰撞空洞。接触
只采用统一的网格尺度容差，不另设深穿透恢复窄带：

```text
surface contact margin                  0.40 mm
maximum direct correction               0.05 mm / substep
jaw Coulomb friction coefficient        1.5
explicit contact spread layers          0
```

为了减少粒子间横向联动，同时保留最基本的连续体连接，固定材料调整为：

```text
Young's modulus                         20 Pa
Poisson ratio                           0.00
material solve                          4 iterations × 0.04 relaxation
velocity damping                        20 / s
volume safety                           OFF
```

Gaussian 四粒子蒙皮、masked RGB 视觉力、`lr_means=0.0002`、`kp=0.05` 和默认
三次视觉迭代均保留；深度 residual 和在线刚度优化继续关闭。

实际 dense closed-jaw、三物理步 CPU 回归：

```text
state 895 jaw contacts before / after   255 / 5
state 1065 jaw contacts before / after  309 / 2,589
shaft / top-barrier contacts            0 / 0
active contact shape ids                25, 27
persistent grip constraint              false
contact spread layers                   0
maximum PSM pose error                  0
all configured contact gates            PASS
```

该配置有意允许极局部粒子被高摩擦沿夹爪牵拉：浅/深测试的 `p99` 位移分别约
`0/0.0098 mm`，说明位移没有大范围传播；少量直接接触节点的最大位移约
`3.17/12.71 mm`。由于按用户要求不恢复体积安全层，两个直接跳入接触位姿的压力
测试仍记录到 `21/107` 个翻转四面体，不执行隐藏裁剪或器械位姿回退。

## 移除乱飞的红色碰撞表面调试显示（2026-07-29）

红色乱飞平面不是额外的物理组织，也不是夹爪接触面。原因是此前物理调试
渲染器把 `24,196` 个组织碰撞三角面作为不透明红色表面绘制；超软组织中少量
接触节点发生较大局部位移时，相连三角形会被拉成长片，视觉上便成为各种乱飞
的红色平面。

现已从通用物理调试渲染器删除碰撞三角面的注入，只保留随实时
`particle_q` 更新的红色物理粒子。接触求解器内部的连续组织三角面没有删除，
所以双夹爪三角面接触、摩擦、Gaussian 蒙皮和视觉力均不受影响。
SUPER 启动配置同时明确关闭红橙色的 visual-force Gaussian meshes/outlines，
视觉力只保留实际粒子箭头。

验证：

```text
physics debug collision-skin injection    removed
live physics particle refresh             preserved
paper_soft configuration gates            PASS
two-jaw triangle contact projector         preserved
soft Gaussians / visual force              preserved
```

## 恢复红色组织表面并消除粒子乱飘（2026-07-29）

上一节“隐藏红色碰撞表面”已被本次要求取代。红色连续组织表面重新显示：
物理调试渲染器继续用实时 `particle_q` 更新 `24,196` 个组织表面三角形。

本次确认乱飞不只是显示问题：原 `20 Pa + 4×0.04` 同时把四面体剪切连接削得
过弱，局部接触粒子容易近似脱离邻域。为了保持组织非常软、同时避免重新产生
大范围联动，采用以下固定材料：

```text
Young's modulus                         50 Pa
Poisson ratio                           0.00
material solve                          8 iterations × 0.10 relaxation
velocity damping                        30 / s
explicit contact spread layers          0
volume safety                           OFF
```

`ν=0` 和零接触扩散继续限制横向传播；增加的是相邻四面体的剪切连接强度，不是
夹爪捕获约束、体积安全层或全局位移传播。

视觉力实际数值、`lr_means=0.0002`、`kp=0.05` 和力上限均未改变。仅显示调整：

```text
soft force display gain                 8 → 40
force vector scale                      8 (unchanged)
soft arrow line width                   2 → 4
force Gaussian meshes/outlines          OFF
```

`paper_soft + dense_contact_closedjaw` 三物理步 CPU 回归：

```text
                                      shallow       deep
inverted tetrahedra                  0             0
maximum particle displacement        1.039 mm      1.056 mm
p99 particle displacement            0.085 mm      0.112 mm
minimum tetrahedron volume ratio      0.536         0.631
shaft / top-barrier contact           0             0
all contact gates                     PASS          PASS
```

## 删除夹爪 0.4 mm 接触捕获硬截断（2026-07-29）

此前夹爪接触把同一个 `0.4 mm` 同时用于：

1. 期望保持的表面接触间隙；
2. 最近组织三角面的最大查询距离。

第二项会导致夹爪一旦穿过表面超过 `0.4 mm`，查询直接返回无命中，接触从求解器
中消失。现已解耦：

```text
desired contact margin                  0.4 mm
jaw nearest-triangle query              10.0 mm
explicit distance <= margin cutoff      removed
```

`10 mm` 只是 Warp 最近三角面查询所需的有限 broad-phase 范围，不是接触激活
间隙。实际是否产生修正仍由有向表面距离与 `0.4 mm` 目标间隙决定：夹爪位于表面
上方且间隙足够时不施力；夹爪已穿入表面时，即使深度超过 `0.4 mm` 仍持续产生
法向恢复修正。

连续下压到额外 `-3 mm` 的 CPU 诊断中，最深记录约为 `-3.62 mm`，此时仍有
`1,497` 个 jaw-triangle contacts，证明旧的突然掉线条件已经删除。

同时保留一个重要的未解决事实：查询不再掉线并不等于能瞬间消除深穿透。当前
稳定上限仍为 `0.05 mm/substep`；正式深位姿压力测试从 `-2.91 mm` 改善到约
`-2.39 mm`（三物理步），但没有在三步内回到表面。曾试验
`0.25 mm/substep`，浅接触恢复更快，但深接触产生 `50` 个翻转四面体和约
`13.9 mm` 局部拉飞，因此已拒绝并恢复 `0.05 mm/substep`。当前
`paper_soft` 配置门禁通过；旧 smoke 中要求三步内完全无穿透的两个门禁会如实
失败，不再用 `0.4 mm` 查询截断隐藏真实深度。

## 论文式组织软化标定：阶段 A 输入冻结（2026-07-30）

按照 `组织软化.md` 的顺序，本轮只完成 arXiv:2309.11656 适配流程的阶段 A，
没有提前实现残差映射或材料优化。论文原方法以可见表面点云的 Chamfer 残差和
PBD 几何能量估计体内残差，再用残差、历史静止一致性和空间平滑更新局部刚度；
本项目后续将保留这一结构，但把待标定量换成当前四面体 XPBD 的全局杨氏模量
和速度阻尼。

新增 `scripts/freeze_super_tissue_calibration_inputs.py`，生成
`data/super/tissue_calibration_v1/stage_a_frozen_manifest.json`。manifest 固定：

- 当前 ThinLinc 主入口的
  `raw_paper_lnd_sam2_dense_contact_closedjaw` pose driver；
- 人工修正 `self-spin=-9.184 deg`、image X=`+0.332 mm`、
  image Y=`0 mm`、camera Z=`-6.735 mm`、jaw offset=`0 deg`；
- 双目视频、时间戳、内外参、`5.31613697 mm` baseline、table frame；
- PSM URDF、mimic、所有本地 CAD mesh、两片 jaw contact link 和原始 q7 jaw；
- `37570` 粒子、`193845` tetra、`24196` surface faces、`2863` Gaussian
  的自适应组织资产及四粒子绑定；
- 左右组织 mask、双目器械 part mask、关键 runtime 源码和当前固定材料/接触
  常数。

左右离线视频各有 `1441` 帧。双目观测采用最近时间戳和严格 `20 ms` 门槛；
左帧 `86,158,240,716,738,1108,1213,1440` 超限并显式排除，剩余 `1433`
帧可用于阶段 B。冻结的操作时刻为首次接触候选 `270`、闭合 `548`、牵拉参考
`700`、最大回撤 `906`、释放完成 `1276`、回弹参考 `1300`。

输入哈希、时间单调性、双目标定、pose driver 状态数、jaw 未被视觉修改、
两片接触 jaw、组织拓扑/Gaussian 绑定和 ThinLinc 主入口共 14 项门禁全部通过。
原始 `28 GB` bag 的完整 SHA256 复用既有 raw-kinematics 报告并重新检查大小，
其他文件均在本轮重新哈希。

复验：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/freeze_super_tissue_calibration_inputs.py --verify-only
```

下一步严格进入阶段 B：只从冻结双目范围生成剔除器械遮挡后的可见组织表面
观测；视觉力、残差映射和刚度优化仍不在本阶段启用。

## 论文式组织软化标定：阶段 B 完整表面观测（2026-07-30）

按照 `组织软化.md` 的阶段 B，本轮完成全部 `1433` 个合格双目时刻的组织表面
观测，而不是只做关键帧 smoke。新增：

```text
scripts/build_super_tissue_surface_observations.py
scripts/postfilter_super_tissue_surface_observations.py
scripts/build_super_tissue_keypoint_tracks.py
scripts/validate_super_tissue_surface_observations.py
```

完整稠密基线位于
`data/super/tissue_calibration_v1/stage_b_dense_baseline/`：FoundationStereo
对 `1433/1433` 帧完成推理，0 缺失、0 失败、0 配置混入，每帧
`5135..6058` 点。正式观测位于
`data/super/tissue_calibration_v1/stage_b_surface_observations/`：在左目组织
mask、器械遮挡剔除和 table/world 变换基础上，把保存的三维点按标定视差
重投影到右目，只保留右目组织内且不在右目器械 mask 中的点。正式序列仍为
`1433/1433` 帧，每帧 `4749..5597` 点，右目语义保留率
`90.03%..94.07%`，产物有有序摘要 SHA256。

左右网络一致性使用覆盖完整时段的 `64` 个等距样本加全部操作关键时刻，共
`69` 帧，输出在
`data/super/tissue_calibration_v1/stage_b_lr_stratified_audit/`。`69/69`
帧通过 `1.5 px` 左右误差门禁，每帧 `4751..5487` 点，有效可见比例
`92.73%..96.48%`。反向 FoundationStereo 推理只作为观测质量审计；论文式
材料标定使用的正式表面仍覆盖全部帧。

夹取区关键点在 `270..1300` 帧使用前后向 LK、组织 mask 和器械排除区跟踪。
共创建 `186` 条 tracklet，选择最长的 `32` 条；长度 min/median/max 为
`731 / 896 / 1031` 帧。`28428` 个二维观测中 `27521` 个关联到正式三维
表面，三维关联率 `96.81%`。这些关键点只作为组织状态损失，不能修改阶段 A
冻结的 PSM 轨迹。

统一报告
`data/super/tissue_calibration_v1/stage_b_validation_report.json` 的
`17/17` 门禁全部通过：

- 全部 `1433` 帧点数、有限性、`20 ms` 同步、左右组织/器械语义和坐标门禁
  无异常帧；
- 正式表面相对完整稠密基线的逐帧双向 Chamfer p95 最大值为
  `0.554 mm`；
- 关键时刻正式表面相对严格左右审计的 Chamfer p95 最大值为
  `0.240 mm`；
- 最大回撤抬升 `9.298 mm`，释放下降 `9.523 mm`，回弹高度误差
  `0.429 mm`；
- 初始观测到冻结 rest top surface 的距离 p95 为 `0.743 mm`；
- 32 条关键点长轨迹及 `96.81%` 三维关联率通过。

本阶段没有启用视觉力、残差映射或刚度更新，也没有修改 GUI runtime。下一步
严格进入阶段 C：读取同一冻结 PSM 轨迹和正式表面观测，建立关闭视觉力、
关闭逐帧深度残差、关闭在线刚度更新的确定性接触回放。

## 论文式组织软化标定：阶段 C 完整确定性离线回放（2026-07-30）

阶段 C 已完成冻结视频全序列的无视觉力物理回放，不是关键帧或短时 smoke。
新增：

```text
scripts/replay_super_tissue_calibration_stage_c.py
scripts/validate_super_tissue_calibration_stage_c.py
```

正式产物位于：

```text
data/super/tissue_calibration_v1/stage_c_offline_replay/
```

回放覆盖左帧 `0..1440` 共 `1441` 帧，固定物理时钟 `60 Hz`、每步 `12`
子步，共 `2888` 个物理步。每个物理目标时刻按照阶段 A 契约，对冻结的
`5458` 状态 pose driver 使用 ZOH，并应用
`self-spin=-9.184 deg`、image X=`+0.332 mm`、image Y=`0`、
camera Z=`-6.735 mm`、jaw offset=`0`。只有两片夹爪参与 triangle-skin
接触；轴杆、粒子—shape、粒子—粒子、持久 grip/follow 均关闭。视觉力、
逐帧深度 residual 和在线 stiffness optimization 均关闭；Gaussian 四粒子
tetra 绑定保留。

短接触探针发现原 CUDA 材料 Jacobi 与 jaw contact 的浮点原子加法会在接触后
产生非确定求和顺序：无接触段逐字节一致，但旧实现到帧 300 可分叉约
`0.56 mm`。没有放宽“确定性”定义或增加状态量化，而是改成固定顺序归约：

- 四面体各自写四个 corner contribution，再按预计算的
  `particle -> sorted tet corner` 邻接顺序求和；
- jaw sample 各自写三个 barycentric contribution，再按唯一的
  `particle id + contribution id` key 排序求和。

这不修改能量、接触修正、摩擦或边界条件。修改后 300 帧接触探针以及正式
1441 帧双运行均逐字节一致。

正式结果：

```text
full replay frames                         1441 / 1441
fixed-clock physics steps                  2888 per run
tissue particles                           37,570
tissue tetrahedra                          193,845
all / soft Gaussians                       8,318 / 2,863
contact-active frames                      435
first / last active left frame             184 / 1284
maximum contact count                      3,503
maximum tissue displacement                6.762 mm
maximum tissue speed                       0.190 m/s
maximum inverted tetrahedra                0
maximum fixed-anchor drift                 0 m
particle trajectory repeat max error       0 m
Gaussian mean/quaternion repeat max error  0 / 0
contact integer/float mismatch             0 / 0
7 landmark float/u8 render mismatch        0 / 0
runtime gates                              23 / 23 PASS
artifact validation                        PASS
```

第一遍保存全部粒子位置、全部 Gaussian means/quaternions、逐帧接触/稳定性
指标和 7 个关键时刻的真实渲染；总量约 `940 MiB`。第二遍逐帧与第一遍比较。
`stage_c_artifact_validation.json` 又独立核对三个大数组的文件 SHA256、全部
逐帧状态 SHA256 以及渲染 NPY/PNG，一项不缺。

当前固定初值 `E=50 Pa, damping=30/s` 在闭合后保留约 `6.762 mm` 形变，
牵拉参考帧已无接触；这与阶段 B 的继续牵拉、释放下降和回弹观测不一致。
阶段 C 的职责是把该失配稳定、可复现地暴露出来，而不是提前调参。

复验已有产物：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/validate_super_tissue_calibration_stage_c.py
```

下一步严格进入阶段 D：视觉力和 residual mapping 继续关闭，固定器械轨迹、
接触、泊松比和边界条件，只对全局杨氏模量与速度阻尼做对数粗搜索和局部搜索。

## 论文式组织软化标定：阶段 D 全局材料参数（2026-07-30）

阶段 D 已完成第一版完整离线材料标定。新增：

```text
scripts/calibrate_super_tissue_material_stage_d.py
scripts/validate_super_tissue_material_stage_d.py
```

正式产物位于：

```text
data/super/tissue_calibration_v1/stage_d_material_calibration/
```

标定器直接复用阶段 C 的冻结 `60 Hz / 12 substeps` 因果调度、绝对 PSM
轨迹和确定性求解器，不重跑 FoundationStereo，也不保存每个候选的 GiB 级完整
轨迹。每个候选只在固定评估帧传回组织粒子，并写独立 JSON 报告。两张 A800
各自持有一个 Warp 环境并并行处理候选；第二张卡显式绑定
`wp.set_device(cuda:1)` 和 `torch.cuda.set_device(1)`，避免默认当前设备仍为
`cuda:0` 的跨卡 kernel launch。

只改变：

- 全局 Young's modulus；
- particle velocity damping。

固定不变：

- `nu=0`、gravity=`0`；
- jaw friction、contact query/margin、接触求解和 no-flip；
- tissue fixed mask；
- 阶段 A PSM driver 与全部人工修正；
- visual force iterations=`0`；
- residual mapping=`OFF`；
- online stiffness optimization=`OFF`。

严格连续划分为 calibration `270..906`、validation `907..1276`、test
`1277..1439`。损失为 `0.70` 的 12 条 first-contact 长轨迹相对位移鲁棒
RMSE、`0.30` 的运动加权可见表面单向鲁棒距离，以及非有限/翻转/anchor drift
硬惩罚。选参只读取 calibration；validation/test 只用于对 calibration 排名
前三做未来预测，不反向泄漏。

搜索和复验：

```text
log coarse candidates                    13
local log candidates                      8
full future finalists                     3
dense deterministic verification          2
total physical replays                    26
dual-A800 wall time                    909.5 s
```

正式选中 `E=100 Pa, damping=5/s`。数值结果：

```text
selected calibration objective       1.872780 mm
selected validation objective        2.667120 mm
selected test objective              1.287354 mm
frozen E50/d30 calibration objective 1.931738 mm
relative improvement                    3.052 %
keypoint calibration robust RMSE      2.068545 mm
surface calibration robust RMSE       1.415996 mm
maximum displacement                  5.903613 mm
maximum speed                         0.320288 m/s
minimum tet volume ratio              9.869303e-5
maximum inverted tetrahedra                    0
maximum anchor drift                           0 m
dense checked frames                1440 / 1440 per run
repeat metric mismatch                         0
landmark/final particle hash mismatch          0
artifact validation                    16 / 16 PASS
```

前三个 full finalist 的 validation 仍由 calibration 最优最低；test 上另一个
候选略低，但没有用 test 重新选择材料。当前全局参数相对初值只改善约 `3.05%`，
且搜索表面存在多个相距较远但在最优 `10%` 内的候选，所以该结果是可复现的
第一版等效全局材料基线，不宣称为唯一真实组织参数。

已有产物的无 GPU 复验：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/validate_super_tissue_material_stage_d.py
```

下一步严格进入阶段 E：固定 `E=100 Pa, damping=5/s`，保持 PSM/contact/
boundary 不变，加入论文式仅作用于可见组织表面的 residual mapping；物理预测
状态、残差修正状态和残差分布必须分开保存。

阶段 E/F 已完成：残差映射平均 surface RMSE 改善 30.43%、P95 改善 35.59%，
两次回放 byte-exact、无翻转单元；局部刚度 5 候选均通过，选中区域倍率
primary 1.5 / transition 0.75 / far 1.0。详见 stage_e_residual_mapping 与
stage_f_local_stiffness 产物目录。

阶段 G：已导出冻结 profile，并接入 GUI 启动流程。GUI 默认加载
`stage_g_calibrated_profile/tissue_calibrated_profile.json`，应用 E=100 Pa、
damping=5/s 和平滑局部刚度；在线 residual/visual force 默认关闭，后续做 A/B。
实际 ThinLinc 入口已同步：`bash scripts/run_demo_thinlinc.sh` 现在显式传入该
profile（也可用环境变量 `CALIBRATED_PROFILE=/path/to/profile.json` 覆盖）。

### 持久夹持与局部安全回退（进行中）

确认原 Stage-C 在约 frame 382 触及 `1e-4` 体积门后，全局 scale=0 会冻结整个
组织。现已实现双 jaw 持续接触触发的柔顺 attachment：连续 3 子步双侧接触后
记录粒子相对 jaw 的局部坐标，以每子步最大 0.1 mm、relaxation 0.35 跟随；
jaw 间距比捕获时增加 1.5 mm 后释放。材料安全门新增四轮局部 one-ring 回退，
全局回退仅作最终兜底。

ThinLinc 使用 Stage-A 冻结的人工修正，而不是旧轨迹：driver 为
`raw_paper_lnd_sam2_dense_contact_closedjaw`，roll `-9.184 deg`，相机平移
`[+0.332, 0, -6.735] mm`。CPU 约束门禁已验证激活、跟随和张开释放；完整
GPU 接触段与 1440 帧无翻转/恢复验证待 CUDA 权限可用后执行。

## 器械位姿临时默认切换：22 锚点、XYZ 无硬边界（2026-08-01）

按当前检查要求，`scripts/run_demo_thinlinc.sh` 的临时默认器械位姿已切换为：

```text
raw_paper_lnd_sam2_dense_contact_unbounded_xyz
```

脚本附加 `roll` 和相机平移同时归零，避免与位姿 driver 的双目 SE(3) 重复
叠加。该版本使用 22 对双目锚点、3 倍人工尖端权重、完整 1631 对时变
SE(3)，XYZ 没有硬边界，原始 q1–q7 位级不变。优化、GUI registration 与
CUDA runtime smoke 均通过。旧 `closedjaw`、`se3_only` 及全部原始基线未被
覆盖，可继续显式切换。

当前入口：

```bash
bash scripts/run_demo_thinlinc.sh --visual-force-iterations 1
```

注意：Stage A–G 组织材料参数是在旧的冻结 `closedjaw + roll/translation`
轨迹上标定的；本次仅临时切换 GUI 器械位姿，没有重新声称旧组织标定对新轨迹
仍为最优。若将本版升级为正式物理基线，需要重新冻结接触轨迹并复验组织标定。

## 受重力组织的 rest-shape 结论（2026-08-02）

- 当前重建组织是已经受重力的观测平衡态 `X_obs`；直接把它当零应力 rest shape
  再施加 `-9.80665 m/s^2` 属于重复加载，20 步会平均下沉约 `4.14 mm`。
- 当前 `paper_soft` 暂时恢复 `gravity=0`，20 步平衡门禁的最大动态节点位移为
  `0`、最低四面体体积比为 `0.999998`。
- 最终方案不是永久关闭重力，而是离线逆求隐藏的材料参考形状 `X_rest`：运行时
  粒子/Gaussian 从 `X_obs` 初始化，四面体材料使用 `X_rest` 的 inverse rest
  matrix，使初始预应力平衡正常重力。
- 资产和 builder 需要拆分 `initial_positions`、`material_rest_positions` 与
  `skinning_reference_positions`；刚度标定完成后必须重新运行逆静力求解。

## 当前器械完整高密度小 Gaussian 资产（2026-08-02）

- 当前 ThinLinc driver `raw_paper_lnd_sam2_dense_contact_unbounded_xyz` 已切换到
  独立的 `psm_paper_lnd_surface_gaussians_dense_v2.npz`，旧 455 点资产未覆盖。
- 新资产使用约 `0.35 mm` 的表面采样间距，共 `76,798` 个 Gaussian：完整长杆
  `71,568`、末端腕部主体 `3,016`、左右夹爪各 `1,107`。
- Gaussian 切向尺度约 `0.203 mm`、法向尺度约 `0.051 mm`；四个 CAD link 的
  mesh vertex 到最近 Gaussian 最大距离不超过 `0.311 mm`。
- 生成器新增可选 URDF material color 读取；当前杆部/腕部和夹爪保留各自 CAD
  材质颜色，不再全部强制为同一种灰色。
- `example_embodied_super_offline.py` 和 `scripts/run_demo_thinlinc.sh` 默认使用
  `psm_visual_mode=full`，显示杆部、腕部和双夹爪。器械—组织物理接触仍只允许
  两片夹爪，杆部 Gaussian 仅用于外观，不会扩大接触范围。
- CPU 完整环境门禁已确认加载 `76,798` 个器械 Gaussian，同时保留组织
  `2,863` 个四面体蒙皮 Gaussian；组织 20 步平衡和 q7 抓取/释放门禁全部通过。

### 首帧双目真实颜色（2026-08-02）

- 当前 76,798 点器械 Gaussian 几何不变；颜色改由第 0 个已人工确认的严格同步
  双目对直接烘焙，不再使用 URDF/CAD 材质色。
- 左、右目分别结合 shaft/distal SurgicalSAM2 掩码、朝向过滤和点 z-buffer
  取样；双眼都看到的粒子取双目 RGB 中位值，只被一眼看到的粒子保留该眼 RGB。
- 首帧左目直接观察 2,730 点，右目直接观察 2,228 点，双眼共同观察 1,933 点，
  并集直接观察 3,025 点。
- 长杆 `PSM1_tool_main_link` 共 71,568 点，首帧双目直接观察 826 点；其余
  70,742 个首帧双目不可见点按要求置为纯黑，不使用后续帧或 CAD 色补全。
- 腕部和两侧夹爪的首帧双目不可见面，仅在各自刚体 link 内从最近的首帧可见点
  传播颜色，避免运动后出现无色孔洞。
- 当前 GUI 资产为 `psm_paper_lnd_surface_gaussians_dense_v2_first_stereo_rgb.npz`；
  烘焙脚本为 `scripts/bake_super_psm_gaussian_colors.py`，报告和双目投影预览与资产
  同目录。

## 组织 Gaussian 忠实增密与多帧遮挡补全（2026-08-02）

- 原始 `2,863` 个组织 Gaussian 的权威生成方法已追溯到
  `scripts/build_super_bodies_from_first_frame.py`：第 0 帧 FoundationStereo
  深度先做 `1 mm` voxel downsample，再随机选择最多 `3,500` 个表面初值，调用
  `SimpleBodyBuilder._grow_gaussians` 以 RGB Gaussian-splatting MSE 联合优化
  means/opacity/color/quaternion/scale `600` 次；学习率分别保持
  `1e-4/1e-3/1e-2/1e-2/1e-2`，尺度约束为粒子半径的 `0.5--2.0x`，最后删除
  离组织物理表示过远的点。因此原资产的三轴尺度中值约为
  `1.142/0.991/0.984 mm`，不是从当前四面体三角面直接显示出来的粒子。
- 新增 `scripts/build_super_multiview_tissue_gaussians.py`，继续使用同一个
  `_grow_gaussians` 优化器、同样的五类参数、学习率、600 次训练、距离裁剪和
  四节点重心绑定；没有把视觉增密带入力学网格，也没有改刚度、质量、碰撞面或
  深度残差路径。
- 视觉初值改用术前静止段 6 个严格同步双目对融合出的无空间平滑表面。单独供
  Gaussian 使用的细表面间距为 `0.41 mm`，共 `27,063` 个顶点、`53,090` 个
  三角；力学仍使用原来的 `2,680` 节点、`7,810` tetra 和 `4,994` 个封闭 skin
  三角，且新旧资产的 26 个非 Gaussian 数组逐项完全相同。
- `MaskedPosedImageAndDepth` 已有的 `mask=2` 遮挡语义现在在原优化损失中真正被
  排除：器械、高光和无效深度不会被当成组织纹理，也不会被当成随机背景惩罚；
  现有只有 `0/1` 的旧训练保持与原 full-image MSE 完全一致。
- 新资产经过 12 张全分辨率图像训练后保留 `18,530` 个 Gaussian，是原来的
  `6.47x`；尺度限定为 `0.175--0.700 mm`，三轴中值约为
  `0.384/0.373/0.368 mm`。
- 第一个双目对直接观察 `12,386` 点；首帧双目均不可见、但后续静止帧实际可见
  并参与监督的点为 `2,183`。资产显式保存
  `gaussian_first_pair_observed`、`gaussian_later_only_observed` 和
  `gaussian_observation_count`，因此遮挡补全是可审计的真实多帧支持。
- 当前 runtime 已切换到
  `paper_pbd_tissue_v3_dense_multiview/tissue_paper_pbd_dense_multiview.npz`。
  独立拓扑/质量/绑定验证与完整 CPU 场景回归全部通过，20 步参考平衡最大动态
  位移仍为 `0`、最低 tetra 体积比为 `0.999998`。

## 取消 1.5 mm 深度硬裁剪并修复 Gaussian 空洞（2026-08-02）

- `1.5 mm` 不是论文或四面体绑定的要求，只是旧 v3 构建中的离群点清理阈值。
  由于双目深度本身存在系统误差，而且后续还要做视觉力修正，不能把“距力学
  collision skin 超过 1.5 mm”直接等价为错误点。
- 四节点绑定可为组织外部 Gaussian 保存 `gaussian_rest_offset_table`，运行时该
  偏移随绑定 tetra 的极分解旋转更新。因此表面外点仍可精确重建和随组织运动，
  没有必须按 1.5 mm 删除的技术限制。
- v4 不再按表面距离删除任何有限 Gaussian。RGB 优化后完全失去 12 个视图支持
  的位置不删除，而是退回其自己的多视图融合初始位置；有视图支持的点即使超过
  1.5 mm 也保留。每点的表面距离、最终/原始优化位移、观测次数和是否回退均写入
  资产，供后续视觉力置信度使用。
- v4 保留全部 `27,063` 个细表面顶点，包括 `225` 个拓扑补洞点；`0` 点被距离
  裁剪，`11,092` 个无支持漂移位置回到初始表面。超过 `1.5 mm` 的 `1,200` 点
  全部具有图像观测支持。
- 细表面顶点到最近 Gaussian 的 p95 距离由 v3 的约 `1.62 mm` 降到
  `0.70 mm`，最大值由约 `3.40 mm` 降到 `1.43 mm`。尺度仍为
  `0.175--0.700 mm`，没有通过大幅膨胀 Gaussian 掩盖细节。
- 当前 runtime 已切换到
  `paper_pbd_tissue_v4_dense_multiview_uncropped/tissue_paper_pbd_dense_multiview.npz`；
  四面体、碰撞三角面、粒子质量、材料和约束保持不变。

## 物理网格、GauSTAR 面绑定与大变形方案（2026-08-05）

### 已确认的当前结构

- 当前组织是“体内四面体 PBD + 四面体边界三角碰撞面”：`2,680` 个 PBD
  粒子、`7,810` 个 tetra、`4,994` 个表面/碰撞三角形。表面三角形不是第二套
  独立力学网格，而是四面体边界面，三个顶点直接引用四面体 PBD 粒子。
- Gaussian 是视觉表示，不是额外的 PBD 粒子。当前每个 Gaussian 绑定一个四面体，
  保存四个粒子索引、四个重心权重和 `rest_offset`；位置和朝向随四面体变形更新。
- 当前视觉力也沿相同的四节点重心权重回传到 PBD 粒子，因此视觉层和物理层的
  绑定关系是一致的。

### 与论文的对应关系

- 论文同时讨论 thin-shell 和 volumetric 两种组织。thin-shell 使用单层表面粒子；
  volumetric 使用四面体网格，并使用距离、体积和形状匹配约束。当前组织对应论文
  的 volumetric 路线，不是只有三角网格的路线。
- 论文初始化后不进行 remeshing；固定的是网格拓扑，不是节点的世界坐标。粒子位置
  变化后，边长、三角面面积和四面体形状都可以变化，因此固定拓扑仍然可以拉伸、
  弯曲和剪切。

### GauSTAR 面绑定的评估结论

- GauSTAR 的方法是把 Gaussian 绑定到表面 mesh face，适合表面跟踪和拓扑变化；它
  本身不是内部体积 PBD 求解器，不能直接替代当前四面体物理网格。
- 可以将 GauSTAR 式三角面绑定作为一个可选的视觉绑定模式：每个 Gaussian 保存
  三角面 ID、三个顶点的重心坐标以及面局部偏移。
- 如果同时改变视觉力回传，力将从当前的“四节点回传”变成“三节点表面回传”。这
  会使力更集中在表面，可能改变夹爪附近的局部变形、抖动和抓取稳定性。
- 因此下一步不直接替换当前绑定，而是保留四面体绑定作为基线，新增可切换的
  `surface_face_binding` 实验模式，并比较渲染覆盖、reset 稳定性、抓取变形和视觉力。

### 大变形下一步方案

- 当前拓扑固定但几何可变，有限刚度允许组织拉长；过大拉伸时主要风险是四面体
  畸变或接近翻转，而不是“网格不能变长”。
- 当前 Gaussian 中心会随四面体移动，但 Gaussian scale 暂时保持参考尺度。后续
  可根据四面体变形梯度的奇异值更新 Gaussian scale，以表现局部拉伸；这会影响
  渲染覆盖和视觉残差，但不直接改变现有视觉力散射公式。
- 只有在固定拓扑无法承受目标变形、出现严重单元畸变时才考虑自适应 remeshing。
  remeshing 必须同时转移 PBD 粒子位置/速度/质量、距离和形状约束、Gaussian 绑定、
  碰撞面索引及抓取状态，不能只替换表面三角网格。

## 单双目并集重建与 GauSTAR 式表面面绑定（2026-08-06）

- 新的 Gaussian 细表面使用 `dense_union`：只要 FoundationStereo 在某一只相机中
  给出有限且可见的深度就保留，不再用左右一致性、LR/RAFT 置信度或两视角高度
  分歧作为硬删除条件。融合后共有 `27,103` 个点，其中左目独有 `1,444`、右目
  独有 `544`、双目共有 `24,910`、拓扑补洞 `205`；双目分歧的 `18` 个单元也保留
  并只作为审计数据。
- 新资产位于
  `data/super/grasp5_native/tissue_multiview_v1/paper_pbd_tissue_v5_gaustar_face_union/`。
  全部 `27,103` 个 Gaussian 均保留，未按深度置信度或距力学面距离裁点；runtime
  和 GUI 已切换到该 v5 资产。
- 视觉绑定由旧的四面体内部绑定改为 GauSTAR 式表面三角面绑定。每个 Gaussian
  保存三角面 ID、三个表面粒子索引和三个面内重心权重；中心先投影到物理表面，
  `rest_offset=0`。这里采用论文的面内重心坐标分布，不把同一面的多个 Gaussian
  强制合并到唯一几何中心，以免减少数量和丢失细节。
- 运行时位置由三角形三个顶点插值，旋转由当前三角面的切向/法向局部坐标系更新；
  视觉力也只通过这三个表面节点回传。为兼容现有四槽存储，面相邻 tetra 的第四个
  角仍存在于数组中，但权重严格为零，不会收到该 Gaussian 的视觉力。
- 资产专项验证确认：所有面 ID、顶点对应、重心权重、相邻 tetra 关系和零偏移均
  有效，静止面重建最大误差为 `4.74e-9 m`。完整 CPU 场景回归通过；20 步参考
  平衡最大动态位移为 `0`，最小 tetra 体积比为 `0.999998`。

## 抓取区自适应四面体、碰撞面与面中心椭球 Gaussian（2026-08-06）

### 本轮冻结目标与分层原则

本轮只重建组织几何、物理拓扑、碰撞拓扑和 Gaussian 蒙皮；没有重新做刚度
优化或深度 residual mapping。用户要求的空间分层被实现为两套相互独立但绑定的
表面：

1. 物理层：中心抓取区密集四面体及其真实边界碰撞面，过渡区中密，外围稀疏；
2. 视觉层：中心最密、过渡中密、外围较稀的独立三角网格；每个视觉面中心一个
   椭球 Gaussian；
3. 视觉顶点嵌入新的物理边界面，视觉面/Gaussian 随物理节点形变；
4. 碰撞面只来自四面体真实外边界，不随 Gaussian 数量增加而伪造碰撞几何。

抓取区不是按组织包围盒中心猜测，而是从当前 GUI pose driver 中筛选
`jaw midpoint z <= 25 mm` 且 `q7[6] <= 0.10 rad` 的 `1,129` 个低位闭合样本，
得到 table frame 中位中心：

```text
grasp center x/y = [-2.368353, -3.481765] mm
sample time range = [18.209428, 41.738795] s
x p05/p95         = [-5.270636, -0.106128] mm
y p05/p95         = [-3.999267, -2.929571] mm
```

径向区域固定为中心半径 `20 mm`、过渡外半径 `35 mm`、其余为外围。物理目标
间距由中心表面 `0.82 mm` 到外围 `2.0 mm` 做几何插值，并随向内深度从表层间距
过渡到 `2.4 mm`；深度过渡终点为 `14.5 mm`。这样密度变化是连续的，不在区域
边界产生突跳。

### 被拒绝的物理候选与质量门槛

第一版物理 PLC 为保留视觉轮廓而冻结了过多外围边界点。TetGen 即使使用
`minratio=2`、`mindihedral=5 deg`，仍被约束侧壁强制出一个极薄四面体：最小
体积约 `5.72e-7 mm^3`、最大 rest matrix condition number 约
`2.04e6`。该候选没有接入 runtime。根因是侧壁上的三个冻结边界点近共线，单纯
收紧 TetGen quality flag 无法修复约束几何。

最终物理顶面使用 `fine/transition/outer stride = 2/3/4`，外围轮廓受控稀疏，
同时保持单一流形边界。生成器新增并冻结以下硬门槛：

```text
minimum tet volume          >= 1.0e-4 mm^3
maximum rest condition      <= 250
neighbor target ratio       <= 1.35
particle/tet/skin budgets   <= 50k / 220k / 40k
all rest volumes positive
closed skin and one tet component
```

最终物理资产：

```text
particles                              23,377
tetrahedra                            119,748
PBD unique edges                      151,010
closed collision triangles            15,772
  top / side / bottom          10,638 / 2,241 / 2,893
center / transition / outer nodes 13,002 / 8,490 / 1,885
center / transition / outer target-spacing median
                               0.881 / 1.252 / 2.017 mm
minimum tet volume                  4.856616e-4 mm^3
condition p50 / p99 / max       2.835 / 10.976 / 127.214
integrated tet volume / mass      42.246741 cm^3 / 42.246741 g
```

`particle_mass` 由 incident tetra rest volume 以 `1000 kg/m^3` 积分得到，不再
把重叠显示球体积相加。`pbd_edge_indices`、`pbd_rest_edge_length`、
`pbd_shape_cluster_indices`、`visible_surface_mask` 已写入最终资产。

### 视觉自适应面与保留单双目独有点

视觉网格继续以不做 LR 硬删除的 `dense_union` 表面为输入。视觉采样 stride 为
中心/过渡/外围 `1/2/3`，最大候选边长 `2.0 mm`。保留全部左目独有、右目独有
和补洞点。原 occupancy 轮廓有四处“洞边界与外边界只在一个顶点接触”；这里不
删除观测，而是为各 incident face fan 复制同坐标顶点，共复制 `4` 个，使边界
拓扑成为合法流形环。

```text
visual vertices                        13,583
visual triangles / Gaussians            26,754
  center / transition / outer     14,815 / 7,244 / 4,695
left-only / right-only vertices      1,444 / 544
bilateral / inferred vertices       11,390 / 205
dense input -> visual surface p95       0.0364 mm
dense input -> visual surface max        0.4515 mm
```

因此三个数量不再相等：`119,748` 是体内 tetra 数，`15,772` 是物理碰撞三角
面数，`26,754` 是视觉三角面/Gaussian 数。增加 Gaussian 或视觉面不会自动增加
碰撞面；本轮碰撞面之所以增加，是因为物理四面体本身也被重新加密。

### 面中心椭球绑定公式与 GauSTAR 对应边界

每个视觉顶点先投影到最近物理边界面。设其三个物理支持点为 `x_i`、重心权重为
`beta_i`，静止偏移为 `d = v(0) - sum_i beta_i x_i(0)`，则运行时：

```text
v(t) = sum_i beta_i x_i(t) + Bp(t) Bp(0)^T d
```

`Bp` 是支持物理面的切向/法向正交 frame。对视觉面 `f=(v1,v2,v3)`，Gaussian
中心严格固定为：

```text
mu_f(t) = [v1(t) + v2(t) + v3(t)] / 3
barycentric coordinate = [1/3, 1/3, 1/3]
```

椭球不被改成圆片，也不对三轴排序。令视觉面 frame 为 `Bv`、优化得到的静止
椭球旋转为 `R_g(0)`，运行时方向为：

```text
R_g(t) = Bv(t) Bv(0)^T R_g(0)
```

三轴 scale 当前保持优化值，不随形变梯度缩放。本实现是 GauSTAR 的“面附着 +
重心坐标 + 局部 frame”思想的一个明确特例：每个视觉面只取唯一中心
`[1/3,1/3,1/3]`。GauSTAR 原文可在同一三角形内按预定义重心坐标均匀布置
多个 Gaussian，所以不能声称两者整体算法或效果完全相同。

视觉属性从未裁剪 v4 椭球资产的最近 Gaussian 初始化，然后在 6 个严格同步双目
对（12 张全分辨率图）上固定中心，联合优化 RGB、opacity、quaternion 和三个
scale `600` 次。尺度范围 `0.175--0.700 mm`；没有强制球形：

```text
training loss                         0.008349 -> 0.001825
scale all-axis median                          0.361974 mm
anisotropy max/min p50 / p95               1.906 / 4.000
first stereo pair observed Gaussians              23,845
later-only observed Gaussians                       1,872
unobserved in selected projection audit             1,037
```

后 `1,037` 个没有按投影可见性删除，因为本轮原则是保留 dense-union 几何证据，
而不是把当前深度/遮挡判断再次作为硬裁剪。它们仍从最近的已有组织椭球获得有限
初值；`gaussian_observation_count` 保留供后续置信度使用。

### 视觉力、可变形能力与当前限制

- 四面体拓扑固定，但节点坐标、边长、面面积和 tetra 形状在 PBD/XPBD 中均可
  改变。独立验证对全体节点做 `x` 方向 `5%` 仿射拉伸，PBD edge 最大拉伸比
  为 `1.05`，全部 tetra 体积比为 `1.05` 且无翻转；因此“固定网格”不等于
  “不能拉伸”。真正限制是材料约束及过大变形导致的单元劣化/翻转。
- mode 2 的 Gaussian 位置与旋转严格走上述视觉面重建。当前视觉位移力仍使用每个
  Gaussian 最近物理边界面的三节点平移支持（存于兼容四槽数组，第四权重为零）；
  视觉顶点 rest-offset 旋转的完整 Jacobian 尚未进入力散射。刚度优化和深度
  residual 按用户要求留到后续，论文中必须如实说明这个近似。
- 当前观测形状是已经受重力后的平衡形状。为避免把重力重复加载，runtime 仍用
  `gravity=0` 的补偿基线。最终恢复真实重力仍需逆静力得到隐藏 material rest
  shape / initial prestress；本轮没有伪称已经完成该逆问题。

### 产物、验证与 GUI 状态

生成与验证脚本：

```text
scripts/build_super_adaptive_visual_surface.py
scripts/build_super_adaptive_tissue.py
scripts/build_super_visual_face_centroid_gaussians.py
scripts/validate_super_adaptive_centroid_tissue.py
scripts/test_super_paper_soft_mode.py
```

最终 runtime 资产：

```text
data/super/grasp5_native/tissue_multiview_v1/
  paper_pbd_tissue_v8_adaptive_centroid_ellipsoids/
    tissue_paper_pbd_centroid_gaussians.npz
    metadata.json
    asset_validation_report.json
```

独立验证全部通过：碰撞 skin 是 tetra 的精确闭合边界；PBD edges/lengths 和
shape clusters 与 tetra 一致；质量积分一致；视觉网格单一 edge-connected 且
边界流形；Gaussian 与视觉面一一对应；float32 面中心最大误差 `4.49 nm`；视觉
顶点物理嵌入最大重建误差 `3.90 nm`。

完整 CPU `paper_soft` 场景回归全部通过：

```text
20-step maximum dynamic displacement             0 m
20-step maximum anchor drift                      0 m
minimum tetra volume ratio                 0.99999851
q7 close capture / q7 opening release          PASS / PASS
depth residual / stiffness optimization        OFF / OFF
```

`PAPER_PBD_TISSUE_PATH` 已切换到 v8。ThinLinc GUI 已在 `DISPLAY=:14` 启动并
确认加载上述数量；VirtualGL 环境不能注册 CUDA--OpenGL interop buffer，自动
回退到 copy path。该提示不阻止 GUI 运行，但新物理网格比旧 v5 tetra 数约多
`15.3x`，实时速度必须以 GUI 实测为准；若速度不足，优先调整 material/contact
求解迭代或缩小精细 ROI，不应把 Gaussian 数与物理碰撞面错误地绑成同一数量。

## v4 接触标尺与 v8 密集网格的压入/抓取修正（2026-08-06）

### 失效原因与 v4 对照

当前 pose driver 在闭合段开始的 state `1807` 会把夹爪从非接触姿态直接放到组织
内部，初始最小 signed gap 为 `-5.490 mm`。v8 继续使用 v4 的窄局部支撑与仅表面
节点粘附时，密集小四面体无法吸收该跳变；局部防翻转会回退粘附位移。因此虽然
状态显示 `persistent_grip_active=True`，到 state `2500` 时绑定节点 z 位移中位数
实际为 `-3.586 mm`，对夹爪目标误差中位数为 `13.091 mm`，视觉上等价于没有
夹起。

在完全相同的 `1807 -> 2500` 轨迹上，用户认可的 v4 粗物理网格定量结果为：

```text
v4 particles / tetrahedra / skin faces       2,680 / 7,810 / 4,994
v4 attached-node z displacement median                 +9.570 mm
v4 attached-node target error median                    2.362 mm
v4 maximum tissue displacement                         14.662 mm
```

这说明差异不是 Gaussian，也不是视觉面和物理面错位，而是网格加密后抓取约束没有
按物理影响体积做分辨率补偿。仅把单个粘附节点的 correction 放大仍会触发局部
no-flip 回退；诊断性关闭回退虽然可增加位移，却产生 `1,106` 个倒置 tetra，已
明确拒绝，未写入 GUI 配置。

### 最终压入与抓取策略

最终实现仍只用两片夹爪的真实三角代理接触；杆部不参与组织碰撞。夹爪表面按
`0.5 mm` 间距采样，查询物理碰撞三角 skin。压入阶段同时使用：

```text
triangle contact query distance                    10.0 mm
top-surface clearance                               0.4 mm
top support lateral radius / inward depth      3.0 / 8.0 mm
contact correction                         2 passes x 0.10 mm max
physics substeps / material passes                 12 / 8
```

top barrier 使用有方向的顶面三角形；当夹爪从视觉顶面上方进入时，将接触修正分配
到顶面下方的局部四面体节点，所以先表现为“压下去”，而不是允许夹爪从表面穿过去。
四面体 local no-flip 始终开启。

持久抓取只有同时满足以下条件才建立：

1. `q7[6] <= 0.10 rad`，或仍在闭合且已进入 `<= 0.15 rad`；
2. 左右夹爪各至少有 `8` 个接触样本；
3. 初始深穿透已被推出，最大穿透不超过 `2.7 mm`；
4. 条件连续满足 `3` 个 contact projection pass。

建立时记录 q7 值和时间戳。直接接触的表面节点存成对应夹爪局部坐标。为消除网格
分辨率依赖，另在 rest mesh 上预计算半径 `4 mm` 的 particle-to-skin 邻域；抓取
后做 `4` 级传播，使中心抓取区的一块有限体积共同随动。每级边权为：

```text
w_edge(d) = exp[-0.5 (d / 2 mm)^2]
w_path    = product of w_edge along the propagation path
```

直接节点权重为 `1`，远处节点自然衰减到零。抓取 correction 使用
`relaxation=1.0`、每 pass 最大 `0.5 mm`，之后仍通过四面体防翻转检查。它不是把
整块组织设成刚体，也没有改变固定拓扑；只是让同一物理抓取范围不随 tetra 变小而
退化成拽少数几个节点。

释放仍严格由记录的 q7 控制：相对抓取值增大超过 `0.08 rad`，或处于 opening 且
达到 `0.15 rad`，或张开到 `0.50 rad`，立即清除全部直接/体积粘附节点。

### 最终回归结果

v8 完整 Gaussian 场景、真实 pose driver、`1807 -> 2500 -> 4200` GPU 回归：

```text
state 1807 initial signed gap                         -5.490 mm
first solve: grip inactive, penetration                2.972 mm
second solve: grip active, penetration                 2.670 mm以内
direct / supported attached particles at capture       99 / 6,585
state 2500 direct-node z displacement median           +6.289 mm
state 2500 direct-node z displacement range     +5.604 .. +9.823 mm
state 2500 maximum tissue displacement                11.957 mm
inverted tetrahedra                                             0
minimum tetra volume ratio                           1.00162e-4
state 4200 q7                                          0.676898 rad
attachments after release                                      0
```

最终抬升幅度略软于 v4，但已恢复明确的“先压入、再夹住、随后抬起”，同时维持新
密集四面体不翻转。CPU kernel gate 另验证：深穿透拒绝建立、q7/时间戳记录、直接
节点跟随、体积节点距离衰减、q7 增量释放、释放后 attachment 清零，全部 PASS。

## 密集四面体压入时的全局静止修正（2026-08-06）

用户观察到压入到某一时刻后整块组织严格静止。逐步读取 material projector 后
确认这不是自然阻尼，而是旧的最终全局 no-flip fallback：任意一个 tetra 接近零
体积时，整块连通组织该 substep 的位移都乘同一个安全系数。

旧 `0.10 mm/pass` 接触步长在固定 state `1807` 的诊断为：

```text
step 1: maximum increment 3.679 mm, global scale 0.184097, unsafe tets 12
step 2: maximum increment 0.919 mm, global scale 0.000000, unsafe tets  6
step 3--6: maximum increment 0.000 mm, global scale 0.000000
minimum volume ratio at stop                         1.00015e-4
```

因此确实是 `6` 个临界单元把全体自由度锁成零。完全删除 no-flip 的诊断候选产生
`1,106` 个倒置 tetra；仅把最终缩放作用到局部节点的候选也产生约 `100` 个倒置
tetra。两者均已拒绝并恢复，不能作为论文或 GUI 的正式实现。固定拓扑的 tetra
体积不能无限穿过零，否则内外会翻转；“没有任何限制”在当前力学表示下不成立。

最终修正不删除安全器，而是把三角接触 correction 从 `0.10 mm/pass` 降为
`0.02 mm/pass`。该尺度相对中心区 `0.82 mm` 目标粒子间距足够小，使 5.49 mm
的初始姿态重叠在多次 substep 中逐步传给材料，而不是两步内压扁一个小 tetra。
抓取体积邻域、q7 门控和 `0.5 mm/pass` 持久跟随上限保持不变。

固定 state `1807` 连续 12 步回归：

```text
material safety scale, every step                       1.0
unsafe / inverted tetrahedra                            0 / 0
maximum per-step increment                 2.719 -> 0.046 mm
```

增量平滑衰减表示固定工具下逐渐达到力学平衡；不再是从某一步开始被全局强制成严格
零。真实 GUI 更接近连续 pose，而不是在人为固定的 1807 停留多步。采用一次初始
压入、随后以 stride `10` 走完 `1807 -> 2500` 的更保守连续回归：

```text
physics/pose steps                                      71
steps with material safety scale == 0                    0
minimum material safety scale                          1.0
direct-grip z displacement min / median / max
                                  7.689 / 7.920 / 11.540 mm
maximum tissue displacement                         14.955 mm
inverted tetrahedra                                       0
minimum tetra volume ratio                       8.08218e-4
```

GUI 的旧文本 `volume safety: OFF` 也已修正：现在实时显示
`material safety scale` 和 `unsafe tetrahedra`。正常应为 `scale=1.000, unsafe=0`；
若未来再次出现严格静止，可直接从面板判断是否又触发安全 fallback。

## 固定位姿偏移、留缝抓取、视觉力与性能修正（2026-08-06）

### 本轮目标与固定运行参数

本轮按用户要求把器械位姿加入世界坐标固定偏移，不增加新的 GUI 实时调参：

```text
PSM world X translation                         -1.0 mm
PSM world Z translation                         -1.5 mm
camera-frame manual translation              0 / 0 / 0 mm
minimum effective q7[6] jaw opening              0.12 rad
```

`scripts/run_demo_thinlinc.sh` 通过
`--psm-world-translation-mm -1 0 -1.5` 和
`--psm-minimum-jaw-opening-rad 0.12` 固定载入。位姿流原始闭合段并非停在
`q7[6]=0`：state `1807` 为 `+0.032669 rad`，state `1810` 一度为
`-0.461774 rad`，稳定闭合段约为 `-0.099 rad`。负值会让两片夹爪继续越过几何
闭合位。现在机器人关节状态、LND link pose、碰撞代理和抓取 q7 信号统一使用
`max(q7[6], 0.12)`；不是只把视觉 mesh 张开。这样允许两片夹爪之间保留一小段
真实组织，同时原始张开段不受影响。

持久抓取的 closed 门槛同步设为 `0.13 rad`，所以 `0.12 rad` 是有效的“已闭合但
留缝”状态；仍必须左右两片夹爪都有接触。建立抓取时记录这个有效 q7 和原始时间
戳，之后 q7 相对抓取值增大超过 `0.08 rad` 仍立即释放。由于新增 Z 偏移令初始
几何重叠从约 `5.49 mm` 增到约 `6.42 mm`，抓取建立前允许的最大瞬时 penetration
门槛由 `2.7 mm` 调到 `4.5 mm`；防翻转、双侧接触和 q7 门控均没有删除。

### “更软”不能等于允许四面体压成零

最初直接采用 `E=25 Pa, nu=0, damping=10/s`，虽然拉伸更软，但 `nu=0` 没有体积
耦合。密集中心网格中少数 tetra 被压到 `1e-4` volume-ratio 安全边界，随后任一
继续恶化的位移都会触发全局 no-inversion scale `0`，再次表现为突然静止。把抓取
correction 从 `0.5` 降到 `0.1 mm/pass` 仍会发生，说明根因不是单纯“抓取太硬”，
而是局部体积已接近退化。

最终正式材料与接触参数为：

```text
Young modulus                                      25 Pa
Poisson ratio                                      0.35
velocity damping                                  10 /s
material passes                       6 x relaxation 0.1
physics substeps                              12 per frame
surface contact correction                 2 x 0.01 mm
persistent-grip maximum correction             0.05 mm
contact substep stride                                  3
gravity (observed equilibrium compensation)        0 m/s^2
```

这里低 `E` 提供拉伸/剪切顺应性，`nu=0.35` 只保留适度体积响应，避免用少数塌陷
tetra 换取“视觉上很软”。固定拓扑仍可移动和拉伸；禁止的是 tetra 穿过零体积翻面。

### 完整抓取轨迹回归

正式配置在 A800 上对 state `1807 -> 2500`、stride `10` 的 71 个连续 pose/physics
步进行回归；使用本轮 `X=-1 mm, Z=-1.5 mm` 和 `q7>=0.12 rad`：

```text
capture state                                      1827
direct / volume-support attached nodes       153 / 3,158 at capture
final direct-node Z displacement
  minimum / median / maximum             8.814 / 8.920 / 12.506 mm
maximum tissue displacement                          14.733 mm
steps with material safety scale == 0                       0
minimum material safety scale                              1.0
maximum unsafe tetrahedra                                    0
minimum tetra volume ratio                          0.00396587
inverted tetrahedra                                           0
```

capture 当步只完成 4 级体积传播中的 2 级；后续 contact solve 继续传播到配置的 4
级，因此不能把 capture 瞬间的 `3,158` 当成最终支持节点上限。独立 persistent-grip
CPU gate 继续通过深穿透拒绝、双侧建立、q7/时间戳记录、直接和衰减支持节点跟随、
开爪释放及 attachment 清零。

### 视觉力的边界与运行频率

本轮按用户新决定恢复 tissue 视觉力，但不恢复 stiffness optimization 或 depth
residual。左右目 masked RGB 残差只对 `26,754` 个 tissue Gaussian 求梯度，再经
视觉面/物理面绑定散射到四面体节点；PSM 的 `76,798` 个 Gaussian 只参与遮挡和
渲染，不接收视觉力。视觉力能把图像中可见的组织轮廓和纹理向观测位置拉近；完全
被夹爪遮挡的内部组织没有直接像素证据，必须由双侧接触后的持久抓取负责留在夹爪
中间，不能声称视觉力能直接观测不可见组织。

旧 GUI 是每个物理步执行左右目 `3` 轮可微 Gaussian 优化。现在改为：

```text
visual optimizer iterations per update                 1
visual-force update interval                    6 physics steps
between updates                         reapply last bounded force
paused playback                                  visual force OFF
```

点击 Play 后下一物理步立即更新一次；Pause 后不再持续施加缓存力。刚度优化和深度
残差仍为 OFF。

### 卡顿的实测分解与优化

正式完整场景包含 `23,377` 节点、`119,748` tetra、`15,772` 碰撞面、`8,602`
夹爪表面 samples 和总计 `108,552` 个 tissue+PSM Gaussian。A800 warm-run 实测：

```text
physics without jaw contact median                    6.24 ms
contact every substep (old stride=1) median           56.46 ms
contact every 2 substeps median                       30.73 ms
contact every 3 substeps (final) median               23.96 ms
Gaussian deformable skinning median                    0.30 ms
one stereo visual-force iteration                      9.14 ms
```

因此主要物理瓶颈不是 Gaussian 绑定，而是 12 个子步内对密集三角 skin 做 8,602 个
夹爪采样查询、top-volume support 和接触投影；旧版又在每步额外做约 `3 x 9.14 ms`
双目可微渲染。最终 stride `3` 仍保留每 physics frame 四次 contact update，完整抬升
回归优于安全门槛，同时把 warm contact physics 从约 `56.46 ms` 降到约 `23.96 ms`。
视觉力 `1/6` 更新使其摊销约为 `1.52 ms/physics step`，而不是旧约 `27.4 ms/step`。

GUI 还需要绘制约 10.9 万个 Gaussian；当前 VirtualGL 无法注册 CUDA--OpenGL
interop buffer，会走 GPU/GL copy fallback，这部分仍会限制最终显示帧率。若后续还要
继续提速，应先优化 top-support 邻接表示或做接触 ROI，而不是减少中心物理网格、
删除防翻转或把视觉 Gaussian 重新等同于碰撞粒子。

诊断入口：

```bash
PYTHONPATH=src /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/diagnose_super_contact_offset.py --device cuda \
  --end-state 2500 --state-stride 10 --baseline-steps 3
```

## 更正：共轴运动学 + 组织阻力限位夹爪（2026-08-06）

> 本节明确取代上一节的 `X=-1.0 mm` 和 `q7>=0.12 rad` 两项结论。上一节保留
> 只是为了记录实验演变，不能再作为论文方法或当前正式配置引用。

### 正式偏移与废止项

当前 GUI 启动配置固定为：

```text
PSM world X translation                         -0.2 mm
PSM world Y translation                          0.0 mm
PSM world Z translation                         -1.5 mm
camera-frame translation                    0 / 0 / 0 mm
artificial minimum q7 opening                    REMOVED
```

`scripts/run_demo_thinlinc.sh` 现在载入
`--psm-world-translation-mm -0.2 0 -1.5`。GUI 中已经删除 roll、wrist、jaw 和
XYZ 位姿的实时滑块，只显示冻结后的位姿与夹爪执行器状态，避免观察过程中把正式
参数改掉。

### 为什么废止固定 `q7>=0.12`

固定最小开口会在命令层直接篡改机器人 q7：无论两爪之间是否有组织都不允许继续
闭合。这不代表组织抵抗力，也不能解释最终开口来自何处。因此已同时从 CLI、回放
控制器和 `apply_psm_lnd_pose()` 删除 minimum-opening 截断。原始逐时刻 q7 现在只
作为目标角，实际夹爪角由接触限力执行器决定。

### 严格铰链运动学

当前 paper LND URDF/运动学的夹爪定义为：

```text
shared parent             PSM1_tool_wrist_sca_shaft_link
joint origin                                      [0,0,0]
joint axis                                  local [0,0,1]
jaw link 1 rotation                               +q7 / 2
jaw link 2 rotation                               -q7 / 2
```

视觉 Gaussian body、碰撞代理和隐藏 articulation 都使用同一个实际 q7。对 pose
driver 的修正不是平移两片夹爪，也不是绕各自中心旋转，而是在 tracked parent
坐标中应用 `T_parent Rz(±Δq7/2) T_parent^-1`。验证结果：非夹爪 link 位姿误差为
0，左右夹爪对 `17 deg` q7 增量各旋转 `8.5 deg`，共用铰链 pivot gap 为
`0.0 mm`，轴平行误差小于 `8.3e-5 deg`。

### 组织阻力限位，而不是直接“不闭合”

三角 skin 接触核对每个 penetrating jaw sample 计算继续减小 q7 时的瞬时运动：

```text
v_close = -(sign / 2) * (hinge_axis x (sample - hinge_pivot))
closing_leverage = max(0, -dot(v_close, outward_surface_normal))
```

只有 `closing_leverage >= 0.25 mm/rad` 的点才算“阻止闭合”的接触，擦过表面或在
铰链附近的点不会触发限位。左右夹爪各至少 8 个这类点并且两侧 resistance proxy
均为正，才认定组织确实位于闭合路径内。单侧接触仍允许夹爪继续运动。

为避免“一碰到就完全静止”在视觉上仍像人为不闭合，执行器采用有限顺应行程：

```text
raw q7                         unmodified commanded angle
maximum closing speed                         2.4 rad/s
bilateral-contact compression travel              0.12 rad
opening command                             immediate release
```

首次双侧阻力出现后，夹爪仍沿正确铰链闭合最多 `0.12 rad` 以压缩软组织；达到该相对
行程且闭合命令仍更小时，实际 q7 才停住。这个停止角是
`q7_at_first_bilateral_contact - 0.12`，随接触发生时刻变化，不是固定绝对开口。
组织离开闭合路径则恢复运动；开爪命令立即使实际角跟随并清空限位/粘附状态。

持久抓取也改为只在 `closing_requested && closure_blocked_by_tissue` 时允许建立，
然后仍要求双侧接触并记录实际 q7 与时间戳。因为指定的 `Z=-1.5 mm` 会令图像对齐
后的器械进入重建表面包络，capture penetration 上限放宽到 `10 mm`，但每步粘附
修正仍严格限制为 `0.05 mm`，材料 no-flip 安全约束未删除。

### 实际姿态证据与测试入口

在正式偏移 `(-0.2, 0, -1.5) mm`、闭合开始前 state `1806`、原始
`q7=0.474841 rad`，未变形组织上的 closing-resistance 扫描为：

```text
left / right closure-resisting samples             282 / 321
left / right resistance proxy (m^2)       6.553e-4 / 5.885e-4
bilateral tissue block candidate                         YES
```

因此闭合命令到达后，执行器会先从约 `0.47484` 沿共轴铰链压缩约 `0.12 rad`，预期
在阻力持续存在时停在约 `0.35484 rad`；不是启动时预设这个终值。

验证入口：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/test_super_tissue_resisted_jaw.py

/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/validate_super_psm_gui_pose_offsets.py

PYTHONPATH=src:examples /Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/diagnose_super_resisted_jaw_scan.py --device cpu \
  --state-index 1806 --angles 0.474841
```

前两项 gate 当前全部通过：有限速闭合、单侧不假抓、双侧阻力后的 `0.12 rad` 顺应
压缩、阻力保持、开爪立即释放，以及 shared-pivot `±q7/2` 运动学均已验证。

## 刚度回调与 GUI 位置微调恢复（2026-08-06）

用户在 GUI 观察后认为 `E=25 Pa` 过软。正式 `paper_soft` Young modulus 已回调到
`50 Pa`；Poisson ratio `0.35`、`10/s` 阻尼、材料迭代、防翻转、接触投影和组织阻力
夹爪均保持不变。因此这是单独的材料刚度修正，不改变网格、Gaussian 绑定或运动学。

同时恢复 GUI 中器械位置的三轴人工微调：image X（向右，`±5 mm`）、image Y
（向下，`±5 mm`）和 camera Z（向远，`±30 mm`），并恢复 `Reset manual position`
按钮。它们是在冻结世界基准 `(-0.2, 0, -1.5) mm` 上叠加的相机坐标增量；面板同时
显示当前合成后的 world XYZ 和冻结基准。夹爪角度不恢复手工滑块，仍由原始 q7、
共轴运动学和双侧组织阻力执行器决定。

## 更正：原始 q7 完全闭合 + 严格 3--5 粒子抓取（2026-08-06）

> 本节取代“共轴运动学 + 组织阻力限位夹爪”中的阻力限位执行器，以及上一节末尾
> “由双侧组织阻力执行器决定”的表述。共轴铰链 `+q7/2`、`-q7/2` 的运动学结论、
> PSM 世界偏移 `(-0.2, 0, -1.5) mm`、`E=50 Pa` 和 GUI 三轴位置微调继续有效。

### 最终闭合与候选定义

GUI/回放现在直接使用 pose driver 的原始 `q7[6]`，不再由组织阻力修改实际夹爪角，
也不设置人为最小开口。夹爪因此会沿共同铰链正常闭合到数据中的目标角。

碰撞面仍由密集三角网格采样，一个物理节点可能被同一夹爪的许多 sample 重复命中。
为避免把一次擦碰误认为稳定抓取，节点必须在**同一次接触求解**中同时满足：

```text
left-jaw sample hits on the node                 >= 48
right-jaw sample hits on the same node           >= 48
particle inverse mass                               > 0
particle state                                    active
```

满足上述条件的节点组成“完整严格候选集合”。抓取规则为：

```text
candidate count > 5       reject the whole set; select none; keep closing
candidate count = 3..5    capture every node in that complete set once
candidate count < 3       do not capture
later contact solves      never append newly appearing candidates
support neighbors         disabled (radius = 0 mm)
```

因此这里不存在“候选很多时挑最好的 3--5 个”的排序、截断或随机采样。只有随着原始
q7 继续闭合、完整候选集合本身自然降到 3--5 个，才在该次 solve 建立抓取。抓取时
记录 q7 角和时间戳；其后 q7 相对抓取角增大超过 `0.08 rad`，或进入明确开爪状态时，
立即清空全部绑定。只有首次命中的 3--5 个直接节点跟随夹爪，其余组织通过固定四面
体连接产生连续变形，不再额外绑定数千个邻域支持节点。

### 真实帧扫描证据

在 state `1807`、PSM 世界偏移 `(-0.2, 0, -1.5) mm`、未变形组织上，使用上述
双侧各 `48` hits 门槛扫描闭合角：

```text
actual q7 (rad)             -0.10   -0.20   -0.30   -0.40   -0.46
complete strict candidates      19      15      11       4       3
capture result              reject  reject  reject  capture  keep existing
maximum penetration (mm)      7.90    8.51    9.13    9.72    9.90
```

在 `q7=-0.40 rad` 首次出现完整的 4 节点集合时立即抓取全部 4 个；下一次闭合到
`-0.46 rad` 虽然当前候选变为 3 个，也不会替换或追加节点。独立 CPU gate 同时验证
了 `>5` 全拒绝、4 个全抓取、无邻域支持节点、绑定节点跟随夹爪以及开爪全部释放。

验证入口：

```bash
PYTHONPATH=src:examples python scripts/test_super_tissue_persistent_grip.py

PYTHONPATH=src:examples python scripts/diagnose_super_resisted_jaw_scan.py \
  --device cpu --state-index 1807 \
  --world-translation-mm -0.2 0 -1.5 \
  --angles=-0.10,-0.20,-0.30,-0.40,-0.46
```

### GUI 全部手动位姿偏移恢复

按用户后续要求，GUI 不再只显示 X/Y/Z 平移，已恢复当前 pose-driver 明确支持且经过
运动学验证的全部 6 项手动偏移：

```text
roll (q4)                                     -180..180 deg
paper-LND wrist pitch (q5)                      -60..60 deg
jaw opening (q7)                                -30..30 deg
image X / image Y                                  +/-5 mm
camera Z                                         +/-30 mm
```

`Reset manual position` 只复位三轴平移；`Reset all manual pose offsets` 同时复位
roll、wrist、jaw 和三轴平移。默认值仍为命令行载入的 roll/camera baseline，当前
launcher 默认为 joint offsets 全零、camera offsets 全零，并保留独立的固定世界偏移
`(-0.2, 0, -1.5) mm`。jaw slider 是在记录的 q7 上叠加校准量，两片夹爪仍严格绕
共享局部 z 轴旋转 `+q7/2` 与 `-q7/2`；它不平移夹爪，也不改变严格 3--5 粒子抓取
规则。GUI 中候选与释放判定使用叠加后的 effective q7。

## 四表面节点抓取、局部体积安全与 GUI 零位姿偏移（2026-08-07）

> 本节覆盖并替代上文“同一物理节点双侧各 48 hits、候选必须自然降到 3--5”以及
> `(-0.2, 0, -1.5) mm` 固定世界偏移的旧实验结论。旧段落保留仅用于记录方案演进，
> 不代表当前正式实现。

### 当前正式抓取判定

当前方案参照项目组相关工作中“抓取附近少量表面粒子建立附件”的思路，但以下门控、
双层体积传力和 Warp 实现是本项目针对密集四面体组织的工程化扩展，不能表述为原论文
逐行复现：

```text
q7 gate                         原始/有效 q7 正在闭合或已经闭合
bilateral contact               两片夹爪各至少 8 个有效接触 sample
jaw-patch coherence             两侧接触斑质心间距 <= 6 mm
capture penetration             最大穿入 <= 10 mm
temporal sustain                连续 3 次接触 solve 均成立
direct particles                接触斑中点最近的 4 个动态 collision-skin 节点
attachment frame                两片夹爪共同父体 tool_wrist_sca_shaft_link（body 13）
release                         q7 比抓取角增大 > 0.08 rad 或进入明确开爪状态
```

四个节点只在首次捕获时写入绑定，后续接触检测不会替换它们。绑定局部坐标记录在共同
父体坐标系中，因此夹爪继续绕共同铰链闭合时，不会因选择左夹爪或右夹爪坐标系而把组织
横向甩动。松开门控会一次清空全部直接绑定节点。

### 双层局部传力与防翻转

直接锚点纠正采用零历史 compliant-XPBD 形式：

```text
compliance                      0.5 m/N
relaxation                      1.0
single-substep correction cap   0.50 mm
transfer layers                 2 tetrahedral adjacency layers
direct anchors                  exactly 4
extra persistent support        disabled (radius = 0)
```

“两层传力节点”不是额外粘在夹爪上的抓取节点；它们只接收四锚点本次位移的局部体积
扩散，使锚点 incident tetrahedra 不会独自承受全部拉伸。材料求解后的旧全局安全缩放
已删除，因为它曾把整块组织的步长缩到约 `3.05e-5`，产生肉眼可见的突然静止。

当前安全路径分为三部分：

1. 材料约束只对危险四面体的一环做局部修复，不再全局冻结组织。
2. 两层传力区先提出位移；对会把正体积压到 `1e-8` 以下且继续恶化的四面体，只回滚
   相关的移动支撑节点，最多迭代 8 次。
3. 四个直接锚点最后独立做 20 次二分的正体积安全线搜索。二分从 12 次增到 20 次，
   避免小于 `1/4096` 的合法位移被量化成完全静止。

为防止长轨迹中支撑区在锚点受限时反复累加未实现位移，支撑预松弛只使用上一物理子步
直接锚点实际允许的安全倍率；倍率为零时最多只会发生一次预松弛，不会持续向远端组织
注入能量。公开诊断指标新增：

```text
persistent_grip_support_safe_scale
persistent_grip_direct_safe_scale
persistent_grip_support_unsafe_tetrahedra
persistent_grip_safety_tet_count
```

### 定量回归与已知边界

独立门禁 `scripts/test_super_tissue_persistent_grip.py` 已验证：双侧接触斑不必共享同一
网格节点、固定选择最近四点、第三次持续接触才捕获、共同父体跟随、q7 开启立即释放。

在正式资产（`23377` 粒子、`119748` 四面体、`15772` collision triangles、
`108552` 总 Gaussians）上，局部修复版本在 state `1795--2000` 的代表性 GPU 回归曾
得到：夹爪目标 z 位移约 `3.40 mm`，四直接节点 z 位移约 `3.40--3.42 mm`，最终跟踪
误差约 `0.0003--0.0047 mm`，翻转四面体 `0`，最小体积比约 `1.26e-4`，接触物理
中位耗时约 `54.6 ms/trajectory step`。随后完整到 state `2500` 的实验发现支撑位移
可能在锚点受限后累积；因此当前正式代码加入“上一子步实际倍率”门控。该门控的合成
抓取/释放测试已通过，完整 `1795--2500` 数值回归仍应在论文定稿前重新冻结结果；不要
把上述 2000 帧数值误写成当前最终长轨迹结论。

### GUI 启动与零偏移

`scripts/run_demo_thinlinc.sh` 当前明确使用：

```text
PSM world translation offset    (0, 0, 0) mm
PSM roll offset                 0 deg
PSM camera translation offset   (0, 0, 0) mm
tissue mode                     paper_soft
pose driver                     raw_paper_lnd_sam2_dense_contact_unbounded_xyz
cameras                         stereo_left, stereo_right
instrument visual              full shaft + wrist + both jaws
```

因此旧的 `x=-0.2 mm、z=-1.5 mm` 不再由 launcher 注入。2026-08-07 已通过
ThinLinc `DISPLAY=:14`、VirtualGL `egl0` 启动当前 GUI。

## 局部压下传力与非冻结安全修复（2026-08-07）

> 本节记录当前正式的“夹爪压下”路径。它只处理尚未建立持续夹取时的几何接触；
> q7 门控、四锚点持续夹取和开爪释放仍使用上一节的独立逻辑，二者不能混写。

### 原问题与原因

旧实现虽然检测到了夹爪/三角表面穿入，但普通接触只修正直接命中的 collision-skin
节点，局部体积没有足够的邻域传力。同时，接触提案使用 `1e-4` 材料体积质量阈值做
单一全局缩放：只要一个很薄的边界四面体越过该阈值，整个压下区域的修正倍率就可能
接近零，视觉上表现为夹爪继续进入组织、周围组织不下沉，某一时刻组织又突然静止。

当前实现把普通压下改成如下路径：

```text
direct contact                    jaw triangle / oriented top-plane contact
contact spread                    1 tetrahedral adjacency layer
surface correction cap            0.015 mm / solve
contact iterations                2 / selected physics substep
contact substep stride             3 (12 substeps/frame, therefore 4 updates/frame)
local unsafe repair passes         8
normal material-quality threshold  local repair only
absolute non-inversion boundary    volume ratio 1e-8
```

直接命中的表面节点保留完整的有界接触位移；一层四面体邻接节点接收归一化的局部扩散
位移，因此夹爪压下一点时，附近的小块体积随之下沉，而不是只有一个表面节点被推进去。
普通压下在应用提案后最多做 8 次危险单元局部检查，只回滚导致危险四面体继续恶化的
已移动节点。材料求解也先做逐危险单元局部修复。最后仍保留 `1e-8` 的绝对正体积
线搜索作为兜底；该线搜索只阻止四面体真正翻面，不再用 `1e-4` 的形变质量阈值限制
正常压缩。GUI 新增显示 `contact spread layers / contact local unsafe tetrahedra`，用于
区分“邻域传力是否开启”和“是否真的触发局部安全修复”。

另修正了一个通用配置默认值：没有显式设置 oriented top mesh 时，
`top_barrier_shape_ids` 默认为空，而不是错误地把全部工具 shape 当成 top barrier。
这不会改变 `paper_soft` 的双夹爪 top-plane 设置，但恢复了通用三角皮肤接触的预期语义。

### 正式资产回归

零 PSM 世界偏移、state `1795--1830`、正式 `paper_soft` 资产的 GPU 回归得到：

```text
particles / tetrahedra / collision faces     23377 / 119748 / 15772
initial maximum penetration                  8.629 mm
3 mm press ROI nodes                         384
ROI nodes depressed by more than 0.1 mm      374 / 384
ROI median z displacement at state 1830     -6.496 mm
four-anchor capture state                    1807
final persistent grip                        active, exactly 4 direct anchors
inverted tetrahedra                          0
minimum tetrahedron volume ratio             8.62e-8
material global safety scale minimum         1.0
contact physics median                       54.55 ms / trajectory step
contact-free physics median                   9.37 ms / trajectory step
```

需要如实说明：当前记录位姿在接触刚启用时已经有约 `8.6 mm` 的几何重叠，所以无偏移
且禁止穿透时，组织必须产生数毫米的局部下沉来消除已有重叠；这不是 `0.015 mm` 单次
修正上限造成的突然跳变。如果最终视觉希望只压下很浅的一点，应从器械/组织初始相对
位姿或接触启用时机消除这段初始重叠，而不应重新引入让组织冻结的全局缩放。

验证入口：

```bash
PYTHONPATH=src:examples python scripts/test_triangle_skin_contact.py --device cpu
PYTHONPATH=src:examples python scripts/test_super_tissue_persistent_grip.py
PYTHONPATH=src:examples python scripts/test_super_paper_soft_mode.py
PYTHONPATH=src:examples python scripts/diagnose_super_contact_offset.py \
  --device cuda --world-translation-mm 0 0 0 \
  --start-state 1795 --end-state 1830 --state-stride 1 \
  --baseline-steps 1 --grip-max-correction-mm 0.50
```

## PBD 组织整体降密重建 v9（2026-08-07）

### 目标和分层原则

本次按要求将物理三角网格整体降低约两档，同时严格保持“物理层”和“视觉层”解耦：

```text
物理层    较稀疏四面体体网格 + 其闭合边界碰撞三角面
视觉层    原 v8 的高密度三角面中心椭球 Gaussian
运行绑定  视觉顶点重新嵌入新物理表面，Gaussian 仍位于视觉三角面中心
```

因此，物理碰撞面减少不会自动删减 Gaussian，也不会把椭球重新变成圆球。原 v8 的
`RGB / opacity / quaternion / anisotropic scales` 逐项复用，仅重建视觉顶点到新物理
边界的嵌入和力映射绑定。

### 物理表面和体网格参数

正式物理顶面为：

```text
rest surface
  data/super/grasp5_native/tissue_multiview_v1/
  rest_surface_adaptive_physics_v4_coarse/rest_surface.npz

surface decimation strides       center / transition / outer = 4 / 4 / 4
maximum retained edge            3.5 mm
grasp-centered radii              20 mm / 35 mm
top PLC vertices / faces          2133 / 4038
top face zones                    1446 / 1639 / 953
boundary loops                    one outer loop, 226 vertices
duplicate XY boundary pinches     0
filled inner holes                0
top face area min / p50 / p95     0.0841 / 1.3839 / 1.6612 mm^2
source surface coverage p95       0.130 mm
```

体网格采用 TetGen 约束 PLC 四面体化：

```text
surface spacing                   1.5 mm
dense depth                       1.5 mm
transition end depth              14.5 mm
deep spacing                      3.5 mm
lateral outer spacing             3.0 mm
TetGen minimum ratio              1.5
TetGen minimum dihedral           10 deg
minimum accepted tet volume       1e-4 mm^3
maximum accepted rest condition   250
```

正式物理资产：

```text
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v9_coarse_physics/tissue_soft_adaptive.npz
```

与旧 v8 所用物理网格对比：

```text
                                      old v8       new v9       new / old
particles                              23377         6959          29.8%
tetrahedra                            119748        31044          25.9%
collision boundary triangles          15772         7420          47.0%
visual triangles / Gaussians           26754        26754         100.0%
```

新网格为单一连通体，闭合边界是精确的四面体外边界；没有退化单元。最小四面体体积
`0.000462 mm^3`，最大静止矩阵条件数 `153.36`，均通过配置门槛。

### 被拒绝的更粗候选

曾测试 `4/6/8`、`4/5/6`、`3/5/6`、`4/5/5`、`3/5/5`、`4/4/5` 和 `3/4/5`
等更激进组合。部分候选在外边界产生重复 XY pinch 或小内孔，另一些虽然拓扑闭合，
但 outer stride 5 会保留一个面积仅 `0.0197 mm^2` 的近共线 PLC 三角面，强制产生
最大条件数约 `436` 的劣质四面体。该问题不能靠放宽 TetGen 参数可靠修复，所以没有
为了追求更低计数而接受。正式 `4/4/4` 是通过拓扑、覆盖和四面体质量门槛的最粗候选。

### 视觉外观复用和绑定

正式运行资产：

```text
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v9_coarse_centroid_ellipsoids/
tissue_paper_pbd_centroid_gaussians.npz
```

其视觉网格仍为 `13583` 顶点、`26754` 三角面，并严格保持一面一个 Gaussian：

```text
Gaussian center                    (v1 + v2 + v3) / 3
visual-face barycentric weights    (1/3, 1/3, 1/3)
fine / transition / outer faces    14815 / 7244 / 4695
left-only / right-only vertices    1444 / 544
later-frame-only observations      1872
anisotropy p50 / p95 / max         1.906 / 4.0 / 4.0
```

独立验证确认：视觉面中心为 float32 精确重建，视觉顶点可从新物理表面嵌入重建，椭球
各向异性和单位四元数保留，双目单侧证据和后续帧补洞证据保留。运行入口
`PAPER_PBD_TISSUE_PATH` 已切换到 v9。

### 运行时回归

已通过：

```text
scripts/validate_super_adaptive_centroid_tissue.py
scripts/test_super_paper_soft_mode.py --device cpu
scripts/test_super_tissue_persistent_grip.py
```

零位姿偏移、state `1795--1830` 的 GPU 真实轨迹回归：

```text
capture state                              1807
captured direct nodes                      exactly 4
3 mm press ROI nodes                       81
ROI depressed beyond 0.1 mm                75 / 81
final persistent grip                      active
inverted tetrahedra                        0
minimum tetrahedron volume ratio           2.46e-4
physics without contact median             7.95 ms / trajectory step
physics with contact median                53.40 ms / trajectory step
Gaussian skinning trajectory median        0.75 ms / trajectory step
```

碰撞面和四面体降密显著减少了资产规模，但接触阶段耗时只小幅下降。原因是器械仍保留
`8602` 个高密度接触采样，并且当前接触支持表和局部防翻转检查仍覆盖较大的四面体
邻域；因此本次首先解决“组织物理网格过密”，不能把它写成接触求解已经按 4 倍加速。

## 器械 2D 距离场约束的局部视觉力（2026-08-07）

### 修改边界

本次仅修改 RGB 视觉力优化的逐像素 loss weight。按最终要求，明确没有加入：

```text
q7 / 器械状态视觉力门控       未加入
旧视觉力缓存失效策略           未修改
per-Gaussian 二次空间门控      未加入
PBD 四面体/碰撞面/刚度         未修改
Gaussian 外观和物理绑定        未修改
```

旧实现只对整块传播 tissue mask 做 `7 px` 二值腐蚀；只要像素仍在组织 mask 内，远离
器械的区域和部分不可靠边界仍可参与 RGB residual。新实现使用现有双目 SurgicalSAM2
mask：完整器械并集只负责遮挡清零，视觉力距离场只以 distal/夹爪 mask 为种子；同时
构造组织内部到 tissue-mask 边界的距离场和到图像裁切边框的距离场。

器械 mask 输入：

```text
data/super/psm_visual_calibration/raw_paper_lnd_stereo_dense_contact_v4/
surgicalsam2_multianchor_parts_dense_contact_v6/
stereo_multianchor_part_masks.npz
```

使用 `left_masks_packbits / right_masks_packbits` 作为 shaft+distal 遮挡并集，使用
`left_distal_masks_packbits / right_distal_masks_packbits` 作为视觉力作用距离场种子。
这样长杆附近和远离相机的组织不会仅因靠近 shaft 而获得视觉力。mask 原始分辨率为
`960x540`；tissue mask 和 RGB loss 分辨率为 `1920x1080`。严格使用
`stereo_left_index / stereo_right_index` 对齐各自视频帧。1441 帧 GUI 区间内，左右目
共同缺失的 strict-pair 帧只有 frame 498；该帧使用相邻 frame 497，允许的最大替代
间隔固定为一帧。

### 权重公式和正式参数

设：

```text
M_tissue(p)     当前相机传播得到的组织 mask
d_tool(p)       像素 p 到 distal/夹爪 mask 的外部距离
d_occ(p)        像素 p 到完整 shaft+distal 器械并集的外部距离
d_edge(p)       组织内部像素 p 到 tissue-mask 边界的距离
d_frame(p)      像素 p 到图像裁切边框的距离
f_tool          器械距离平滑衰减
f_edge          组织边界平滑置信度
f_frame         图像裁切边缘平滑置信度
```

正式逐像素权重为：

```text
W(p) = M_tissue(p) * f_tool(d_tool(p)) * f_edge(d_edge(p)) * f_frame(d_frame(p))
```

在 `1920x1080` 下：

```text
完整器械及其外侧 6 px        W = 0
d_tool <= 120 px              f_tool = 1
120 px < d_tool < 360 px      f_tool 用 cubic smoothstep 从 1 降到 0
d_tool >= 360 px              f_tool = 0

d_edge <= 24 px               f_edge = 0
24 px < d_edge < 64 px        f_edge 用 cubic smoothstep 从 0 升到 1
d_edge >= 64 px               f_edge = 1

d_frame <= 48 px              f_frame = 0
48 px < d_frame < 96 px       f_frame 用 cubic smoothstep 从 0 升到 1
d_frame >= 96 px              f_frame = 1
```

因此 near/far 覆盖范围较宽，可对器械周围较大组织区域做形状修正；但即使 360 px
作用半径覆盖到组织外轮廓，`f_edge` 仍优先把最外侧 24 px 清零，并在随后 40 px 内
渐入。若 tissue mask 一直贴到图像下沿，普通 tissue 边界距离看不到画面外的零值，
因此 `f_frame` 独立把所有图像边框 48 px 清零、48--96 px 渐入，明确去除靠近相机一端
的裁切边缘力。高光降权仍保留，但改为乘法，即 `W_highlight = 0.1 * W`，不会把本来
很低的距离场权重反向抬高到 `0.1`。

距离变换在四分之一分辨率 `480x270` 计算，再将连续权重双线性上采样；所有半径仍以
最终 1080p 图像像素定义。上采样后重新应用精确 tissue 二值支持和器械遮挡支持，防止
插值向背景或器械内部泄漏。实测单相机中位计算时间约 `10.50 ms`、p95 约
`10.99 ms`；两目每个新视频帧约 `21 ms`，低于最初半分辨率实现的约 `49 ms`。

### 真实 mask 抽样和门禁

在左右目 frame `0 / 420 / 498 / 1000 / 1440` 上验证：

```text
保留的非零权重像素 / 原 tissue mask     36.7% -- 64.3%
非零区域平均权重                         0.448 -- 0.555
器械及 6 px 遮挡带最大权重               0.0
最大器械帧替代间隔                       1 frame
```

合成几何门禁同时验证了：器械遮挡为零、near 区为满权重、near/far 之间连续衰减、far
以外为零、组织边界优先归零、tissue mask 接触图像边缘时仍归零、背景为零以及所有
权重位于 `[0,1]`。

### 低质量节点的视觉力加速度上限

v9 节点质量由 incident tetra rest volume 积分得到，动态节点质量跨度为：

```text
minimum / p01 / p50 / maximum mass
7.51e-9 / 3.05e-8 / 4.16e-6 / 6.40e-5 kg
```

旧的统一 per-particle force cap 为 `3e-5 N`；它对中位质量节点约等于 `7.21 m/s^2`，
但对最轻节点可达到约 `3993 m/s^2`，这是局部粒子乱飞的直接数值风险。第一版新增的
`5 m/s^2` 质量上限仍然过宽：视觉力在 6 个视频/物理更新之间持续施加，在当前
`10/s` 速度阻尼下仍允许约 `0.5 m/s` 的稳态速度。正式值因此降低 50 倍。现在保留原
`3e-5 N` 力上限和 `0.02 N` 总预算，同时新增质量相关限制：

```text
|f_i| <= mass_i * 0.1 m/s^2
```

因此实际粒子上限为 `min(3e-5 N, mass_i*0.1 m/s^2)`，固定节点仍为零。该限制只约束
视觉力产生的瞬时加速度，不更改 RGB loss、Gaussian 绑定、材料刚度或接触力。
`0.1 m/s^2` 与 `10/s` 阻尼对应约 `0.01 m/s` 稳态速度；连续 0.1 秒的无阻尼上界位移
约 `0.5 mm`，仍保留局部形状修正能力。v9 功能门禁另外用故意过大的目标和
正式 `0.1 m/s^2` 测试值验证 clamp 数学实现，所有力有限且固定节点为零。

另外在第二块 GPU 上使用正式 v9 网格、`E=50 Pa`、`nu=0.35`、零重力、`10/s` 阻尼，
对中央 Gaussian patch 连续重复施加 300 个 60 Hz 物理步（约 5 秒），得到：

```text
maximum particle acceleration       0.10000002 m/s^2
maximum observed particle speed     0.001151 m/s
maximum displacement after 300      0.300 mm
minimum tetrahedron volume ratio    0.799
maximum total-volume error          0.089%
inverted tetrahedra                 0
anchor drift                        0
```

该回归直接验证持续施力而非单次 clamp；按此结果，纯视觉力路径不会再产生高速飞散节点。
若后续仍在特定夹取帧观察到高速节点，应将来源归到接触/持续夹取位置修正并单独记录，
不能再次用增大或缩小 RGB loss mask 来替代诊断。

正式实现和验证入口：

```text
src/embodied_gaussians/embodied_simulator/visual_force_masks.py
scripts/test_super_tool_distance_visual_force_weights.py
scripts/test_super_stereo_visual_force_weights.py
scripts/test_super_soft_visual_force_wiring.py --device cpu
scripts/test_super_soft_force_scatter.py --asset <v9 asset> --physics-steps 0
```

ThinLinc GUI 已重启，启动日志明确打印正式 `120/360 px` distal/夹爪场、`24/64 px` 组织
边界场、`48/96 px` 图像边框场、`6 px` 完整器械遮挡带和 `0.1 m/s^2` 粒子加速度上限，
并继续加载 v9 降密 PBD 组织。

### 视觉力幅度回归修正（2026-08-07）

实际 GUI 检查发现，上述 `0.1 m/s^2` 紧急上限虽然抑制了低质量节点飞散，
但也把 RGB 视觉力的形状修正压到 GUI 中几乎不可见。这是过度限幅，不是
双目、mask、Gaussian 绑定或 soft-force scatter 链路被关闭。正式上限修正为：

```text
mass-aware particle acceleration cap    0.1 -> 1.0 m/s^2
old unstable candidate                  5.0 m/s^2 (not restored)
pixel SDF / tissue-edge / image-edge    unchanged
Gaussian binding / material / contact   unchanged
```

使用正式 v9 网格、`E=50 Pa`、`nu=0.35`、零重力、`10/s` 阻尼和中央
Gaussian patch 连续施力 300 个 60 Hz 物理步：

```text
maximum particle displacement           0.512 mm
maximum observed particle speed         0.001594 m/s
minimum tetrahedron volume ratio         0.81035
inverted tetrahedra / anchor drift       0 / 0
maximum total-volume error               0.134%
```

因此 `1.0 m/s^2` 比过弱值提高 10 倍，但仍只是旧 `5.0 m/s^2` 飞散候选的
1/5。GUI 同时显式打印 `Visual force solver: ACTIVE` 或
`PAUSED (press Play)`；视觉力仅在回放 Play 状态求解，Reset 后暂停时不再被
误判为链路消失。

## v10 中央物理粒子降密（2026-08-07）

用户要求消除抓取中心的小范围高密粒子团，中心密度与周边接近。本次只重建
四面体 PBD 和它的外边界碰撞面，不减少视觉三角面或 Gaussian，不改 RGB、
透明度、椭球尺寸和朝向。

正式新资产：

```text
physics:
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v10_uniform_coarse_physics/tissue_soft_adaptive.npz

runtime visual + physics:
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v10_uniform_coarse_centroid_ellipsoids/
tissue_paper_pbd_centroid_gaussians.npz
```

空间参数由中心 `1.5 mm` / 外围 `3.0 mm` 改为中心 `2.7 mm` / 外围
`3.0 mm`；中间过渡区中位目标间距为 `2.920 mm`。因此中心与外围只差约
11%，不再有中心独立加密档。

```text
                                      v9              v10
all physical particles                6959             4438   (-36.2%)
central-zone physical particles       2992             1281   (-57.2%)
tetrahedra                            31044            17016   (-45.2%)
collision boundary triangles          7420             6406   (-13.7%)
visual triangles / Gaussians          26754            26754   (unchanged)
zone target-spacing medians (mm)      1.57/2.22/3.00   2.70/2.92/3.01
```

完整质量门禁通过：网格为单一连通四面体，外皮为精确闭合二流形，所有静止
体积为正，最大 rest-matrix condition 从 `153.36` 改善到 `112.67`。原 v9 的
`26754` 个面中心椭球 Gaussian 外观逐数组复用，只重算了它们到新物理表面的绑定。

零位姿偏移的 state `1795--1830` GPU 真实轨迹短回归也通过：

```text
capture state                          1807
captured direct nodes                  exactly 4
3 mm press ROI nodes                   36
ROI depressed beyond 0.1 mm            32 / 36
final persistent grip                  active
inverted tetrahedra                    0
minimum tetrahedron volume ratio       2.04e-6
physics with contact median            35.11 ms / trajectory step
Gaussian skinning median               0.382 ms / trajectory step
```

运行入口和验证默认路径均已切换到 v10。

## v11 消除中央推断补全点的强制高密补丁（2026-08-07）

v10 降低了体网格的中心目标密度，但 GUI 仍显示中央有一小块极密物理点。独立统计
确认该现象不是 `2.7/3.0 mm` 的体采样差异，而是旧物理顶面构建器的特殊保留策略：

```text
inferred-completion source vertices     205
their nearest-neighbour spacing         0.410 mm
ordinary stride-4 surface spacing       1.640 mm
inferred patch XY extent                x=-4.92..4.10 mm,
                                        y=-8.20..19.68 mm
```

这 `205` 个点是深度缺口推断补全点。旧代码即使对所有区域设置 stride 4，仍将
source class 4 的点及一环邻域全部强制加入物理 PLC，所以产生了局部 4 倍线密度、
约 16 倍面密度的小补丁。

`build_super_adaptive_visual_surface.py` 新增显式
`--[no-]preserve-inferred-completion` 策略。正式物理顶面使用
`--no-preserve-inferred-completion`，使推断补全点与普通点一样遵循 stride 4。
这个参数只改物理顶面；独立视觉表面仍保留全部 `205` 个推断顶点和全部
`26754` 个 Gaussian，因此不丢失补洞视觉细节。

正式 v11 资产：

```text
rest_surface_adaptive_physics_v5_uniform/rest_surface.npz
paper_pbd_tissue_v11_uniform_surface_physics/tissue_soft_adaptive.npz
paper_pbd_tissue_v11_uniform_surface_centroid_ellipsoids/
tissue_paper_pbd_centroid_gaussians.npz
```

```text
physical top inferred vertices          205 -> 12
physical top vertices                   2141 -> 1835
particles                               4438 -> 4059
tetrahedra                              17016 -> 15830
collision triangles                     6406 -> 5754
visual triangles / Gaussians            26754 -> 26754
central 0--8 mm top NN p05/median        0.410/0.410 -> 1.544/1.640 mm
minimum rest tet volume                 0.000462 -> 0.019107 mm^3
```

独立网格与绑定门禁全部通过：单一连通体、闭合二流形外皮、正静止体积、
一面一 Gaussian、椭球外观复用、双目单侧及后续帧证据保留均通过。

零偏移 state `1795--1830` 真实夹爪回归：

```text
capture state / direct anchors           1807 / 4
press ROI depressed beyond 0.1 mm        22 / 26
grip direct safe scale at capture        1.0
proposed nonpositive tetrahedra           0
final persistent grip                    active
inverted tetrahedra                      0
physics with contact median              29.46 ms
Gaussian skinning median                 0.380 ms
```

运行入口和验证默认路径已切换到 v11。

## v12 统一粗网格与论文式固定 XPBD 基线（2026-08-08）

### 本阶段范围

本阶段按用户要求只完成两项：重新生成组织物理网格，以及把主材料求解器切换为
Liang et al.（arXiv:2309.11656）volumetric tissue 使用的 distance、volume、
shape-matching 三类约束。以下模块明确不在本阶段启用：

```text
online stiffness optimization   OFF
depth residual mapping          OFF
RGB visual-force redesign       OFF（保留现有链路，不在本阶段改算法）
contact/grip redesign           OFF（保留现有链路，待 v12 轨迹回归）
gravity                         0 m/s²（观测形状已是承重平衡形状）
```

论文在 volumetric tissue 上使用四面体网格和三类约束；距离与形状刚度由参与约束的
节点刚度取平均，体积刚度固定为 `1e10`。论文给出的第三组统一初始化为
`k_dist=0.2, k_shape=0.005`，本阶段选它作为固定、可复现的起点。论文的 residual
mapping 与逐节点在线刚度优化留到下一阶段，不应写进当前实验结果。

### 物理顶面与四面体重建

v11 虽已去除中央推断点补丁，但仍有 4,059 个节点和 15,830 个 tetra。v12 进一步把
物理 PLC 顶面统一降为 stride 9；中心、过渡、外围使用完全相同的横向采样规则，
不再为抓取中心创建局部加密岛。视觉表面不随之降采样。

正式资产：

```text
physical top:
data/super/grasp5_native/tissue_multiview_v1/
rest_surface_paper_pbd_v12_uniform_s9/rest_surface.npz

tetrahedral physics:
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v12_paper_constraints_physics/tissue_soft_adaptive.npz

runtime physics + visual:
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v12_paper_constraints_centroid_ellipsoids/
tissue_paper_pbd_centroid_gaussians.npz
```

网格参数与结果：

```text
physical top stride                       9 / 9 / 9
physical top maximum edge                 6.5 mm
physical top vertices / faces             407 / 702
dense visual surface coverage p95         0.4535 mm

surface / background / deep spacing       4 / 5 / 6 mm
lateral adaptive density                  OFF
TetGen min ratio / min dihedral            1.5 / 10 deg
particles / tetrahedra                    1,062 / 3,750
unique tetrahedral edges                  5,647
collision triangles                       1,672
  top / side / bottom                     772 / 437 / 463
surface / top / fixed-support nodes       838 / 444 / 113
minimum rest tetrahedron volume           0.126687 mm³
maximum rest matrix condition             119.1723
integrated volume / mass                  41.4576 cm³ / 41.4576 g
```

所有构建门禁通过：单一连通四面体体、所有静止体积为正、碰撞外皮为精确闭合二流形、
最小体积与最大 condition 均在阈值内、相邻节点目标间距突变门禁通过。物理节点真实
碰撞半径仍为零；`0.45 mm` 仅为 GUI 调试显示半径。

### 视觉层和 Gaussian 不降质

v12 只替换物理节点、tetra、碰撞外皮以及视觉层到物理表面的绑定。高分辨率视觉层和
椭球外观保持：

```text
visual vertices / triangles               13,583 / 26,754
ellipsoid Gaussians                       26,754
left-only / right-only visual vertices    1,444 / 544
dual-view / inferred visual vertices      11,390 / 205
later-frame-only Gaussians                 1,872
Gaussian scale                            0.175 .. 0.700 mm
anisotropy p50 / p95 / max                1.91 / 4.0 / 4.0
```

v11 的 RGB、opacity、四元数和三轴 scale 被逐数组、bit-identical 复用；仅重新计算
高分辨率视觉顶点到 v12 物理三角面的嵌入。Gaussian 中心仍严格位于高分辨率视觉
三角面中心，运行时随视觉面局部坐标架旋转，不是改成圆形物理粒子。视觉、物理、
碰撞三层数量依旧相互独立。

构建器原有的物理数组保持门禁曾因 `grasp_roi_center_xy_table=[NaN,NaN]` 使用普通
`np.array_equal` 而误报失败。已改为 `equal_nan=True`；这是验证器修复，不改变资产。

### 固定 distance / volume / shape-matching XPBD

新增运行模式 `paper_pbd`。旧 `paper_soft` 的 Neo-Hookean 路径保留用于历史回归，
但 GUI、ThinLinc 和 browser 启动脚本默认均切换到 `paper_pbd`。

距离约束：

\[
C_{ij}^{dist}=\|x_i-x_j\|-L_{ij}^{0}.
\]

四面体有符号体积约束：

\[
C_e^{vol}=\frac{((x_1-x_0)\times(x_2-x_0))\cdot(x_3-x_0)}{6}-V_e^0.
\]

形状匹配以每个 tetra 的四个节点作为一个 cluster。先计算
\(F=D_sD_m^{-1}\)，对 \(F\) 做 SVD/polar decomposition 得到无反射旋转 \(R\)，
再以四个节点到 `center + R q_i` 的联合残差范数作为标量约束。使用变形梯度而不是
未预条件的原始 `Apq`，可避免 condition 较大的 tetra 在静止态累积 SVD 浮点偏置；
另设相对 `1e-4` 位置死区，仅覆盖远低于图像/深度分辨率的 float32 噪声。

每类约束使用 XPBD 更新：

\[
\Delta\lambda=
\frac{-C-\widetilde\alpha\lambda}
{\sum_i w_i\|\nabla_iC\|^2+\widetilde\alpha},\qquad
\widetilde\alpha=\frac{1}{k_c\Delta t_{sub}^{2}}.
\]

固定运行参数：

```text
k_dist / k_vol / k_shape             0.2 / 1e10 / 0.005
physics frame / substeps             1/60 s / 12
substep dt                            1/720 s
generic XPBD iterations              3
paper constraint iterations          8
paper constraint relaxation          1.0
velocity damping                     10 s^-1
material minimum volume ratio        1e-4
```

距离和形状刚度在求解器中以 per-particle array 保存，约束刚度取参与节点平均；当前
每个节点都填固定相同值。这个数据布局只为以后在线优化留接口，不代表当前已启用优化。
体积刚度是全局固定标量。三类 Jacobi 修正按 distance -> volume -> shape matching
顺序执行，局部非翻转修复、ground 和器械接触仍使用共同的后处理路径。

主要代码：

```text
src/embodied_gaussians/physics_simulator/integrator.py
src/embodied_gaussians/physics_simulator/simulator.py
examples/embodied_environments/super_embodied/super_embodied.py
examples/example_embodied_super_offline.py
scripts/test_super_paper_constraint_xpbd.py
```

### 数值验证

CPU 已实际编译新 Warp kernels，并运行独立固定约束门禁。测试先在精确参考构型运行
20 个子步，再把中央动态顶面节点拉动 `0.75 mm`（另带 `0.3375 mm` 横向分量），
无外力运行 80 个恢复子步：

```text
rest maximum particle drift              2.7999e-10 m
rest minimum / maximum volume ratio      0.9999976 / 1.0000074
rest inverted tetrahedra                 0

initial perturbation inverted tets       0
recovery inverted tetrahedra             0
recovery minimum / maximum volume ratio  0.896964 / 1.117050
fixed support drift                       0
edge-constraint RMS                      1.7602e-5 -> 4.6088e-6 m
volume-constraint RMS                    7.5571e-11 -> 1.8372e-11 m³
```

验证还逐项确认求解器派生的 5,647 条边及静止长度与资产中的
`pbd_edge_indices/pbd_rest_edge_length` 相同，并确认视觉数量与椭球层未变化。报告：

```text
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v12_paper_constraints_centroid_ellipsoids/
paper_constraint_solver_report.json
```

完整环境 CPU 构建也通过：`paper_pbd` 加载 1,062 个节点、3,750 个 tetra、26,754 个
soft Gaussians，打印 `depth_residual=OFF, stiffness_optimization=OFF`，并成功完成一次
关闭器械接触的完整环境物理步。本机当前无 CUDA 设备，因此本阶段没有冒充 GPU 性能
或完整器械轨迹结果。下一阶段在调整接触参数前，必须先用 v12 跑 state 1795--1830
以及完整序列回归；v11 的接触数据不能直接当作 v12 结论。

### 可复现命令

```bash
python scripts/build_super_adaptive_visual_surface.py \
  --output-dir data/super/grasp5_native/tissue_multiview_v1/rest_surface_paper_pbd_v12_uniform_s9 \
  --fine-stride 9 --transition-stride 9 --outer-stride 9 \
  --maximum-edge-mm 6.5 --minimum-faces 350 \
  --no-preserve-view-exclusive --no-preserve-every-boundary \
  --no-preserve-inferred-completion --overwrite

python scripts/build_super_adaptive_tissue.py \
  --rest-surface data/super/grasp5_native/tissue_multiview_v1/rest_surface_paper_pbd_v12_uniform_s9/rest_surface.npz \
  --rest-surface-report data/super/grasp5_native/tissue_multiview_v1/rest_surface_paper_pbd_v12_uniform_s9/report.json \
  --output-dir data/super/grasp5_native/tissue_multiview_v1/paper_pbd_tissue_v12_paper_constraints_physics \
  --surface-spacing-mm 4 --dense-depth-mm 2 --transition-end-mm 14.5 \
  --deep-spacing-mm 6 --background-spacing-mm 5 --visual-radius-mm 0.45 \
  --tetgen-min-ratio 1.5 --tetgen-min-dihedral-deg 10 --overwrite

python scripts/build_super_visual_face_centroid_gaussians.py \
  --physics-asset data/super/grasp5_native/tissue_multiview_v1/paper_pbd_tissue_v12_paper_constraints_physics/tissue_soft_adaptive.npz \
  --reuse-appearance-asset data/super/grasp5_native/tissue_multiview_v1/paper_pbd_tissue_v11_uniform_surface_centroid_ellipsoids/tissue_paper_pbd_centroid_gaussians.npz \
  --output-dir data/super/grasp5_native/tissue_multiview_v1/paper_pbd_tissue_v12_paper_constraints_centroid_ellipsoids \
  --asset-version 12 --overwrite

PYTHONPATH=src:examples python scripts/test_super_paper_constraint_xpbd.py --device cpu
```

## v13 稍密统一网格与小幅固定硬化（2026-08-08）

用户在 GUI 中检查 v12 后认为物理网格略少、组织略软。本次采用温和调整，不恢复任何
中央局部加密，也不启用在线优化：

```text
physical top stride             9 -> 8（所有横向区域一致）
surface/background/deep spacing 4/5/6 mm（保持）
k_dist                          0.2 -> 0.3
k_vol                           1e10（保持论文固定值）
k_shape                         0.005 -> 0.0065
online stiffness optimization   OFF
depth residual mapping          OFF
gravity                         0 m/s²
```

`k_dist=0.3, k_shape=0.0065` 是从论文第三组 `0.2/0.005` 出发的小幅固定硬化，
不是论文原表中的另一组参数。论文写作中应表述为“沿用 distance/volume/shape-matching
约束结构和固定体积参数，针对当前数据做固定参数微调”，不能声称三项数值全部来自原文。

### v13 正式资产与规模

```text
physical top:
data/super/grasp5_native/tissue_multiview_v1/
rest_surface_paper_pbd_v13_uniform_s8/rest_surface.npz

physics:
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v13_uniform_s8_paper_constraints_physics/
tissue_soft_adaptive.npz

runtime physics + visual:
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v13_uniform_s8_paper_constraints_centroid_ellipsoids/
tissue_paper_pbd_centroid_gaussians.npz
```

与 v12 对比：

```text
                                      v12        v13       change
physical top vertices / faces         407/702     501/874   +23.1%/+24.5%
particles                             1,062       1,201     +13.1%
tetrahedra                            3,750       4,199     +12.0%
unique tet edges                      5,647       6,366     +12.7%
collision triangles                  1,672       1,934     +15.7%
surface/top/support nodes             838/444/113 969/532/128
visual triangles / Gaussians          26,754      26,754    unchanged
mass                                  41.4576 g   41.5746 g +0.28%
minimum rest tet volume               0.1267      0.1730 mm³
maximum rest condition                119.17      111.43
dense-surface coverage p95            0.4535      0.4106 mm
```

增加量主要来自全局统一 stride 变化，中心没有单独密集档。视觉层继续逐数组复用 v12
的 26,754 个椭球 Gaussian 外观，只重新计算到 v13 物理外皮的嵌入；左右目、后续帧、
RGB、opacity、旋转和三轴 scale 均保持。

### 构建时发现并修复的底面耳切问题

stride-8 顶面本身通过所有表面门禁，但第一次封闭 PLC 四面体化被 TetGen 正确拒绝，
报告 4 个自交三角面。诊断确认不是顶面交叉，而是底面 ear clipping 将一组源自
float32 的“几乎共线”边界点误判成微小凸角，生成了一条与原边界重叠的长对角线。

`triangulate_bottom_polygon()` 现在除了绝对面积容差，还要求候选耳的转角满足相对条件：

```text
cross(ab,bc) > max(absolute_tolerance, 1e-6 * |ab| * |bc|)
```

修正后同一 stride-8 顶面可直接生成闭合体，TetGen 不再报告相交。最终网格为单一连通
体、所有静止 tetra 正体积、闭合外皮边 incidence 全为 2；所有既有质量门禁通过。
该修复保留了几何检查，未使用跳过自交或强制接受坏 PLC 的选项。

### 固定硬化对比

使用同一个 v13 网格和同一个中央动态顶面节点，施加 `0.75 mm` 法向加
`0.3375 mm` 横向初始扰动，随后无外力恢复 80 个 `1/720 s` 子步：

```text
                                             old 0.2/0.005   v13 0.3/0.0065
recovery maximum displacement                0.17390 mm      0.10218 mm
recovery edge-constraint RMS                 5.1336e-6 m     3.7110e-6 m
recovery volume-constraint RMS               2.4817e-11 m³   1.4854e-11 m³
recovery minimum volume ratio                0.87497         0.85841
recovery inverted tetrahedra                 0               0
fixed support drift                          0               0
```

即在相同恢复时间内，最大残余位移降低约 41%，边约束 RMS 降低约 28%，体积约束 RMS
降低约 40%。最小体积比仍远高于非翻转底线，静止 20 子步最大漂移为
`3.59e-10 m`。因此该参数比 v12 明显稍硬，但没有改成接近刚体的数量级。

正式报告：

```text
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v13_uniform_s8_paper_constraints_centroid_ellipsoids/
paper_constraint_solver_report.json
```

GUI、ThinLinc 和 browser 继续使用 `paper_pbd` 模式，但其资产路径已切到 v13，启动
日志应打印：

```text
particles=1201, tetrahedra=4199, skin_triangles=1934
k_dist=0.3, k_vol=1e10, k_shape=0.0065
stiffness_optimization=OFF, depth_residual=OFF
```

完整环境已在 CPU 成功加载 26,754 个 soft Gaussian 并完成一个关闭器械接触的物理步。
当前机器无 CUDA，v13 的真实器械轨迹接触与性能仍需在 GUI/GPU 上观察后再冻结；本次
没有借“稍硬”之名修改 contact、grip、视觉力或在线优化逻辑。

## v14 宽中心温和加密与第二档固定硬化（2026-08-08）

用户要求中间部分四面体再稍密、刚度再高一点。本次没有改变 stride-8 物理顶面，
也没有恢复旧版 0.41 mm 的中央补全点密集岛。改动只作用于 TetGen 连续尺寸场：

```text
grasp center                         (-2.3684, -3.4818) mm in table x-y
center radius                        20 mm
center surface target spacing        3.5 mm
transition radius                    20--45 mm
transition target spacing            3.5 -> 4.0 mm（几何连续插值）
outer surface target spacing         4.0 mm
background/deep target spacing       5.0 / 6.0 mm
k_dist                               0.3 -> 0.4
k_vol                                1e10（保持）
k_shape                              0.0065 -> 0.008
online stiffness optimization        OFF
depth residual mapping               OFF
gravity                              0 m/s²
```

第一次使用 20--35 mm 过渡时，相邻节点目标间距最大比为 `1.3708`，超过既有 `1.35`
门禁；20--40 mm 时仍为 `1.3542`。没有放宽质量门禁，最终把过渡终点扩大到 45 mm，
最大比降为 `1.3471` 并通过。这里扩大的是平滑过渡带，不是扩大最密的 20 mm 中心区。

### v14 正式资产与规模

```text
physics:
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v14_center_mild_paper_constraints_physics/
tissue_soft_adaptive.npz

runtime physics + visual:
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v14_center_mild_paper_constraints_centroid_ellipsoids/
tissue_paper_pbd_centroid_gaussians.npz
```

与 v13 对比：

```text
                                      v13        v14       change
particles                             1,201      1,309     +9.0%
tetrahedra                            4,199      4,785     +14.0%
unique tet edges                      6,366      7,083     +11.3%
collision triangles                  1,934      1,980     +2.4%
surface/top/support nodes             969/532/128 992/535/129
center r<=20 mm nodes                 279        327       +17.2%
center r<=20 mm tetra centroids       1,187      1,506     +26.9%
transition 20<r<=45 mm nodes          813        869       +6.9%
transition tetra centroids            2,741      2,992     +9.2%
visual triangles / Gaussians          26,754     26,754    unchanged
mass                                  41.5746 g  41.5746 g unchanged
minimum rest tet volume               0.1730     0.1168 mm³
maximum rest condition                111.43     103.88
```

中心增量显著高于过渡区和全局增量，但细区覆盖半径 20 mm，不是把粒子塞进一个很小
区域。单连通、正静止体积、闭合外皮、预算、最小体积、条件数和连续尺寸场门禁全部通过。

视觉层仍为 13,583 顶点、26,754 三角面和 26,754 个面中心椭球 Gaussian。
RGB、opacity、静止四元数和三轴尺度从 v13 逐数组原样复用；只重新计算到 v14 物理表面
的视觉顶点嵌入和 Gaussian 绑定。左右目独有点和后续帧补点均未删除。

### 固定刚度对照与验证

在同一 v14 网格、同一中央节点、同一 `0.75 mm z + 0.3375 mm x` 扰动、80 个
`1/720 s` 恢复子步下：

```text
                                      old 0.3/1e10/0.0065  v14 0.4/1e10/0.008
recovery maximum displacement         0.11304 mm            0.04881 mm
recovery edge-constraint RMS          4.5849e-6 m           3.4524e-6 m
recovery volume-constraint RMS        1.7820e-11 m³         1.4343e-11 m³
recovery minimum volume ratio         0.72532               0.72328
recovery inverted tetrahedra          0                     0
fixed support drift                   0                     0
```

新档残余位移降低 56.8%、边约束 RMS 降低 24.7%、体积约束 RMS 降低 19.5%。
静止 20 子步最大漂移为 `2.80e-10 m`。`0.4/0.008` 是在论文第三组初始化基础上的
第二档固定数据集内硬化，不是论文表中原值，论文不能写成三项数值完全复现原文。

正式 solver 报告：

```text
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v14_center_mild_paper_constraints_centroid_ellipsoids/
paper_constraint_solver_report.json
```

### 可复现命令

```bash
python scripts/build_super_adaptive_tissue.py \
  --rest-surface data/super/grasp5_native/tissue_multiview_v1/rest_surface_paper_pbd_v13_uniform_s8/rest_surface.npz \
  --rest-surface-report data/super/grasp5_native/tissue_multiview_v1/rest_surface_paper_pbd_v13_uniform_s8/report.json \
  --output-dir data/super/grasp5_native/tissue_multiview_v1/paper_pbd_tissue_v14_center_mild_paper_constraints_physics \
  --surface-spacing-mm 3.5 --dense-depth-mm 2 --transition-end-mm 14.5 \
  --deep-spacing-mm 6 --lateral-adaptive --lateral-fine-radius-mm 20 \
  --lateral-transition-radius-mm 45 --lateral-outer-spacing-mm 4 \
  --background-spacing-mm 5 --visual-radius-mm 0.45 \
  --tetgen-min-ratio 1.5 --tetgen-min-dihedral-deg 10 --overwrite

python scripts/build_super_visual_face_centroid_gaussians.py --overwrite

PYTHONPATH=src:examples python scripts/test_super_paper_constraint_xpbd.py --device cpu
```

完整 CPU 环境已成功加载 `1,309` 个节点、`4,785` 个 tetra、`1,980` 个碰撞面和
`26,754` 个 soft Gaussian，并完成一次关闭视觉力和器械接触的物理步。GUI 默认资产
和固定参数已切到 v14；contact、grip、视觉力、器械位姿、重力和在线优化逻辑均未改动。

## v15 整体温和加密（2026-08-08）

用户检查 v14 后要求网格继续变密。本次不是继续向一个很小中央区域堆点，而是将中心、
过渡、外围和深层的连续尺寸场整体收紧一档；刚度保持 v14 数值不变，以隔离网格分辨率
对观感的影响：

```text
center r<=20 mm               3.50 -> 3.25 mm
transition 20<r<=45 mm        3.25 -> 3.75 mm
outer surface                 4.00 -> 3.75 mm
background                    5.00 -> 4.75 mm
deep                          6.00 -> 5.50 mm
k_dist/k_vol/k_shape          0.4 / 1e10 / 0.008（保持）
```

最初尝试深层 `5.75 mm` 时，相邻目标间距最大比为 `1.35036`，略超 `1.35` 门禁。
没有放宽门禁，最终将深层也加密到 `5.50 mm`，最大比降至 `1.31925`。

正式资产：

```text
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v15_denser_mild_paper_constraints_physics/
tissue_soft_adaptive.npz

data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v15_denser_mild_paper_constraints_centroid_ellipsoids/
tissue_paper_pbd_centroid_gaussians.npz
```

与 v14 对比：

```text
                                      v14        v15       change
particles                             1,309      1,465     +11.9%
tetrahedra                            4,785      5,502     +15.0%
unique tet edges                      7,083      8,024     +13.3%
collision triangles                  1,980      2,116     +6.9%
surface/top/support nodes             992/535/129 1,060/549/131
center r<=20 mm nodes                 327        392       +19.9%
center r<=20 mm tetra centroids       1,506      1,782     +18.3%
transition 20<r<=45 mm nodes          869        953       +9.7%
transition tetra centroids            2,992      3,399     +13.6%
visual triangles / Gaussians          26,754     26,754    unchanged
```

网格门禁全部通过：单连通、无退化/负体积、闭合碰撞外皮、最小体积 `0.11677 mm³`、
最大静止矩阵条件数 `117.29`。视觉层 RGB、opacity、四元数和三轴椭球尺度逐数组复用
v14，只更新到 v15 物理外皮的嵌入和绑定。

固定刚度独立验证通过：静止 20 子步最大漂移 `2.88e-10 m`；0.75 mm 扰动恢复 80
子步后最大位移 `0.05166 mm`、最小体积比 `0.80465`、无翻转、固定支撑零漂移。
v14 相同刚度的最大恢复位移为 `0.04881 mm`，说明这次没有通过材料参数暗中改变软硬。

可复现四面体构建命令：

```bash
python scripts/build_super_adaptive_tissue.py \
  --rest-surface data/super/grasp5_native/tissue_multiview_v1/rest_surface_paper_pbd_v13_uniform_s8/rest_surface.npz \
  --rest-surface-report data/super/grasp5_native/tissue_multiview_v1/rest_surface_paper_pbd_v13_uniform_s8/report.json \
  --output-dir data/super/grasp5_native/tissue_multiview_v1/paper_pbd_tissue_v15_denser_mild_paper_constraints_physics \
  --surface-spacing-mm 3.25 --dense-depth-mm 2 --transition-end-mm 14.5 \
  --deep-spacing-mm 5.5 --lateral-adaptive --lateral-fine-radius-mm 20 \
  --lateral-transition-radius-mm 45 --lateral-outer-spacing-mm 3.75 \
  --background-spacing-mm 4.75 --visual-radius-mm 0.45 \
  --tetgen-min-ratio 1.5 --tetgen-min-dihedral-deg 10 --overwrite
```

`scripts/test_super_paper_constraint_xpbd.py --device cpu` 与完整 `paper_pbd` 环境一步均
通过；完整环境加载 1,465 个节点、5,502 个 tetra 和 26,754 个 soft Gaussian。
contact、grip、视觉力、器械位姿、重力与在线优化逻辑均未修改。

## 2D 距离场后方单侧抑制（2026-08-08）

用户指出远离相机的后方组织仍出现不应有的视觉力。根因是既有 distal/jaw SDF 使用
各向同性 `120 px full / 360 px zero` 圆形扩张，夹爪上方也得到与侧向、靠近相机方向
相同的覆盖。针对当前固定 SUPER 内窥镜构图，图像负 y 对应远端/后方，因此新增一个
只作用于该方向的平滑因子：

```text
reference row               distal/jaw interaction mask 的中位图像行
posterior full reach        reference 上方 140 px
posterior transition        上方 140--280 px，smoothstep 1 -> 0
posterior exact zero        reference 上方 >=280 px
lateral/camera-near reach   保持原 120--360 px SDF，不缩小
tissue/image edge fields    保持 24/64 px 与 48/96 px
instrument occlusion        保持 6 px
```

该门控明确是固定相机下的 2D 方向先验，不是三维深度，也没有加入用户已否定的 q7、旧力
缓存或 Gaussian 二次门控。核心实现在
`src/embodied_gaussians/embodied_simulator/visual_force_masks.py`，GUI 显式传入
`posterior_full_reach_px=140`、`posterior_zero_reach_px=280`。

实际第 420 帧统计：

```text
                         left       right
old nonzero pixels       571,968    552,704
new nonzero pixels       543,136    523,728
old pixels beyond zero    28,308     28,484
new pixels beyond zero         0          0
```

靠近相机半侧的新旧权重逐像素一致；新支持集严格是旧支持集的子集；左右目、器械遮挡、
组织边缘、图像边缘、平滑过渡和 `[0,1]` 有界门禁全部通过：

```bash
PYTHONPATH=src python scripts/test_super_tool_distance_visual_force_weights.py
PYTHONPATH=src python scripts/preview_super_visual_force_posterior_gate.py
```

正式左右目对比图：

```text
outputs/super_visual_force_posterior_gate/frame_0420_stereo_before_after.png
```

四列依次为原图与 mask、旧各向同性场、新后方衰减场、被移除的权重；红线是 280 px
精确清零边界，黄线是 140 px 全权重边界。通用双目视频 smoke 脚本在当前非 CUDA shell
因 TorchCodec 强制 CUDA 解码无法执行，但它不测试器械条件场；实际左右目 mask 门禁和
直接 MP4 第 420 帧预览均已通过。

## 视觉力箭头细长显示（2026-08-08）

用户要求只把现有箭头变细、变长，不改 Adam。软组织箭头的数据源保持为完成
Gaussian-to-particle 散射、节点/质量相关限幅和全局预算后的实际粒子力：

```text
soft_force_display_gain     40 -> 60
soft_force_line_width        4 -> 2 px
visual_forces_scale          8（保持）
Adam/loss/force scatter/clamp  全部不变
```

`soft_force_line_width` 从硬编码改为 GUI 设置，调节范围 `1--6 px`；显示增益 GUI 范围
由 `1--50` 扩到 `1--100`。当前箭头显示长度满足

\[
L_i^{display}=8\times60\times\|f_i^{applied}\|.
\]

无显示环境下的 headless 几何门禁验证：`10 µN -> 4.8 mm`、`20 µN -> 9.6 mm`，
长度比严格为 `2.0`。因此变细变长只采用统一显示比例，仍正确表达 post-clamp 粒子力
的方向和相对大小，不会改变组织运动；它也不会让箭头直接线性等于 2D 像素权重。

## 视觉力更新加密与远区平方衰减（2026-08-08）

为减轻“每 6 个物理帧才出现一组新视觉力”可能产生的阶梯跳变，同时不把昂贵的双目
gsplat 反向传播直接提高到每帧一次，正式 GUI 取中间档：

```text
physics nominal rate                    60 Hz
visual solve interval                   6 -> 4 physics frames
configured visual solve rate            10 -> 15 Hz
maximum cached-force hold               100 -> 66.7 ms
cached force reapplication              60 Hz（保持）
expected visual-solve count/cost         +50%
```

这只改变何时重新求解视觉力；四个物理帧之间仍逐帧重放最近一次完成限幅的粒子力，
没有修改 Adam、学习率、Gaussian-to-particle 散射、节点限幅或全局力预算。15 Hz 是配置
频率，不是墙钟性能承诺，完整 CUDA GUI 的实际帧率仍需交互观察。

为了继续保留较宽的组织形状修正范围，near/far 半径保持 `120/360 px`，不把 far 半径
向内截断。只把原来的过渡函数从一次 smoothstep 改为平方：

\[
s(d)=\operatorname{smoothstep}\left(
\operatorname{clamp}\frac{360-d}{240},0,1\right),\qquad
f_{tool}(d)=s(d)^2.
\]

因此 `d<=120 px` 仍为 `1`，`d>=360 px` 仍为 `0`；`d=240 px` 从 `0.5` 降到
`0.25`，`d=300 px` 从约 `0.156` 降到约 `0.024`。后方单侧 `140/280 px` 门控、
组织边缘 `24/64 px`、图像边缘 `48/96 px` 和器械遮挡 `6 px` 均保持不变。

第 420 帧正式双目统计（旧值为 power-1 且未施加后方单侧门控，新值为 power-2 加
当前后方门控）：

```text
                         left          right
old weight sum           315,749.88    306,665.94
new weight sum           233,223.69    224,245.69
weight-sum reduction          26.1%         26.9%
old mean nonzero weight        0.5520        0.5548
new mean nonzero weight        0.4294        0.4282
new pixels beyond posterior zero  0             0
```

`scripts/test_super_tool_distance_visual_force_weights.py` 的全部合成和真实左右目门禁通过。
近区理论权重严格为 `1`；实际四分之一分辨率距离场经双线性上采样后的近区最大差异仅
`0.0009982`。对比图和数值报告已重生成：

```text
outputs/super_visual_force_posterior_gate/frame_0420_stereo_before_after.png
outputs/super_visual_force_posterior_gate/frame_0420_stereo_before_after.json
```

需要注意：逐像素权重进入归一化 RGB loss，而当前每次视觉更新会重置 Adam 并只迭代
一次，所以权重下降可以可靠减少远区残差的相对贡献，但不保证某个远区物理箭头按同一
比例线性缩短。GUI 箭头仍忠实显示最终 post-clamp 粒子视觉力，而不是直接显示像素权重。

## 论文式 Gaussian 视觉残差映射首版与强度 A/B（2026-08-08）

已核对 arXiv:2309.11656v2。论文每步先做 PBD 预测，再联合可见表面点云 Chamfer
与 PBD 约束能量优化粒子残差 \(\Delta_t\)，以 \(x_t+\Delta_t\) 修正状态；online
stiffness 再用当前残差范数、最近 20 帧中 4 个历史修正状态的零控制静止误差和空间
刚度平滑项逐步更新。论文体积版本报告约 `2.6 s/step`，感知另约 `0.4 s/image`，其
online 表示逐步自适应，并非 30--60 FPS。

本项目新增
`physics_simulator/visual_tissue_residual_mapping.py`，将点云项替换为当前 masked 双目
Gaussian RGB Smooth-L1，优化变量改为物理节点残差。目标包含归一化 distance、volume、
shape、空间、时间和幅值项；固定节点清零、每节点 `0.25 mm` 限幅，并在输出前做最小
体积比回溯。Simulator 的 `solve_visual_tissue_residual()` 只返回独立修正状态，不自动
写回 Warp；online stiffness 仍为 OFF。

真实 frame 420 强度 A/B：

```text
iterations / lr       max residual   loss reduction   min volume   warm solve
1 / 2.0e-5 m           0.0346 mm        1.96%           0.746       26.7 ms
3 / 2.0e-5 m           0.1032 mm        4.51%           0.403       40.8 ms
4 / 2.0e-5 m           0.1368 mm        5.50%           0.583       52.6 ms
5 / 2.0e-5 m           0.1697 mm        6.03%           0.556       73.5 ms
4 / 2.5e-5 m (selected)0.1701 mm        6.27%           0.458       68.7 ms
```

全部档位固定边界为零、翻转四面体为零、回溯为零。选中 `4 / 2.5e-5 m`，因为它用
4 次而不是 5 次迭代达到约 `0.17 mm`；但组织-only warm 时间仍略高于四物理帧的
`66.7 ms`，完整器械/接触/GUI 尚未计时，因此后续按异步残差更新设计，不能直接阻塞
当前物理循环。合成门禁和真实报告分别由
`scripts/test_super_visual_tissue_residual_mapping.py` 与
`outputs/visual_tissue_residual_mapping/real_frame_0420_iter4_lr25um.json` 保存。

## 论文式 Gaussian 视觉残差正式接入实时 GUI（2026-08-08）

旧版只输出独立 `corrected state`。现已在
`examples/example_embodied_super_offline.py` 中增加 `residual/force/off` 三种视觉反馈模式，
正式 ThinLinc 启动脚本默认使用 `residual`：

```text
visual feedback mode                  residual
image scale                           0.25
Adam iterations / learning rate       4 / 2.5e-5 m
maximum residual per accepted update  0.25 mm / particle
new solve interval                    4 physics frames
online stiffness update               OFF
legacy Gaussian-to-force path         available, but not active
```

每次循环仍先完成 XPBD 预测和器械位姿更新，再对 `1,465 x 3` 个物理节点求增量残差。
接受后只回写 `state_0.particle_q` 并立即重算 Gaussian 绑定；`particle_qd` 逐元素保持不变，
因此视觉状态估计不会被转换成额外速度或力。上一帧残差只作为 temporal reference，优化
变量每次从零增量开始，避免把已应用残差再次叠加。

首次全轨迹 GUI 发现：前两次分别接受约 `0.077/0.164 mm`，随后接触/物理过程使极少数
tet 的体积比接近零。首版绝对 `0.02` 体积门禁因此把所有后续候选都回溯成零，造成用户
看到“几乎没有改善”。门禁现改为：

```text
initial volume ratio >= 0.30     corrected ratio must remain >= 0.30
initial volume ratio < 0.30      one residual update may lose at most 2% relative volume
initial positive tet             forbidden to become non-positive
pre-existing compressed/bad tet  cannot globally disable all other visual correction
```

这不是解除体积约束，而是区分“视觉残差新造成的损坏”和“进入求解前已存在的物理压缩”。
合成预压缩门禁从初始体积比 `0.0100` 修正到 `0.1654`，无新增翻转且视觉残差继续工作。
新版 GUI 实际连续日志已跨过原停机点，周期样本仍全部接受：warm `69.2--80.3 ms`，
单次最大残差约 `0.078--0.081 mm`，初始/最终最小体积比分别为
`0.846->0.358`、`0.875->0.473`。第一次 CUDA/gsplat 冷启动约 `1.15 s`，不代表稳态。

为降低实时开销，残差反向渲染只包含 `26,754` 个 tissue Gaussian；完整 GUI 正向显示
仍有 `108,552` 个 Gaussian（含 `76,798` 个器械 Gaussian）。器械区域本来已由 loss
mask 清零，因此不参与反向渲染不会改变有效组织像素目标。完整场景两次 solve+apply 的
cold/warm 实测为 `932.0/76.0 ms`，报告在：

```text
outputs/visual_tissue_residual_mapping/full_scene_frame_0420.json
outputs/visual_tissue_residual_mapping/realtime_feedback_frame_0420_5steps.json
```

### Gaussian 尺度是否随视觉三角面变化

当前 mode-2 绑定会随 `26,754` 个高分辨率视觉三角面更新中心和朝向，但保留首帧双目
优化得到的三轴椭球尺度。它不是按 `2,116` 个粗碰撞三角面定尺寸。静止资产统计：视觉
面等效半径中位数 `0.174 mm`；Gaussian 三轴尺度中位数约
`0.401/0.371/0.330 mm`；`pi*s0*s1/face_area` 中位数 `3.61`，尺度与 `sqrt(area)` 的
相关系数只有 `0.15--0.26`。因此直接用面面积覆盖原尺度会缩小大量 Gaussian、重新产生
空洞，并不能增强物理残差。

后续若处理大拉伸，只允许采用有界的局部协方差 transport：从当前/静止视觉面的二维
形变梯度更新原椭球协方差并重新分解 quaternion/scale，而不是把三个 scale 简单设为面
边长或统一乘面积比。该项当前没有启用；先单独评估空洞、模糊和时序闪烁后再决定。

### 局部体积安全投影与更强凸起修正档（2026-08-08）

连续 GUI 日志进一步发现：进入接触段后，单个接近零体积 tet 会触发候选残差的全局
减半回溯，健康区域的最大修正也被一起压到 `0.001--0.01 mm`。现改为最多 4 轮局部
体积安全投影：只把违反体积条件的 tet 所涉及节点的本次残差置零，再复检相邻 tet；
局部投影仍无法满足条件时才进行全局 backtrack。GUI 新增显示
`locally frozen nodes / local passes / global backtracks`，用于区分局部接触坏单元和全局
视觉修正强度。

针对“夹爪左侧被压塌、观察图应当凸起但修正不明显”，只增加残差求解步长预算，不改
RGB loss、PBD stiffness、mask 或 Adam 类型：

```text
                                  old realtime     current strong realtime
iterations                        4                6
learning rate                     2.5e-5 m         3.0e-5 m
per-update particle cap           0.25 mm          0.40 mm
frame 420 max residual            0.170 mm         0.304 mm
frame 420 RMS residual            0.032 mm         0.048 mm
frame 420 loss reduction          6.27%            9.18%
warm solve                        68.7 ms           92.4 ms
```

强档真实 frame 420 连续 3 次全部接受，累计最大位移 `0.889 mm`，各次最大残差
`0.304/0.303/0.285 mm`，最小体积比均为正，固定节点为零且 `particle_qd` 完全不变。
正式报告：

```text
outputs/visual_tissue_residual_mapping/realtime_strong_frame_0420.json
```

该增强只在观察残差确实要求左侧向外移动时产生凸起；如果强档仍向错误方向更新，下一步
必须检查左侧有效 mask、左右目梯度一致性和接触形变方向，不能继续无上限放大步长。

正式强档 GUI 连续播放实测：初段 warm 更新最大残差 `0.286--0.294 mm`；进入压缩段后
一次更新在初始最小体积比 `0.005` 时仍保留 `0.307 mm` 最大修正，只局部冻结 10 个
节点、全局 backtrack 为 0，并将最终最小体积比提高到 `0.391`。更深接触段即使已有
零体积局部单元，后续接受更新仍保持 `0.304--0.307 mm`，局部冻结约 `45--60` 个节点、
全局 backtrack 为 0；观察到一次局部投影无法通过而拒绝，下一次即恢复接受。说明局部
门禁已解决“单个坏 tet 把整块视觉修正缩没”的主故障，但既有零体积物理单元本身仍需
后续从接触/XPBD 侧单独修复，不能把视觉残差当作材料求解器替代品。

## 夹爪压入后的局部凸起：接触传播与求解顺序修正（2026-08-08）

### 已确认根因

旧顶部 barrier 不只移动直接命中的三角面顶点，还按 `3 mm` 横向半径、`8 mm` 深度
搜索一个柱状邻域，并把其中表面及内部体节点统一沿世界 `-z` 方向移动。与此同时，材料
距离/体积/形状约束在接触之前执行，接触后的显示状态没有再经过体积恢复。因此夹爪左侧
只要位于柱状邻域内就被主动压成碗状凹陷；体积守恒还来不及把材料排向两侧。深压时个别
四面体随后接近零体积，又会触发局部回退或视觉残差节点冻结。

### 当前实现

纸张 PBD 模式采用以下固定顺序：

```text
8 x (distance -> volume -> shape)
2 x [tool contact -> 2 x (distance -> volume -> shape -> no-flip line search)]
local material-quality repair -> persistent grip -> velocity update
```

顶部 barrier 的空间柱状传播已经删除，改为：

```text
direct contacted face vertices       100% bounded downward correction
third vertex of edge-adjacent faces  <=15% correction, one surface ring only
interior tetrahedral support nodes   0%
generic tet contact spreading        0 layers in paper mode
```

一环是按顶部碰撞三角面的共享边拓扑建立，不按欧氏半径捕获任意体节点；原 `3 mm` 半径
仅作为一环候选的额外距离上限，`8 mm` 不再表示向体内传播深度。每个顶部面最多新增
3 个弱支撑节点。直接接触仍受每轮 `0.015 mm` 位移上限约束。

接触局部体积下限设为静止体积的 `0.03`。接触 proposal 先做严格正体积线搜索，再只
回退造成低体积的局部接触节点；每次接触后的材料恢复另存接触后位置，对完整材料 pass
做不低于同一体积下限的线搜索。因此禁止新翻转，但不再依靠把整片组织或整个物理步冻结
来维持拓扑。该机制不修改四面体连接关系、碰撞拓扑、Gaussian 绑定或在线刚度参数。

### 验证结果

当前 v15 资产包含 `1,465` 个物理节点、`5,502` 个四面体和 `2,116` 个碰撞三角面。
完整真实场景 CPU 深压回归得到：

```text
top collision faces                  969
weak one-ring support entries        305
mean / max added nodes per face      0.315 / 3
contact tet spreading layers         0
inverted tetrahedra                  0
minimum tetrahedron volume ratio     0.674
maximum dynamic speed                0.059 m/s (trajectory) / 0.066 m/s (deep)
anchor drift                         0
```

小规模接触回归全部通过：接触生成、动态顶点移动、固定点不动、穿透不增加、无翻转、
无关压缩四面体不冻结接触。独立 paper XPBD 回归也通过：静止状态无漂移/无翻转，受扰动
后距离和体积约束能量均下降。完整旧 smoke 的总 `passed` 仍为 false，是因为它要求只用
一帧就消除预置姿态已有的约 `2.8 mm` 深穿透；在 `0.015 mm/pass` 的有界接触设定下该
判据本来就不成立，不代表本次凸起或防翻转回归失败。

### 接触存在但器械仍穿过表面的后续修正（2026-08-08）

ThinLinc 实际播放表明：弱化向下传播后虽然不再制造大范围凹坑，但组织旁侧看到的凸起
主要来自初始表面；夹爪仍可能跑到表面以下。原因不是没有检测到接触，而是两个求解细节：

1. 旧 `0.015 mm/pass` 上限低于相邻视频帧中的器械法向位移；两次接触后又以材料投影
   收尾，距离/形状约束把直接接触面拉回静止形状。
2. 普通表面接触仍保留一个 mesh-wide 预缩放。视觉残差或早期接触只要造成一个被本轮
   接触触及的非正体积 tet，该 tet 就会把全局接触比例设为 0，导致所有健康接触点也不动。

当前改为 3 次接触：前两次各接两轮 distance/volume/shape 恢复，最后一次只执行有界
surface barrier，使显示状态以非穿透约束而不是材料回弹收尾。普通表面接触上限最终取
`0.040 mm/pass`；测试过的 `0.060 mm/pass` 虽能更快减小旧轨迹的深穿透，但会使非接触
邻域最低体积比降到约 `0.0014`，因此未保留。

普通表面接触的 mesh-wide 预缩放已删除。每个 proposal 先按粒子上限施加，再最多 8 轮
检查受影响四面体，只回退让体积继续恶化且低于 `0.03` 的移动节点。这样预先存在的一个
坏 tet 不再拥有“关掉整片夹爪接触”的全局否决权；persistent grip 的四个直接锚点仍
保留独立 incident-tet line search，不受该修改影响。

实际 ThinLinc 位姿驱动 `raw_paper_lnd_sam2_dense_contact_unbounded_xyz` 的 CPU 回归确认：
全局普通接触比例不再变成 0，而保持 1；全程无翻转、固定锚点漂移为 0。该离线 smoke
在状态 1795/2000 间直接跳过 205 个视频状态，因而会产生数毫米瞬时位姿跳变和主动抓取，
只用于验证局部接触不会被全局关闭，不能用其最大位移评价正常逐帧 GUI 播放。

## 夹取恢复与接触抖动抑制（2026-08-08）

### 根因与论文边界

这次“没有粒子被夹起、接触区剧烈抖动”不是论文的 distance/volume/shape PBD 本身直接
报告的问题。Liang et al. (arXiv:2309.11656) 把器械作用写成已知的 point-based
positional control `u_t`，实验轨迹由标注的夹爪位置提供；论文没有描述双刚性夹爪的
triangle-skin 碰撞、双侧接触门控、摩擦和自动粘附捕获。当前项目为了从器械几何自动
产生压入与夹取，额外增加了这一离散接触层，所以其 chatter/capture failure 必须在该层
单独处理，不能声称论文已经解决。

实际抖动由两个反馈环叠加：

1. 旧材料/接触投影的全部几何修正 `dx` 都以 `dx/dt` 写回粒子速度；下一子步又由该速度
   预测更大的位移，交替 contact/material projection 因而持续注入动能。
2. 抓取 compliance 为 `0.5 m/N`，对质量约 `2e-5 kg` 的表面节点仍太软，材料恢复会把
   节点从虚拟夹爪中点拉回；若直接提高追随步长，又会形成瞬时吸附和乱飞。

### 当前固定实现

速度更新现在显式分离积分预测和投影修正：

```text
v_next = damping * (v_pred + beta * (x_projected - x_pred) / dt)
beta = 0.12  (paper_pbd / paper_soft)
```

位置约束仍在当步完整生效，`beta` 只决定多少几何修正成为下一步的惯性速度；这不是降低
距离/体积/形状刚度，也不是限制视觉 residual 的位移。默认 `beta=1` 保留给其他场景，
因此修改只作用于 SUPER paper tissue 配置。

接触/夹取固定参数改为：

```text
substeps per video physics frame              12
triangle contact stride                       2 (6 contact solves/frame)
surface correction per contact pass           0.025 mm
contact passes per solve                       3
persistent grip direct anchors                 exactly 4
grip compliance                                0.05 m/N
grip correction cap per substep                0.10 mm
grip transfer                                  1 incident-tet layer
material local minimum volume ratio            0.01
```

原配置是 stride `3`、`0.040 mm/pass`，每帧总接触预算约 `0.48 mm`；新配置是更连续的
6 次小步，每帧总预算约 `0.45 mm`，因此没有通过削弱接触来掩盖抖动。抓取仍必须满足
q7 正在闭合/闭合、双侧各至少 8 个接触样本、连续 3 次求解、夹爪接触片间距不超过
6 mm，然后从夹爪中点附近、相邻四面体当前体积比不低于 `0.01` 的候选中选恰好 4 个
最近表面节点。这个健康度筛选不解除防翻转：它是在捕获前避开已经塌成 sliver 的节点，
否则一个坏锚点会把四锚点共享 line search 缩到接近零。q7 张开或相对捕获角增加
`0.08 rad` 时立即清除全部 attachment。

在线刚度更新新增 tet-quality gate：若 residual 前或后的最小体积比低于 `0.10`，视觉
位置修正仍可接受，但该帧不得作为刚度学习样本。shape stiffness 上限从 `0.040` 恢复到
论文搜索范围上限 `0.020`，避免接触退化时错误硬化到 `0.040` 后与夹爪反复争夺节点。

### 验证结果

CPU 回归通过：静止 paper XPBD 无漂移/无翻转，扰动后距离和体积约束能量下降；双侧
接触连续两次不捕获、第三次捕获恰好 4 个节点，张开后全部释放；triangle-skin 接触不
移动固定节点且不冻结无关压缩 tet。

首先用同一接触层的 `paper_soft v11` 做完整
`raw_paper_lnd_sam2_dense_contact_unbounded_xyz` 逐状态 GPU 回放
`1795..2000`（206 physics frames）：

```text
capture state                                  1807
selected / active direct anchors               4 / 4
final grip active                              true
median anchor motion after capture             3.914 mm
virtual jaw-frame motion                       3.323 mm in z
anchor target tracking error                   0.001--0.007 mm
maximum dynamic particle speed                 0.0576 m/s
maximum direct-anchor speed                    0.0142 m/s
inverted tetrahedra                            0
minimum tet volume ratio                       0.00254
material/grip safety scale                     1.0 throughout
```

同一轨迹旧实现曾达到约 `0.49 m/s` 最大粒子速度；现在下降约一个数量级。

随后对 GUI 实际使用的 `paper_pbd v15`（`1,465` 节点、`5,502` tet）单独回放捕获段
`1795..1830`。加入健康候选筛选前，虽在 1807 状态变为 active，但所选节点相邻 tet
最低体积比仅 `1.28e-6`，direct safe scale 最低 `3.81e-6`，所以视觉上等同未夹住。
筛选后的相同回放为：

```text
capture state / active anchors                 1807 / 4
selected-anchor min incident volume at capture 0.0333
minimum direct safe scale                      0.1277
median anchor motion by state 1830             0.875 mm
virtual jaw-frame motion                       0.625 mm in z
three-anchor tracking error                    <=0.04 mm
worst-anchor tracking error                    0.43 mm
maximum direct-anchor speed                    0.0139 m/s
maximum dynamic speed                          0.0758 m/s
inverted tetrahedra                            0
```

因此当前 `paper_pbd` 已从“active 但 line-search 锁死”恢复为真实移动。全局最低体积比仍
提示深压区存在与所选锚点无关的薄四面体，但没有翻转；投影速度比例 `0.12`、局部
`0.01` repair、捕获健康度 `0.01` 和刚度学习 `0.10` gate 分别阻止它影响惯性速度、
锚点选择与材料参数。后续若仍需进一步消除深压 sliver，应重做局部体积恢复，而不能删除
防翻转或继续放大 attachment correction。

## 在线视觉残差与刚度更新强档（2026-08-09）

### 为什么此前效果不明显

此前实时链路同时存在幅度、频率和材料学习三层限制：

```text
visual residual Adam iterations          6
visual residual learning rate            0.030 mm
per-update particle residual cap         0.40 mm
visual update interval                    4 physics frames
stiffness log learning rate               0.10
stiffness maximum log step                0.12
stiffness quality gate                    global min tet ratio >= 0.10
```

最后一项是主要结构性问题：接触区只要出现一个已有 sliver tet，全网格最小体积比就低于
`0.10`，于是整轮在线 distance/shape stiffness 学习被跳过。视觉位置 residual 仍会写回，
因此表现为轮廓有轻微变化，但后续 PBD 的材料响应几乎不改变。直接删除门控会把退化单元
的错误形变方向写入刚度，不能采用。

### 当前强档

```text
visual residual Adam iterations          8
visual residual learning rate            0.040 mm
per-update particle residual cap         0.60 mm
visual update interval                    3 physics frames
stiffness log learning rate               0.18
stiffness maximum log step                0.18
residual / deformation full scale         0.20 / 0.75 mm
shape update gain                         1.50
stiffness spatial smoothing               1 graph pass
shape stiffness bounds                    0.004 .. 0.020
distance stiffness bounds                 0.20 .. 1.60
volume stiffness                          fixed 1e10
```

`0.18` log step 对应单轮最大乘数 `exp(0.18)=1.197`，即约 20%，仍是有界更新。shape 上限
保持 Liang et al. 的 `0.02` 搜索范围上界，volume stiffness 继续固定，不因追求明显效果
而偏离论文的固定体积约束。

刚度质量门控改为逐粒子局部门控。每轮分别计算视觉修正前后的 tet volume ratio；若某个
tet 任一状态低于 `0.03` 或非有限，只屏蔽该 tet 四个顶点本轮的 stiffness update，其余
健康节点继续学习。局部无效节点的 EMA 历史同时清零，避免节点刚变坏后仍被上一轮信号
继续硬化。GUI 新增 `Stiffness local tet quality valid / masked`，日志记录
`stiffness_quality_masked`，可直接确认在线刚度是否真正执行。

### 回归

在线刚度合成回归通过：朝静止形状的视觉 residual 会硬化，远离静止形状会软化，固定点
永不改变；将一个动态节点标记为坏 tet 邻域后，该节点保持基线，而相邻健康动态节点仍
正常更新。paper XPBD 独立回归继续保持静止无漂移、无翻转，扰动后距离和体积约束能量
下降。增强幅度仍需在 ThinLinc GUI 连续播放中以实时 loss、最大 residual、local masked
数和刚度 min/median/max 继续核验，不能把合成回归冒充真实图像效果。

## 压入后侧向隆起与抓取恢复（2026-08-09）

### 可恢复基线

调整压下策略前的完整参数和最后一次 paper_pbd 捕获段指标已冻结为：

```text
outputs/config_snapshots/online_strong_v1_pre_press_rework_20260809.json
```

该文件包含 v15 网格计数、paper PBD 参数、在线视觉/刚度强档、旧接触顺序、persistent
grip 参数以及状态 `1795..1830` 的捕获指标。后续对压入策略的修改不覆盖该快照。

### 旧策略为何把周围一起压下

旧接触顺序为：

```text
full contact -> 2 material passes
full contact -> 2 material passes
full contact barrier (0.025 mm) -> display
```

最后一轮完整 barrier 没有材料恢复，会把前两轮 volume projection 刚形成的侧向/向上位移
再次压低。同时每个直接顶部三角面还把最多三个 edge-adjacent surface node 按 `15%`
比例明确写成向下位移。两者叠加后，直接接触和周围一环都沿 `-z` 运动，形成宽凹坑；这
也会在正式捕获前把候选锚点相邻 tet 压薄，使 persistent grip 的 no-flip line search
减小，从而降低抓取效果。

### 当前策略

新顺序保持每个接触子步总计三次几何查询和四次材料恢复，计算规模基本不变：

```text
full direct contact (0.025 mm)
  -> 2 x distance + fixed-volume + shape recovery
full direct contact (0.025 mm)
  -> 2 x distance + fixed-volume + shape recovery
weak final non-crossing barrier (0.005 mm)
```

邻接表面环的 prescribed downward weight 从 `0.15` 设为 `0.0`；paper contact spreading
仍为 `0` 层。也就是说只有器械直接命中的碰撞面被压，周围表面和内部节点不接收人为的
向下位移，而由 `k_volume=1e10`、distance 和 shape projection 决定。最后 `5 um` barrier
只保留视觉非穿透提示，其幅度为主接触的 20%，不足以把材料生成的 shoulder bulge 压平。

### 真实轨迹前后对比

固定使用 GUI 相同的 `paper_pbd v15`、原始 q7、零手工位姿偏移，逐状态回放
`1795..1830`：

```text
metric                                      old       current
capture state                               1807      1807
active direct anchors                       4         4
3mm press ROI median z displacement         +0.139    +0.309 mm
3mm ROI depressed below -0.1mm              6/13      5/12
minimum grip direct safe scale               0.1277    0.2195
worst direct-anchor target error             0.43      0.006 mm
maximum direct-anchor speed                  0.0139    0.0141 m/s
maximum dynamic particle speed               0.0758    0.0628 m/s
minimum global tet volume ratio              5.4e-7    0.00559
inverted tetrahedra                          0         0
```

因此新策略同时增强了局部上凸、改善了锚点可移动性，并降低了非锚点速度；不是通过取消
防翻转或直接给周围节点施加向上动画得到的。`3 mm ROI` 仍包含五个真正位于夹爪正下方
的下沉节点，这是正确的直接压入区；目标不是让所有节点上升，而是形成“中心下压、肩部
上凸”的空间分布。

### 相机左侧压凸修正（2026-08-09）

继续按 stereo-left 相机坐标单独检查后，发现上一轮的整体 `3 mm ROI` 中位数
不能代表用户指定的左侧：在抓取建立前的状态 `1805`，左侧 `1.5..5 mm`
外环实际是 `5/5` 节点下沉，中位为 `-2.91 mm`。根因有两层：

1. `top_support_lateral_radius=3 mm` 被错误复用成 top barrier 的横向容差，使真实工具
   footprint 外的表面也可能被当成直接下压面。
2. 封闭夹爪网格的上面、侧面和下面所有样本均进入 oriented top barrier，侧面样本
   也会被投影到组织顶面并伪造向下压入。

当前改为：

```text
top barrier lateral tolerance              0.45 mm
top barrier samples                         tissue-facing half of each closed jaw mesh
camera-left shoulder direction              world -x (stereo-left screen-left)
left/right split                            current bilateral contact center
left shoulder start                         0.0 mm from split
shoulder candidate radius                   4.5 mm
shoulder upward scale                       1.0
shoulder maximum correction/pass            0.050 mm
shoulder timing                             after contact + material recovery
shoulder safety                             local tet no-flip repair
after four-particle grip capture            shoulder disabled
```

曾实测对称地把整个外环强制上抬，虽然能让整体 ROI 变凸，但会把抓取锨点
附近四面体压成 sliver，该方案已拒绝，没有作为最终配置。当前定向肩部只修正
视频中确定应向上排量的相机左侧，中心/右侧仍保持器械直接下压。

使用 GUI 相同 `paper_pbd v15`、原始 q7 和零位姿偏移实测：

```text
metric                                           before       current
pre-grip state                                   1805         1805
stereo-left shoulder median z                    -2.91 mm     +1.63 mm
left shoulder raised >0.1 mm                     0/5          7/8
left shoulder depressed <-0.1 mm                 5/5          1/8
pre-grip minimum tet volume ratio                0.0100       0.0127
full-segment capture state                       1807         1807
full-segment active direct anchors               4            4
full-segment worst anchor target error           ~0.006 mm    0.018 mm
full-segment minimum grip direct safe scale      0.2195       0.0484
full-segment maximum dynamic speed               0.0628       0.0627 m/s
full-segment minimum tet volume ratio             0.00559      0.01083
inverted tetrahedra                              0            0
physics median                                   43.48 ms     48.40 ms
```

新策略使左侧压凸成为可单独验收的相机坐标指标，而不再用混合了直接下压点的
整体 ROI 指标代替。抓取安全步长相比上一轮有所下降，但完整段仍正常捕获四个锨点，
直接锨点最终安全比例恢复为 `1.0`，无四面体翻转。

### 取消主动膨胀并改为相对体积 XPBD（2026-08-09，覆盖上一节的最终配置）

上一节的 camera-left pressure shoulder 在 GUI 连续物理子步中会重复写入向上位移；
而且 biased direct override 会把左半侧直接接触节点原本的向下接触改为向上。它虽然能
制造左侧指标为正，却不是材料约束产生的形变，并造成用户观察到的大面积非物理膨胀。
该方案现已明确拒绝并完全关闭：radius/depth/up/out 均为 `0`，bias direction 为零向量，
积分器在 entry count 为零时不再执行 shoulder pass。

进一步诊断发现原 volume constraint 直接使用 SI 米制的 `V - V0`。毫米尺度四面体的
体积和梯度极小，导致 `k_volume=1e10` 仍允许受压 tet 缩到静止体积约 `1%`。现改为
无量纲相对体积约束：

```text
C_V(x) = V(x) / V0 - 1
∇C_V   = ∇V / V0
```

这不改变论文基线参数 `k_volume=1e10`，但消除了网格尺度/米制单位对求解强度的影响。
顶面防穿透继续只接受夹爪朝向组织的一半样本，并新增以双侧接触中心为圆心的
`1.2 mm` 直接压入核心。该半径同时检查 query skin point 和三个物理三角面顶点，
避免“接触点在核心内，但大三角面的核心外顶点仍被同幅度压下”的离散化扩张。
核心外没有向上或向下的 prescribed displacement，只通过 distance + relative-volume
+ shape XPBD 响应。每次 full contact 后的 material recovery 从 2 次增加到 4 次。

相同 `paper_pbd v15`、原始 q7、零手工位姿偏移的 CUDA 逐状态回放结果：

```text
metric                                             rejected shoulder   current
pre-grip maximum dynamic displacement              约 5.54 mm          1.058 mm
pre-grip minimum tet volume ratio                   约 0.010            0.839
pre-grip left median z                              -2.87 mm(*)         +0.159 mm
pre-grip left raised/depressed >0.1 mm              0/5 / 5/5           6/9 / 0/9
full-segment capture state                          1807                1807
full-segment direct anchors                         4                   4
full-segment left raised >0.1 mm                    --                  9/9
full-segment left median z                          --                  +0.828 mm
full-segment press ROI median z                     --                  +0.906 mm
full-segment maximum dynamic displacement           --                  1.933 mm
full-segment minimum tet volume ratio               0.0108              0.553
maximum particle speed                              0.0627 m/s          0.0185 m/s
inverted tetrahedra                                 0                   0
physics median                                      48.4 ms             52.6 ms
```

`(*)` 为完全关闭人工 shoulder、但尚未归一化体积和缩小 footprint 的中间基线。当前
最终左侧上凸来自抓取与相对体积约束的完整动力学结果，不再由相机左侧定向上推动画生成。

### 回退到旧压下策略的温和版本（2026-08-09，当前启用）

用户确认上一节的局部体积版本压下视觉效果不足，因此当前恢复冻结快照
`online_strong_v1_pre_press_rework` 的接触顺序：前两次完整接触各跟两次材料恢复，
第三次完整接触作为最终 barrier。在线残差、在线刚度和四粒子抓取参数保持不变。

旧版邻域下压传播没有原样使用 `15% / 3 mm`，而按用户要求缩小为：

```text
top support weight       0.15 -> 0.05
top support radius       3.0 mm -> 1.0 mm
top barrier lateral tol  3.0 mm -> 1.0 mm
```

人工 pressure shoulder 仍为关闭状态；`1.2 mm` 顶点接触核心和相对体积约束不再作为
当前运行配置。CUDA 状态 `1795..1805` 验证三次接触生效、无四面体翻转，物理中位耗时
约 `43.0 ms`。

### 夹爪闭合运动推动（2026-08-09，当前启用）

逐帧 LND/q7 位姿原本会在 12 个物理子步中插值，但插值只写 `body_q`，没有写
`body_qd`。因此 jaw triangle contact 计算 tool-relative tangential velocity 时看到的
夹爪速度接近零，闭合阶段缺少“夹爪侧面把组织推向中间”的效果。

当前由相邻视频/控制帧的刚体姿态计算每个 LND 刚体的 COM twist：

```text
v_com = (c_target - c_start) / dt_frame
q_delta = q_target * inverse(q_start)
omega = axis(q_delta) * angle(q_delta) / dt_frame
v_sample = v_com + omega x (p_sample - c)
```

该速度随同一条 shortest-path quaternion slerp 写入每个物理子步的 `body_qd`。夹爪闭合
时，现有 jaw contact 使用 `v_sample` 产生切向接触修正；修正仍受 Coulomb 摩擦上限、
每次 `0.025 mm` 接触上限、材料恢复和 tetra no-flip 检查约束。它不是粘附：只有 q7
闭合门控和双侧候选满足后，才会继续建立四粒子持续抓取。

CUDA 原始轨迹 `1795..1830` 验证：捕获状态仍为 `1807`，直接锚点 `4`，最小抓取
safe scale `0.104`，四面体翻转数 `0`，物理中位耗时约 `44.5 ms`。

### 接触位移/材料位移分离与局部柔软压入（2026-08-09，当前启用）

上一节虽然补全了夹爪刚体 `body_qd`，但最终速度回写仍把接触和材料投影合并成
一个总位移，并统一乘 `0.12`。这会同时削弱真实闭合推动。另一方面，顶部邻接环仍
按 `5% / 1 mm` 接收人为向下位移，不能区分材料自然传播与绘制的压下效果。

当前在每次被 no-flip 修复接受后的 triangle-skin contact 前后保存粒子位置并累加：

```text
dx_contact  = sum_k (x_after_contact_k - x_before_contact_k)
dx_total    = x_final - x_predicted
dx_material = dx_total - dx_contact

v_next = damping * (
    v_predicted
    + 0.12 * dx_material / dt
    + 0.50 * dx_contact  / dt
)
```

因此距离、相对体积、形状恢复仍只以 `0.12` 回写惯性速度，避免交替投影产生抖动；
真实夹爪接触以 `0.50` 保留速度传递，不再被材料稳定系数一并吞掉。这里没有增加
外力场或粘附力，粒子位移仍只来自几何非穿透/切向接触。

闭合推动另修正了上表面接触的法向退化：组织上表面法向近似竖直，而 q7 闭合运动
主要水平，旧 Coulomb 上限 `mu * penetration` 会在竖直穿透很小时把水平推动压到近零。
当前从夹爪铰链运动学计算单位 q7 闭合方向，只在工具相对速度确实沿闭合方向、且
triangle-skin contact 已存在时，为该分量采用局部 no-slip 距离；最终仍受每 pass
`0.025 mm`、局部 no-flip 和材料投影限制。

当前局部形变配置：

```text
人工邻域下压权重                     0.05 -> 0.0（关闭）
显式 pressure shoulder              0.0（保持关闭）
直接接触核心半径                     1.25 mm
目标完整可见响应范围                 约 2--3 mm（由 XPBD 自然传播）
distance stiffness                  0.40 -> 0.35
shape stiffness                     0.008 -> 0.006
relative-volume stiffness           1e10（不变）
material/contact velocity scale     0.12 / 0.50
```

`1.25 mm` 仅限制双侧接触中心建立后的直接 jaw/top-barrier 位移；约 `2 mm` 左侧不再
属于向下约束，是否上凸完全由不可压缩四面体的距离、相对体积和形状投影决定。双侧
尚未建立时不启用半径 mask，避免因为没有中心而阻断初次接触。

使用 GUI 相同的 `paper_pbd v15`、原始 q7、零手工位姿偏移，在 CUDA 回放
`1795..1830`：

```text
capture state                               1807
active direct grip particles                4
left 1.5--5 mm shoulder maximum z          +0.333 mm
left shoulder nodes raised >0.1 mm          3 / 7
3 mm press ROI median z                    -0.0615 mm
maximum dynamic displacement                3.323 mm
maximum particle speed                      0.144 m/s（初始深重叠状态）
maximum direct-grip particle speed          0.0152 m/s
minimum grip direct safe scale              1.0
minimum tetrahedron volume ratio            0.01283
inverted tetrahedra                         0
physics median                              60.7 ms
```

这组结果取代上两节的 `5% / 1 mm` 人工邻域下压配置和统一 `0.12` 接触速度配置。

### 四点夹爪位置控制 `u_t` 可行性原型（2026-08-09，当前安全门控启用）

为验证论文式点位置控制能否承担夹持区域的主要运动，原 persistent grip 从“夹爪
接触斑中点附近四个节点跟随虚拟中间刚体”改为显式的双侧 `u_t`：输出槽 `0..1`
只从 jaw A 实际接触节点中选择，槽 `2..3` 只从 jaw B 实际接触节点中选择；四个点
必须唯一且捕获时相邻四面体健康。每个点保存到对应物理夹爪的局部坐标系，之后的
目标由各自夹爪逐帧位姿直接生成。因此 q7 继续闭合时，两侧目标可以相对运动，不再
只有整体中夹爪平移。

正式场景门控同步收紧：

```text
maximum capture correction penetration    10.0 mm -> 0.8 mm
contact margin included above               0.4 mm
maximum permitted signed overlap            0.4 mm
minimum incident-tet volume ratio           0.01 -> 0.20
anchors                                      2 per jaw, actual contact only
```

CPU 独立门禁 `scripts/test_super_tissue_persistent_grip.py` 已验证：每片夹爪各选择两个
接触节点、第三次持续双侧接触才捕获、共同抬升时四点跟随、继续闭合时左右 `u_t`
目标不同、打开后全部释放。`scripts/test_super_paper_soft_mode.py --device cpu` 和普通
triangle-skin contact 回归也全部通过。

使用 `paper_pbd v15`、零位姿偏移、state `1600..2050` 连续 CUDA 回放得到两组结果：

```text
metric                                      safe defaults       diagnostic loose gate
capture state                               none                1807
direct anchors                              0                   4 (2 per jaw)
initial minimum signed distance             -7.026 mm           -7.026 mm
capture minimum selected volume ratio       --                  0.0113
u_t target motion after capture             --                  4.8--5.4 mm
actual median anchor motion                 --                  4.08 mm
final target tracking error norm            --                  0.43--1.02 mm
minimum direct safe scale                   1.0                 4.77e-6
minimum tetrahedron volume ratio            0.0100              3.05e-8
inverted tetrahedra                         0                   0
```

宽松门仅用于隔离机制验证，显式指定 `10 mm / 0.01`，没有写回正式配置。它证明每夹爪
位置控制可以产生夹持后的主要运动，但当前深重叠接触会让局部四面体先接近塌缩，
no-flip line search 随后把 `u_t` 步长压到几乎为零，不能作为可接受结果。安全配置
正确拒绝整段捕获；后期两片粗代理面还会命中相同组织节点，只剩两个唯一健康候选。
因此下一阶段必须先修复初始接触表面对齐/接触开启时机，再重新验收安全门控下的
四点 `u_t`，不应通过放宽捕获阈值掩盖几何错误。现有左右目 2D 距离场未修改。

补充 GUI 式全历史回放：上一段 `1600..2050` 从静止组织直接跳到 state 1600，所述
“安全配置拒绝捕获”仅适用于这种跳帧初始化，不能代表 GUI 从第 0 帧连续播放。使用
同一安全默认值连续回放完整 `0..5457` 后，结果为：

```text
capture state                                2493
capture signed distance                     -0.397 mm
capture selected minimum volume ratio        0.726
direct anchors                               4 (2 per jaw)
capture proposal nonpositive tetrahedra      0
release by end                               YES, final anchors 0
minimum direct safe scale                    1.91e-6 at state 3929
final minimum tetrahedron volume ratio       0.00412
inverted tetrahedra                          0
```

因此完整历史下，四点 `u_t` 的捕获和 q7 释放状态机是可工作的；但抓取后半段目标仍被
局部 no-flip 几乎完全阻断，不能据此声称画面已经自然完成“夹起—放下”。正式 GUI
必须从 Reset 后第 0 帧连续 Play 才能复现该状态；直接拖动到接触帧会缺少组织历史，
产生上一段测得的 4--7 mm 深重叠并拒绝捕获。

### GUI 命名侧视角（2026-08-09，当前启用）

SUPER 3D viewer 原本只在启动代码中直接写入一次侧视相机私有字段；点击左/右目
`Go` 后没有明确方式回到同一物理观察方向，通用 `Reset` 也会返回另一套默认相机。
现在 `EmbodiedViewer` 支持注册并激活经过有限值、非零方向和非平行 up-vector 检查的
命名侧视位姿。SUPER 注册参数保持原先已经目视确认的方向：

```text
camera position world       (-0.18, -0.02, 0.053) m
camera front                normalize(0.995, 0.0, -0.100)
camera up                   (0.0, 0.0, 1.0)
```

相机从原 `x=-0.25 m` 向场景移近到 `x=-0.18 m`，并同步调整高度以保持原观察中心，
侧面内容约放大 `1.4x`，不改变夹爪/组织的物理状态或渲染分辨率。

GUI 启动时自动激活该视角；`Controls -> Camera Controls` 新增绿色 `Side View` 按钮，
用户点击 `Go` 查看任一真实双目相机后可一键返回侧面，观察夹爪压入深度、组织厚度、
四点抓取随动和释放回弹。该按钮复用现有 3D viewport，不增加第二次物理求解或 Gaussian
渲染。`py_compile`、headless 视角归一化/切换测试以及示例 CLI 完整导入均通过。

### 双夹爪 `u_t` 局部安全比例解耦（2026-08-09，当前启用）

完整历史回放暴露出原四点位置控制共用一个 `persistent_grip_direct_scale`：任一夹爪侧
的 incident tet 接近 no-flip 下限时，另一片夹爪的两个健康控制点也会被同步缩小。当前
保持每片夹爪内部两个点的共同刚体目标，但按 jaw A、jaw B 顺序分别执行局部体积线搜索；
第二片夹爪的检查基于第一片已经接受的位置，因此组合结果仍受体积门禁约束。夹持控制的
最低体积比与普通局部接触统一为 `0.03`；已经低于该值的单元只允许保持或改善，不能被
`u_t` 继续压缩。左右目 2D 距离场和视觉残差路径没有修改。

CPU 合成门禁构造了 jaw A 会翻转、jaw B 完全安全的两个控制斑：jaw A 被局部缩放后，
jaw B 仍以 `1.0` 完整移动，两个 tet 均保持 `J>=0.03`。既有四点捕获、每夹爪两点、
共同抬升、继续闭合时左右目标不同和 q7 打开释放门禁全部通过。GUI 新增
`Grip jaw A / B safe scales`，可在放大的 `Side View` 中区分是哪片夹爪受限。

使用 GUI 相同 `paper_pbd v15`、原始 q7、零位姿偏移，从 state `0..4000` 连续 CPU
回放（不是接触段跳帧）得到：

```text
capture state                              2493（与解耦前一致）
direct anchors                             4（每片夹爪 2）
jaw A minimum safe scale / state           9.54e-7 / 3948
jaw B minimum safe scale / state           1.91e-6 / 3969
final jaw A / B safe scales                1.91e-6 / 1.62e-5
minimum tetrahedron volume ratio           9.75e-6
inverted tetrahedra                        0
CPU physics median                         595.3 ms/frame
```

结论是：解耦修复了“一个坏单元冻结四点”的错误，但不足以完成自然夹起；两片夹爪各自
仍会在后段压瘪控制点周围的局部 tet。下一步不能放宽 no-flip 或捕获阈值，而应把每片
夹爪的两个表面 `u_t` 转换成有限体积控制斑：将同侧目标以体积保持的权重传给其 incident
tet 内部节点，再逐 jaw 验证目标跟踪、`J>=0.03`、释放回弹和 GUI 侧视效果。

### 每夹爪有限体积 `u_t` 控制斑（2026-08-09，当前启用、待 CUDA 全历史验收）

上一节所述有限体积控制斑已经实现。每个物理子步先计算四个真实表面控制点的 jaw-local
目标残差，然后分别对 jaw A、jaw B 执行：

```text
筛出同侧两个 u_t -> incident tet 一层平均刚体传播
-> 删除四个真实锚点的 support 位移
-> 对内部 support 节点做局部 J>=0.03 修复
-> 对同侧两个真实锚点做 jaw-local line search 和最终投影
```

内部节点不是新增粘附点，也不保存夹爪局部坐标；它们只接收当步同侧控制的平均刚体
分量，为表面锚点腾出有限体积。另一片夹爪的锚点会从 support pass 中排除，左右闭合
方向不会被混合平均。左右目 2D 距离场、RGB residual、在线刚度和捕获/释放门控均未改。

新增 CPU 合成门禁把一个表面点的目标设置为足以直接翻转单位 tet 的位移。仅表面控制
会受限；加入有限体积传播后四个节点形成整体平移，jaw scale=`1.0`、最终体积比=`1.0`。
原四点选择、三次捕获、每 jaw 两点、不同闭合目标、单侧受阻不冻结另一侧和 q7 释放
门禁全部通过；完整 `paper_soft` 场景静止 20 步仍保持 finite、无位移、最低体积比
`0.9999986`。

真实 `paper_pbd v15` 的 `1795..1830` 跳帧宽门隔离回放只用于确认实际拓扑接线：

```text
capture / direct anchors                    1807 / 4
finite-volume proposal affected particles  19
median anchor motion after capture          1.207 mm
target tracking error                       约 0.1--0.3 mm
minimum tet volume ratio                    0.01005
inverted tetrahedra                         0
```

该回放从静止组织直接进入 `1795`，初始 signed distance 为 `-7.12 mm`，并显式使用仅供
机制隔离的 `10 mm / 0.01` 宽捕获门；捕获时已有锚点 incident `J=0.016<0.03`，因此不能
用其低 jaw scale 否定或验收正式配置。下一项正式门禁仍是 Reset 后 state `0..5457`
连续 CUDA 回放，要求安全 `0.8 mm / 0.20` 门控下完成捕获、夹起、放下、释放，且抓取
后 `J>=0.03`、jaw A/B scale 不长期接近零。通过该门禁后立即进入 RGB residual 优化，
而不是继续修改接触几何。

### RGB 残差与实时修正的进入条件（2026-08-09）

RGB residual 不是尚未接入：当前正式 GUI 已使用左右目 2D 图像、器械 2D 距离场和
Gaussian RGB Smooth-L1 做在线物理节点修正，当前强档为 `8` 次 Adam、`0.040 mm`
学习率、每节点 `0.60 mm` 上限、每 `3` 个物理帧更新一次，并启用局部质量门控的在线
distance/shape 刚度更新。该求解目前在 physics coroutine 内同步执行，所以“在线”表示
播放中实时闭环，不等于已经达到无阻塞 30 FPS。

进入下一轮 RGB 优化只等待上一节的一个正式物理门禁，不再等待新的三维视觉：

```text
Reset 后 state 0..5457 连续 CUDA 回放
安全捕获门                         0.8 mm / J>=0.20
四点捕获、最终 q7 释放             必须完成
抓取后 incident tet                J>=0.03、无翻转
jaw A/B safe scale                 不得长期接近零
Side View                          能看到压入、夹起、放下和释放回弹
```

通过后 RGB 工作按以下顺序立即开始：先在固定物理回放上分别记录左/右 loss、有效像素、
梯度方向和左右一致性，修正 mask/SDF/遮挡错误；再把同步 residual solve 改为单任务
double-buffer 异步求解，物理循环持续运行，只在帧边界应用最新且未过期的安全修正；最后
才比较 `4/6/8` 次迭代和更新间隔，按 loss 降幅、延迟、局部冻结数和体积质量选档，而不
先继续增大 residual 幅度。整个阶段仍只使用左右目 2D 图像，不引入 3D 视觉输入。

当前沙箱没有 CUDA 设备；尝试申请主机 CUDA 执行完整门禁未获执行权限。因此有限体积
控制斑已通过 CPU 合成和场景接线回归，但上述 `0..5457` 正式结果仍待 ThinLinc/CUDA
环境运行，不能把跳帧宽门隔离结果冒充完成。

### 温和下压与更长尖端许可 A/B（2026-08-09，当前试用档）

当前 v15 物理网格并不密：`1,465` 节点、`5,502` tet，物理表面边长中位数约
`3.51 mm`；`26,754` 个 visual face/Gaussian 只提高渲染和 RGB residual 分辨率，不会
改善物理接触。局部增加物理 tetra 能缩小一个接触样本通过粗三角形三个顶点造成的离散
footprint，使凸起更平滑、更局部，但在强相对体积约束下不能消除真实的侧向排量。全局
加密还会明显增加接触/材料耗时和 sliver 风险，因此若后续加密，应只把夹取轨迹周围
约 `6 mm` ROI 的目标边长降到约 `1.8--2.0 mm`，外围保留当前分辨率。

先按用户建议做不改网格的温和 A/B。当前 oriented top barrier 原本已经排除夹爪最前
`1.5 mm`，该段可以进入顶面以下；普通 jaw contact 仍在完整 distal `5 mm` 上提供
切向 q7 推动和抓取候选。试用档改为：

```text
surface correction cap / pass        0.040 -> 0.030 mm
allowed tip entry                    1.5 -> 1.8 mm
remaining top-barrier band           3.5 -> 3.2 mm
jaw tangential contact/grip band      5.0 mm（不变）
```

`0.030 mm` 仍高于实测最坏相邻子步夹爪运动约 `0.023 mm`，不会简单关闭 barrier。对同一
静止组织直接跳到 `1795..1805` 的 CPU 机制隔离（不是正式连续轨迹）比较：

```text
metric                              0.040/1.5   0.030/1.5   0.040/1.8   0.030/1.8
top barrier enabled samples             1033        1033         948         948
maximum dynamic displacement (mm)      7.656       7.202       7.371       7.202
maximum particle speed (m/s)           0.192       0.144       0.192       0.144
minimum tet volume ratio              0.0100      0.0132      0.0102      0.0110
inverted tetrahedra                        0           0           0           0
```

因此降低 correction cap 是减小冲击的主要来源；增加 `0.3 mm` 尖端许可主要减少 barrier
覆盖。跳帧初始化自带约 `-7.1 mm` 深重叠，左右 shoulder 符号和幅度不能代表 GUI 从
Reset 连续 Play 的视觉结果。组合档当前只作为 GUI 试用：重点看 Side View 中左侧凸起、
尖端进入深度、是否仍能建立四点抓取。若尖端过深，优先把 allowance 单独退回 `1.5 mm`，
保留较温和的 `0.030 mm/pass`；不应先通过加密整个网格解决。

### 法向下压与闭合抓取解耦（2026-08-09，当前试用档）

连续 GUI 观察指出，单爪下压不只压低自身正下方，旁边随后要进入两爪之间的组织也被
提前压下。上一档虽把统一 correction cap 降到 `0.030 mm/pass`，但仍有两个结构问题：

1. 顶面法向 barrier 与 q7 切向闭合共用同一个 correction cap，继续整体降低会同时削弱
   后续夹取；
2. `5 mm` distal jaw band 原本既负责闭合/抓取候选，也让尖端许可后的整段 `3.2 mm`
   承担法向压下，后段会提前压低待夹取组织。

当前将两种职责正式分开：

```text
ordinary jaw tangential/grip band           5.0 mm（不变）
ordinary jaw correction cap                 0.030 mm/pass（不变）
tip entry allowance                         1.8 mm（不变）
top-barrier distal extent                   5.0 -> 3.0 mm
actual top-barrier band width               3.2 -> 1.2 mm
top-barrier normal correction cap           0.030 -> 0.015 mm/pass
top-barrier lateral tolerance               1.00 -> 0.45 mm
```

实现上新增独立 `triangle_skin_top_barrier_max_correction_m`：它只限制合并到节点的
`-z` top-sheet barrier 分量，普通 jaw contact 的 q7 切向分量仍可使用 `0.030 mm` 总预算。
同时新增独立 `top_barrier_distal_length_m`，不再通过缩短 jaw contact/grip band 来缩小
法向 footprint。

与上一档同样从深重叠静止组织跳到 `1795..1805` 的 CPU 机制隔离中，预计算的
top-barrier 样本从 `948` 降到 `358`（约 `-62%`），jaw contact/grip 样本仍为 `1714`；
无四面体翻转，最低体积比 `0.01377`。该跳帧中最大速度约 `0.119 m/s`，把独立法向上限
临时改回 `0.030 mm` 后仍约 `0.119 m/s`，说明这一最大值来自深重叠/切向运动，不能用来
判断连续 GUI 下压观感。正式验收仍必须 Reset 后连续播放，在 Side View 检查待夹取组织
是否不再跟随单爪一起下沉、尖端穿入量是否可接受、以及两侧各两个 `u_t` 是否仍能捕获。

CPU `paper_soft` 静态回归、通用 triangle-skin contact、四点/每夹爪有限体积 `u_t` 捕获与
释放测试全部通过。当前环境无 CUDA，尚未宣称连续真实轨迹的 GUI 视觉结果通过。

### 更深尖端许可与弱法向窄带（2026-08-09，当前试用档）

连续 GUI 仍显示单爪下压过强、旁边待夹取组织被带着下沉，因此在保持 `5 mm` 普通
jaw tangential/grip band 和 `0.030 mm/pass` 总接触预算不变的前提下，只继续放松
top-sheet 法向 barrier：

```text
tip entry allowance                         1.8 -> 2.3 mm
actual top-barrier band                     1.2 -> 0.7 mm（2.3..3.0 mm）
top-barrier normal correction cap           0.015 -> 0.008 mm/pass
top-barrier lateral tolerance               0.45 -> 0.30 mm
ordinary jaw tangential/grip band            5.0 mm（不变）
ordinary jaw correction cap                  0.030 mm/pass（不变）
```

同一 `1795..1805` CPU 深重叠机制隔离中，预计算 top-barrier 样本从上一档 `358` 降到
`211`，相对最初 `948` 降约 `78%`；jaw contact/grip 样本保持 `1714`。最低四面体体积比
由上一档 `0.01377` 提高到 `0.01945`，翻转数仍为 `0`。最大速度仍约 `0.119 m/s`，说明
其来源仍是跳帧深重叠/切向运动，而不是当前法向 cap。启动日志打印
`top_barrier_band=2.3..3.0mm`，便于确认 GUI 实际加载了新档。

这一档明确以“允许更多尖端进入，尽量不提前压低待夹取组织”为目标。正式判断仍需
Reset 后连续播放 Side View；如果尖端穿透可接受但周围仍明显下沉，下一步应改接触三角
顶点的法向分配或局部加密物理接触面，而不是继续无限降低同一个 barrier cap。

### 更松顶面与一环柔性夹起约束（2026-08-09，当前试用档）

按用户要求保持尖端进入许可 `2.3 mm` 不变，继续放松法向组织平面，同时让夹爪外侧
紧邻四个 `u_t` 主锚点的一小圈表面粒子也能随夹爪抬起。当前参数为：

```text
tip entry allowance                         2.3 mm（不变）
top-barrier normal correction cap           0.008 -> 0.004 mm/pass
top-barrier band                             2.3..3.0 -> 2.3..2.8 mm
top-barrier lateral tolerance                0.30 -> 0.20 mm
top-barrier clearance                        0.40 -> 0.15 mm
ordinary jaw contact margin                  0.40 mm（不变）
ordinary jaw tangential/grip band             5.0 mm（不变）
local soft lift support                       one surface-edge ring, <=3.5 mm
support propagation generations              1
```

此前 `top_barrier_clearance_m` 实际只被普通 jaw contact 使用，而 oriented barrier 仍误用
统一 `contact_margin_m`。当前将两者正确分开：普通 jaw triangle contact 继续用 `0.4 mm`
margin，保留 q7 闭合和抓取候选；只有 oriented top-sheet barrier 使用 `0.15 mm` clearance。
因此平面更晚介入，同时不通过削减抓取候选来制造“放松”。

旧的 support-radius 路径曾被四点严格抓取的校验显式禁用。当前根据新需求重新启用，但
不采用纯欧氏半径：欧氏球可能跨过薄组织抓到底面。support 邻接现由 collision-skin
三角形共享边构造，只允许与四个直接锚点相连的一条表面拓扑环；边长还需不超过
`3.5 mm`。目标权重按距离 Gaussian 衰减，四个 `u_t` 仍是唯一权重 `1.0` 的主锚点；
周围粒子使用相同 compliant correction、有限体积传播和局部防翻转，q7 松开时全部清除。

机制隔离结果：

```text
asset                   direct u_t   soft ring   barrier samples   support safe scale   inverted
paper_soft v11                  4           4              133                 1.0          0
paper_pbd v15                   4           8              133                 1.0          0
```

两套网格的 soft-ring 数不同是局部表面拓扑密度差异，不是半径无限扩张。宽捕获门仍只用于
`1795..1830` 跳帧机制隔离；正式配置继续使用 `0.8 mm / J>=0.20` 捕获门。CPU 静态
`paper_soft`、通用 triangle-skin contact、四点+一环随动、第二环不传播以及 q7 释放清空
测试全部通过。当前无 CUDA，连续 GUI 仍需 Reset 后用 Side View 验证平面下沉、局部夹起
范围和释放回弹。

### 异常膨胀修复：中等 barrier 与共同腕部外圈（2026-08-09，当前试用档）

用户在上一档观察到异常膨胀。代码审计确认不是单一参数问题，而是两项机制叠加：

1. oriented top barrier 被同时降到 `0.004 mm/pass`、`0.5 mm` band 和 `0.15 mm`
   clearance，器械可积累更深交叠；固定相对体积约束随后必须把这部分排量转移到侧面，
   因而会放大可见凸起；
2. 新增一环粒子沿其最近直接锚点继承 jaw A 或 jaw B 局部坐标。捕获后 q7 继续闭合时，
   外圈会被两个不同运动的夹爪坐标系横向拉扯；而用户只要求它们随整体夹起，不要求
   它们参与双爪相对闭合。

当前把下压恢复到上一强弱档之间，而不是回到最初的强压：

```text
tip entry allowance                         2.3 mm（不变）
top-barrier normal correction cap           0.004 -> 0.008 mm/pass
top-barrier band                             2.3..2.8 -> 2.3..3.0 mm
top-barrier lateral tolerance                0.20 -> 0.30 mm
top-barrier clearance                        0.15 -> 0.25 mm
ordinary jaw contact margin                  0.40 mm（不变）
```

一环拓扑和 `3.5 mm x 1 generation` 范围保留，但约束坐标系改为两爪共同的物理 jaw-hinge
parent/wrist frame：四个直接 `u_t` 仍各自跟随 jaw A/B 并响应 q7；外圈不再进入任何
per-jaw finite-volume source field，只在两侧主锚点投影后应用一次共同腕部 compliant target。
应用前重新对其 incident tet 做 `J>=0.03` line search。这样腕部整体抬升/放下时外圈可
随动，而单独 q7 闭合不会把外圈向两侧拉开。松爪仍清空全部 direct/support 绑定。

CPU 合成门禁明确检查：jaw A/B 的主锚点具有不同 q7 目标，外圈 x 位置不响应 q7；外圈
响应共同腕部抬升；第二拓扑环不传播；打开 q7 后全部释放。`paper_soft` 静态回归、通用
triangle-skin contact 和四点/外圈测试均通过。

`paper_pbd v15` 的 `1795..1830` 宽门跳帧仍从约 `-7 mm` 深交叠起步，最低 support/direct
scale 会接近零且局部最大位移不能代表 Reset 后连续 GUI，因此只用于确认 `4 direct +
8 wrist support` 接线和无翻转，不用它宣称膨胀视觉已通过。正式验收必须 Reset 后连续
播放，观察单爪下压、q7 闭合瞬间是否出现外圈横向鼓起、共同抬升以及松开回弹。

## 深压反向膨胀修复：竖直接触职责拆分（2026-08-09）

论文 `arXiv:2309.11656` 明确使用 distance、volumetric、shape-matching 三类 PBD 约束，
参与节点刚度取平均，并固定 `k_vol=1e10`；正文没有给出 `V/V0-1` 的体积约束公式。
因此保留当前标准 PBD 的 `C_vol=V-V0`，不再把此前自拟的无量纲化称为论文要求。

深压消融确认反向膨胀不是 volume 或 shape projection 产生：`1795..1796` 深重叠段中，
完整配置最大向上位移约 `+3.094 mm`，关闭全部材料迭代后仍为 `+3.309 mm`；只保留
top barrier 时为 `+0.00011 mm`，只保留 ordinary jaw contact 时又恢复到
`+3.094 mm`。根因是夹爪进入重建顶面后，闭合 collision skin 的 nearest-face 查询会
命中底面/陡峭侧面；普通 non-penetration 和切向 no-slip 随其外法向把组织底面向上推。

当前将职责固定为：ordinary jaw contact 只保留世界 `xy` 平面内的 lateral/q7 分量；
pre-grasp 的竖直下压只由 oriented top barrier 负责；capture 后的三维抬升只由每侧两个
`u_t` direct anchors 和共同腕部 support ring 负责。相同两状态回归变为：

```text
maximum upward displacement       +3.094 -> +0.00037 mm
maximum downward displacement     -0.700 -> -0.708 mm
minimum tet volume ratio           0.220 -> 0.586
inverted tetrahedra                0
```

`1795..1830` 的 36 状态深压跳帧回归最大向上位移 `+0.076 mm`，3 mm pressure ROI
最大值 `0`，stereo-left shoulder 没有节点上升超过 `0.1 mm`；最大下压 `-3.713 mm`、
最低体积比 `0.0170`、翻转数 `0`。这段从约 `-7 mm` 初始深交叠开始，仍不用于评价正常
Reset 连续 GUI 的下压幅度，但足以证明底面反向顶起已被移除。通用 triangle-skin、
四点/共同腕部 persistent grip 和 `paper_soft` 完整 CPU 回归均通过。

## 在线刚度安全闭环向新方案对齐（2026-08-11）

本轮先对照 `CURRENT_SUPER_TISSUE_FRAMEWORK.md`、`刚度优化方案.md` 和实际源码审计旧链路。
结论是新方案方向可行，但旧实现有三个不能直接继续加学习率的问题：accepted 位置 residual
会立即原位修改 XPBD 刚度；已有状态快照不包含 persistent grip、运动学插值历史和材料数组；
residual 优化器使用的一阶 Gaussian 平移映射没有包含正式蒙皮中的 offset-frame 旋转，因而
不能直接把近似 loss 当作最终显示状态的验收结果。另外，方案草案曾把现有余弦方向信号
误写成带 `0.20 mm` 投影分母的公式；本轮没有悄悄改变材料信号，而是按真实代码纠正文档，
把信号形式的替换留作独立消融。

### 完整 rollout 状态与可逆影子路径

新增两层快照：`PhysicsRolloutAuxiliaryState` 保存 `sim_time`、运动学插值 start/target、
paper distance/shape 刚度，以及全部 persistent-grip 可变数组和 Python 状态；
`EmbodiedGaussianRolloutState` 再与原有 physics state/control、Gaussian state 合并。
persistent grip 快照覆盖 capture/release 信号、四点选择、direct/support body/local/weight/
level、接触斑计数、支撑传播代数、上一 q7 和 motion state。Reset 和 shadow rollout 现在
使用同一套完整 clone/copy 接口，影子路径结束后不会把临时材料、夹持状态或插值历史留在
真实仿真中。影子运行保留已有 direct/support 绑定，但把 capture/release 输入清零，避免
baseline/candidate 因状态机分叉而比较了不同材料问题。

同时在 triangle-skin contact 指标中加入 direct `u_t` 锚点相对各自 jaw body/local target
的 RMS 与最大误差。`reset_persistent_grip()` 也补齐 q7、时间戳、capture/release、上一角度
和 motion state 清理，避免 Reset 后沿用上次运行的开合历史。

### candidate / verified 双缓冲和材料门控

`ResidualDrivenPaperStiffnessUpdater` 现在把一次更新拆为：

```text
propose -> isolated history/shadow validation -> commit or reject
```

`propose` 只生成独立的 `PaperStiffnessCandidate`，不写 projector。只有 `commit` 才安装
distance/shape 候选和 candidate EMA；`reject` 保持 verified 刚度，旧 EMA 乘 `0.5` 衰减。
EMA 按新方案改为 `0.80 * history + 0.20 * current`。保留实际运行中的余弦方向：

```text
clip(-dot(d,r)/(||d|| ||r||) + 0.15, -1, 1)
* residual_strength * deformation_strength
```

单步 log 上限、distance/shape 范围和 `1.5x` shape gain 不变。新增材料指标包括 candidate/
commit/reject 数、EMA 有效节点、平均绝对 log step，以及 distance/shape 在物理边图上的平均
绝对粗糙度。

节点门控现同时要求：修正前后 incident tet 均 `J>=0.03`、当前 masked RGB 对该节点具有
非零有效梯度、不是 fixed，并且不在 `u_t` direct/support 或它们的物理一环内。`u_t` 区域
既不作源也不作目标，邻域平滑和旧 EMA 都不能把材料信号泄漏回来。视觉梯度只由 data term
反传，因此已经包含左右目 mask、器械遮挡和 renderer visibility，而不是由物理正则伪造
“可见性”。低质量、fixed 和 `u_t` 节点会立即清除 verified EMA，即使随后 candidate 被
拒绝也不会让旧证据在恢复资格后重新出现。capture/release、快速 q7、超过 `1 mm` 穿透或明显接触斑切换会暂停更新并进入
3 次更新冷却；capture/release、穿透和接触切换清空旧 EMA。位置 residual 自身被拒绝时也
清空 EMA。

### 精确重蒙皮验收、历史稳定性和下一观测帧 prediction gap

位置 residual 候选不再只用优化器内部的近似 Gaussian mean 验收。候选节点先临时写入
`particle_q`，调用正式视觉顶点嵌入和 Gaussian skinning 更新 mean/quaternion，再以左右目
真实 mask 渲染。精确总 loss 必须下降，任一有效相机不能超过 `1e-6` 或 `0.5%` 容差明显
退化；失败时恢复原节点并再次正式蒙皮。每目记录 initial/final loss、权重和有效像素/mask
覆盖率。节点级视觉监督梯度另行保留给刚度门控。

准静态历史缓冲最多保存 4 个完整状态，只接收未处在过渡门、q7 较慢且最大节点速度低于
`0.02 m/s` 的 accepted 状态。候选生成后，每个历史状态分别以“今天的 verified 刚度”和
candidate 刚度做 1 个物理帧的冻结工具/冻结持握状态松弛，比较质量加权粒子 RMS。这里已
修复一个审计中发现的对照污染：历史快照虽然保存了当时的材料数组，但 baseline/candidate
两边都会先覆盖为今天的 verified 参数，保证唯一自变量是当前 candidate。

通过历史门后，保存当前 accepted 完整快照，并逐物理步记录相同的 frame/timestamp/PSM
state 命令。真实仿真继续使用 verified；下一张新图像到达后，从同一快照分别运行 verified
和 candidate 两条开放环 shadow rollout，关闭 visual residual/force/online update 和新抓取
判定，精确重蒙皮后比较 prediction gap。提交同时要求：

- candidate 双目平均 loss 至少改善 `max(1e-6, 0.1% * baseline)`；
- 每目 loss 不超过 `max(1e-6, 0.5% * baseline_camera)` 的退化容差；
- 不新增翻转，最小 `J` 不低于 baseline 的绝对 `0.01`/相对 `2%` 双重门；
- 最大穿透不比 baseline 多 `0.1 mm`；
- direct anchor RMS/最大误差不比 baseline 多 `0.05 mm`；
- 历史质量加权 RMS 不超过 baseline 的 `10% + 0.01 mm`。

验证还记录两条路径的 distance/volume/shape 诊断、最小 `J`、翻转、穿透、锚点误差、左右目
loss、影子物理步数和图像 horizon。等待路径最多 120 个物理步或 10 个视频帧，超出拒绝。
同一图像 frame 只允许求解一次 residual，避免视频线程较慢时反复用同一证据生成候选；
Pause 或视频到末帧会取消未验证候选，避免没有新图像时命令列表无限增长。

### 验证与文档同步

新增/扩展门禁结果：

```text
python -m py_compile <全部本轮修改模块>                         PASS
PYTHONPATH=src:examples python scripts/test_super_online_tissue_stiffness.py
  candidate 不改 verified / u_t 屏蔽 / reject 回滚 / EMA invalidation  PASS
PYTHONPATH=src:examples python scripts/test_super_visual_tissue_residual_mapping.py
  loss 下降 / 正体积 / 逐相机像素指标 / node supervision gradient     PASS
PYTHONPATH=src:examples python scripts/test_super_tissue_persistent_grip.py
  双侧四点、support、各 jaw target、释放清理                         PASS
PYTHONPATH=src:examples python scripts/test_super_paper_soft_mode.py --device cpu
  完整 rollout 快照恢复、20 步静态、材料/蒙皮/接触全部 gate           PASS
```

完整 `paper_soft` CPU 场景加载 `4,059` 粒子、`15,830` tet、`26,754` 组织 Gaussian 和
`76,798` 器械 Gaussian；新增 `shadow_rollout_snapshot_restores_all_state=true`，20 步静态
最小体积比 `0.99999857`、最大动态位移和 fixed drift 均为 `0`。当前容器仍无 CUDA，Warp
启动时报告 CUDA device unavailable；因此这只证明状态/机制和 CPU 回归通过，不代表完整
`paper_pbd v15 + 双目渲染 + shadow rollout` 已达到实时或改善真实整段轨迹。

`CURRENT_SUPER_TISSUE_FRAMEWORK.md` 已更新为 candidate/verified 和下一图像验收的实际时序；
`刚度优化方案.md` 新增实现状态表并纠正方向信号公式。尚未完成的是独立的
`H=1/3/5/10` 批量开放环评测、材料隔离/端到端两种协议、动作阶段分组、全轨迹指标持久化
以及完整 CUDA 压下—抓取—抬起—放下—释放 A/B。当前安全结构已经向方案主体靠拢，但
阈值仍是 preliminary，必须先冻结轨迹校准，再讨论材料参数收敛或物理真实性结论。

## `paper_pbd` 明显软档：缩短夹取位移的长程传播（2026-08-11）

用户在 GUI 中观察到夹取影响传播到器械后方 `10 mm` 以上；这明显大于预期主要集中在
接触附近 `3--5 mm` 的范围。体积守恒确实会把局部压入转成邻域侧向位移，但它不能单独
解释如此长的传播范围。当前更直接的可控因素是 distance/shape 恢复刚度以及每子步重复
投影次数：它们越强、重复次数越多，局部位移就越容易沿四面体边图逐层传到远处。

按“大幅度变软”的要求，`paper_pbd` 默认值从此前观察档切为：

```text
distance stiffness             0.35 -> 0.20   (-42.9%)
shape stiffness               0.006 -> 0.004  (-33.3%)
material iterations/substep       8 -> 6      (-25.0%)
volume stiffness              1e10 -> 1e10    (保持)
material relaxation            1.0 -> 1.0     (保持)
```

这里没有一起降低 `k_volume`。原因是当前组织需要保留近似不可压的体积响应；放松 volume
会让模型通过局部塌缩显得更软，同时增加低质量或翻转单元风险，并不等价于真实软组织变软。
新 distance/shape 恰好是在线更新器已经验证的安全下限，因此闭环收到进一步软化信号时会
饱和，而不是越过边界；收到可靠硬化信号时仍可逐节点向上调整。

同步修改了主场景默认配置、离线入口注释、独立材料门禁默认值和固定配置 gate，并更新
`CURRENT_SUPER_TISSUE_FRAMEWORK.md`、`刚度优化方案.md`。独立 CPU 材料回归结果：

```text
settings                              0.20 / 1e10 / 0.004, 6 passes
rest minimum J                        0.9999933
rest maximum displacement             2.77e-10 m
0.75 mm perturbation recovery min J   0.78577
recovery maximum displacement         0.1261 mm
inverted tetrahedra                    0
edge RMS                 1.5836e-5 -> 5.3527e-6 m
volume RMS               5.1076e-11 -> 1.9711e-11 m^3
all independent material gates        PASS
online lower-bound/hardening gates     PASS
```

额外跑过旧版 `smoke_super_triangle_skin_contact.py` 的 CPU 广义场景。其有限性、无翻转和
材料加载均正常，但总报告不是 PASS：该脚本仍要求“禁用 persistent grip”、旧的深交叠
捕获条件和 CPU `<60 ms`，与当前持久夹持主路径及运行环境不一致，因此不能把它作为当前
GUI 轨迹验收。此次回归只证明新软档没有在隔离材料测试和该接触样本中引入数值爆炸；是否
已把后方影响收回 `3--5 mm`，必须在同一 Reset 后连续 GUI 夹取轨迹中和软化前画面做 A/B。

## 稍深进入与稍宽局部夹取圈（2026-08-11）

按用户要求只做小幅接触/夹取调整，不改变刚完成的材料软档：

```text
tip entry allowance                         2.3 -> 2.5 mm
top-barrier active distal band         2.3..3.0 -> 2.5..3.0 mm
maximum capture penetration                  0.8 -> 1.0 mm
  (including 0.4 mm contact margin; actual overlap 0.4 -> 0.6 mm)
soft lift support edge radius                 3.5 -> 4.0 mm
support propagation generations                 1 -> 1
direct jaw-frame anchors                         4 -> 4
finite-volume transfer layers                    1 -> 1
```

进入侧同时缩短 oriented top-barrier 的轴向有效段，并把正式捕获门增加 `0.2 mm`，避免
夹爪已经按新许可稍深进入后又被旧 `0.8 mm` 门拒绝。夹取侧没有增加满权固定点，仍是每爪
两个 direct `u_t`；只是允许一代 collision-skin 表面邻边的最大长度增加 `0.5 mm`，新增
节点仍采用 Gaussian 衰减权重、共同 wrist frame、compliant correction 和 `J>=0.03`
line search。松爪继续清空 direct/support 全部绑定，因此不是永久固定更大组织区域。

验证结果：

```text
scripts/test_super_tissue_persistent_grip.py                    PASS
  4 direct / one support generation / second ring excluded
  per-jaw targets distinct / support ignores q7 / release clears all
scripts/test_super_paper_soft_mode.py --device cpu              PASS
  loaded capture=1.0 mm, support=4.0 mm x1, barrier=2.5..3.0 mm
  20-step minimum J=0.99999857, maximum drift=0, no inversion
python -m py_compile + git diff --check                         PASS
```

v15 collision skin 共有 `3,174` 条无向表面边；`3.5 mm` 半径允许其中 `1,580` 条，
`4.0 mm` 允许 `2,046` 条，即新增 `466` 条候选邻边。它们只是全局可用邻接表，实际夹取仍
只会从四个 direct anchor 沿一条真实表面边扩展，不能越过薄组织抓到底面，也不会继续传到
第二环。当前容器无 CUDA，因此最终夹住粒子数、视觉夹起宽度和稍深进入效果仍以 Reset 后
连续 GUI 为准。

## 在线刚度范围与响应速度扩展（2026-08-11）

用户确认按前述建议扩大在线参数实时修正能力。本轮保持 GUI Reset 材料
`0.20/1e10/0.004 + 6 passes` 不变，只修改 residual 驱动的候选范围和时间滤波：

```text
distance online range              0.20..1.60 -> 0.15..1.60
shape online range               0.004..0.020 -> 0.003..0.020
EMA current/history                  0.20/0.80 -> 0.30/0.70
log learning rate                         0.18 -> 0.18
distance/shape maximum log step        0.18/0.27 -> 0.18/0.27
volume stiffness                            1e10 -> 1e10
```

因此 Reset 初值的向下空间由 `0%` 变为 distance/shape 均 `25%`。EMA 从零开始收到相同满幅
证据时，第一轮合成候选由“软化被旧下限截断”为：

```text
                                     before             current
first accepted full soft evidence
  distance                       0.20000 (0%)       0.19103 (-4.49%)
  shape                          0.00400 (0%)       0.0037339 (-6.65%)
first accepted full hard evidence
  distance                       0.20733 (+3.67%)   0.21110 (+5.55%)
  shape                          0.0042219 (+5.55%) 0.0043375 (+8.44%)
EMA reaches >=50% of steady signal       4 updates          2 updates
EMA reaches >=80% of steady signal       8 updates          5 updates
```

生产默认值的确定性合成门禁中，连续满幅软化且每次直接提交的逐步结果为：

```text
commit                 0          1          2          3          4
distance          0.20000    0.19103    0.17669    0.15979    0.15000
shape             0.00400    0.0037339  0.0033214  0.0030000  0.0030000
```

第 4/3 次后 distance/shape 分别在新下限饱和；从下限施加硬化证据仍能上升。该数据说明的是
“更新器允许多大、响应多快”，不等于真实视频会连续提交满幅软化：下一图像 prediction gap、
历史稳定门、tet 质量、视觉监督、`u_t` 一环排除和 3 次冷却仍全部生效。

同时扩展独立材料门禁，使其可分别锁定 Reset 基线和在线下限。两档 CPU 结果均通过全部
finite、无翻转、固定支撑零漂移和 edge/volume energy 下降门：

```text
                                      reset baseline       online floor
distance / volume / shape          0.20 / 1e10 / .004   0.15 / 1e10 / .003
rest minimum J                         0.9999933            0.9999933
rest maximum displacement              2.77e-10 m           2.56e-10 m
recovery minimum J                      0.78577               0.82600
recovery maximum displacement           0.12608 mm            0.07050 mm
recovery edge RMS                       5.3527e-6 m           4.9399e-6 m
recovery volume RMS                     1.9711e-11 m^3        1.9011e-11 m^3
inverted tetrahedra                     0                     0
```

最后四项只是同一 `0.75 mm` 单点扰动在第 80 子步的瞬时状态；受振荡相位和阻尼影响，不能
据此反推更软档“恢复更强”或材料更真实。它们能支持的结论仅是：新在线下限在当前 timestep、
6 次材料迭代和固定 `k_volume` 下没有引入数值失稳。GUI 刚度面板与启动日志现同步显示
`0.15..1.60 / 0.003..0.020` 和 EMA `0.30/0.70`。

## distance 区域差异范围扩展到 20 倍（2026-08-11）

用户指出不同组织区域的 distance 刚度差异较大，`0.15..1.60` 仍可能使软区和硬区过早
贴住边界。本轮只扩大 distance 的长期有界范围，不增加单次更新幅度：

```text
distance online range             0.15..1.60 -> 0.10..2.00
distance range ratio                    10.67x -> 20.00x
distance soft headroom from Reset          25% -> 50%
shape online range                  0.003..0.020 (保持)
EMA new/history                         0.30/0.70 (保持)
maximum distance log step                    0.18 (保持)
volume stiffness                               1e10 (保持)
```

shape 上限没有同步放大：`0.020` 是 Liang et al. 的 shape 搜索上界，而且过高 shape 会在
抓取区与 direct/support `u_t` 对抗。distance 的每次候选仍受 EMA、`±0.18` log step、历史
松弛、下一图像 prediction gap、tet 质量、视觉监督和 `u_t` 一环排除约束；因此 20 倍是
不同区域在长期、反复验证后允许形成的最大跨度，不是一次图像更新会发生 20 倍跳变。

生产默认合成门禁结果：

```text
full soft commits       0        1        2        3        4        5        6        7
distance             .2000    .1910    .1767    .1598    .1423    .1253    .1094    .1000
shape                .0040   .003734  .003321  .003000  .003000  .003000  .003000  .003000

full hard commit        0        1        8        9       14       15       16
distance             .2000    .2111    .5682    .6754   1.6379   1.9592   2.0000
shape                .0040   .004337  .019156  .020000  .020000  .020000  .020000
```

distance/shape 分别在第 `7/3` 次满幅软化提交到下限，在第 `16/9` 次满幅硬化提交到上限；
两端饱和和反向离开门禁通过，20 倍 distance 区域跨度门禁通过。随后分别把全网材料固定在
两端运行独立 CPU 材料回归，所有 14 项 gate 均通过：

额外区域探针保留生产默认的一次物理一环平滑，让三节点相邻链两端持续接收相反的软化/
硬化证据。40 次合成提交后的 distance 为 `0.10000/0.23896/2.00000`，最大/最小恰好
`20.0x`；软端第 10 次到 `0.10`，硬端第 23 次到 `2.00`。这说明允许的异质范围能在默认
空间平滑下实际形成，而不只是上下界常数之比。

```text
                                      online floor       online ceiling
distance / volume / shape          0.10 / 1e10 / .003   2.00 / 1e10 / .020
rest minimum J                         0.9999933            0.9999933
rest maximum displacement              2.28e-10 m           4.05e-10 m
recovery minimum J                      0.87864               0.94422
recovery maximum displacement           0.10457 mm            0.01739 mm
recovery edge RMS                       4.4160e-6 m           1.3208e-6 m
recovery volume RMS                     1.5473e-11 m^3        4.5961e-12 m^3
inverted tetrahedra                     0                     0
```

这些是 `0.75 mm` 单点扰动第 80 子步的数值健康度快照，不等价于真实视频已经辨识出 20 倍
材料差异。最终区域分布应在 GUI 中看 distance `min/median/max` 和 edge roughness，并确认
极值不是只由少数边界节点或遮挡误差造成。启动日志和 GUI 范围显示已同步为
`0.10..2.00 / 0.003..0.020`。

## 在线刚度小白文档、GUI 热调与实时指标精简（2026-08-12）

新增 `在线刚度参数小白说明与GUI调参指南.md`，从 XPBD constraint、物理预测形变
`d=x_pred-x_rest`、accepted 图像修正 `r=x_acc-x_pred`、余弦软硬方向、幅值门、物理边图
平滑、EMA 和 log-space candidate 公式逐层解释当前参数。文档将“上下限决定长期范围”和
“learning rate/EMA/log cap 决定每次速度”分开，列出所有旋钮变大/变小的现象、过调风险、
安全门不更新的原因、接触参数和推荐 A/B 顺序。

### 运行时调参

`Online stiffness tuning (pause to apply)` 现在暴露 16 个公式参数：

```text
distance/shape lower and upper bounds
log learning rate / maximum log step
EMA new-evidence weight / rejected EMA keep ratio
hardening bias / shape update gain
minimum and full-scale residual
minimum and full-scale deformation
spatial smoothing passes / neighbor blend
```

调节采用 draft -> Apply，而不是拖动滑条立即改物理。Apply 只允许在 Pause 且没有 pending
candidate 时执行；保留已经 verified 的逐节点材料场，只把超出新边界的值裁回范围，然后
清空旧 EMA、准静态历史、上一轮 temporal residual 和 validation 指标，并进入 3 次更新冷却。
粒子位置、Reset 初值和固定 `k_volume=1e10` 不变。`Restore startup values` 恢复启动草稿；
Reset 把材料场恢复为 `0.20/0.004`，但不撤销当前进程已经 Apply 的调参规则。

`ResidualDrivenPaperStiffnessUpdater.reconfigure()` 提供与 GUI 无关的安全重配置入口：拒绝
pending candidate、校验阈值/范围、裁剪 verified 并清空 EMA。`validate()` 同时新增
`minimum residual/deformation <= full scale` 约束，避免生成逻辑相反的证据区间。

### GUI 精简

默认实时窗口不再平铺几十行指标：

- PSM 手动对齐默认折叠；
- 材料默认只显示 Reset 参数和最大位移；
- 刚度默认显示 status、commit/reject、distance/shape min/median/max 和证据方向；
- 接触默认显示 grasp verdict、attached 数、左右接触、gap 和 penetration，只有 unsafe tet
  非零时额外显示安全警告；
- residual 默认显示 accepted、loss reduction 和最大修正。

原有逐目 loss/mask、tet/视觉/`u_t` 屏蔽、EMA、roughness、prediction horizon、contact
safe scale 等没有删除，分别移入 `Stiffness diagnostics`、`Contact diagnostics`、
`Visual residual diagnostics` 和 `Material and contact setup details` 折叠区。

验证：

```text
python -m py_compile <online updater / GUI / tests>                    PASS
PYTHONPATH=src:examples python scripts/test_super_online_tissue_stiffness.py
  原有软硬/隔离/回滚/20x区域差异 + runtime reconfigure             PASS
PYTHONPATH=src:examples python scripts/test_super_stiffness_gui_controls.py
  16个公式控件存在 / paused Apply / clip+clear / playing时拒绝      PASS
python examples/example_embodied_super_offline.py --help              PASS (headless)
git diff --check                                                       PASS
```

当前容器仍无 display/CUDA，因此没有在这里实际打开窗口点击；GUI 绑定采用项目已有
`collapsing_header/slider/button` 接口，并通过 headless fake-imgui 路径覆盖控件构造与 Apply
语义。用户下次有显示环境时可直接打开 GUI 检查排版和滑条手感。

## 接触与持续抓取参数接入 GUI（2026-08-12）

用户要求把接触和抓取参数也放入 GUI。本轮新增不可变的 `ContactGripGuiSettings`，启动时从
`PhysicsSettings` 和当前 `TriangleSkinContactProjector` 完整抓取草稿。用户先要求只保留
tip/capture，随后补充 grip support radius，因此最终面板只有三个可见参数：

```text
Tip entry and grip (pause to apply)（本节记录初版，最终宽范围见后文）
  Tip entry allowance (mm)       2.50 mm startup
  Capture max penetration (mm)   1.00 mm startup
  Grip support radius (mm)       4.00 mm startup
```

其余 contact margin/correction/relaxation、friction、sampling/query/CCD/spread、solver passes、
minimum `J`、support generations/compliance/transfer、接触片/角度状态机等取消 GUI 控件，继续使用启动
配置。synthetic pressure shoulder/top-support weight 继续关闭；四个 direct surface particles
仍是算法固定条件。三个可见长度在 GUI 用 mm 显示，内部仍以 m 存储。

最终宽范围实现会随 tip 自动扩展隐藏的 top-barrier distal length，但不超过 jaw contact distal
length，细节见本文件后面的“跳点稳定调参与宽范围 GUI”。

### 安全 Apply 语义

接触/抓取参数与刚度一样只改草稿，必须 Pause 且没有 pending stiffness candidate 才能 Apply。
Apply 先校验有限值、`query>=margin`、`tip<top distal<=jaw distal`、
`closed<release<=wide-open`、正体积阈值、层数和柔顺度。随后重建整个接触 projector，而不是
直接篡改部分字段，因为 tool sample mask、tip/barrier mask、grip transfer tet 和 support
neighborhood 都是构造时预计算的。

重建使用原工具 shape、jaw IDs 和未暴露的 pressure/top-support 配置；组织粒子位置和 verified
distance/shape 参数不变。新 projector 天然清空旧 persistent-grip anchor，之后同时清空旧
接触语义下的 stiffness EMA、历史、validation 和 temporal residual，并加入 3 次更新冷却。
如果正在持握，Apply 后需要重新闭合并满足双侧接触条件才会再次 capture。

`Simulator.configure_triangle_skin_contacts()` 补齐以下原本只存在于 projector 构造函数的释放
接口，确保热重建不会把它们静默恢复成另一套值：

```text
persistent_grip_release_angle_min_rad
persistent_grip_release_angle_delta_rad
persistent_grip_wide_open_angle_rad
persistent_grip_angle_motion_epsilon_rad
```

`在线刚度参数小白说明与GUI调参指南.md` 的标题和内容范围已扩展到组织刚度、接触与抓取，加入
signed-distance contact correction、摩擦上限、双侧 capture 条件、compliant XPBD grip 和
Gaussian-decayed support 公式。文档明确 `10 mm query distance` 只是宽相搜索半径，不是
`10 mm` 位移传播半径。

验证：

```text
python -m py_compile examples/example_embodied_super_offline.py \
  src/embodied_gaussians/physics_simulator/simulator.py \
  scripts/test_super_stiffness_gui_controls.py                         PASS

PYTHONPATH=src:examples python scripts/test_super_stiffness_gui_controls.py
  刚度16项保持；接触/抓取只有tip/capture/support radius            PASS
  margin/friction/query/spread/angle等旧控件均不存在                PASS
  paused Apply重建projector、写PhysicsSettings并清旧anchor/evidence PASS
  stiffness和contact/grip运行中Apply均拒绝                          PASS

PYTHONPATH=src:examples python scripts/test_super_paper_soft_mode.py
  完整CPU场景、7579工具采样、rollout/persistent-grip/释放门         PASS
  reference equilibrium 20步: min J=0.99999857, max displacement=0   PASS

PYTHONPATH=src:examples python scripts/test_super_tissue_persistent_grip.py
  双侧捕获 / 四锚点+一层support / 独立jaw安全缩放 / opening清空   PASS

git diff --check <本轮代码与文档>                                    PASS
```

当前容器没有 CUDA/display，完整场景回归自动回退 CPU；未在此环境实际点击渲染窗口。fake-imgui
已覆盖所有控件调用和 Apply 状态机，下一次有 X11/ThinLinc 显示时只需检查视觉排版和滑条手感。

## Shape lower bound GUI 范围提高（2026-08-12）

按用户要求，只扩大 `Shape lower bound` 的 GUI 调节范围：

```text
GUI slider range       0.001..0.004 -> 0.001..0.010
startup lower bound    0.003（保持）
startup upper bound    0.020（保持）
verified shape field   未自动改变；Pause + Apply 后按新下限裁剪
```

这次没有修改在线更新公式、学习率、EMA、shape upper bound 或 Reset 基线。把 lower bound 调高
会抬高“最软 shape 区域”的下限，使局部更不容易剪切/弯折；若草稿 lower 高于 upper，原有
`OnlineTissueStiffnessSettings.validate()` 会拒绝 Apply。fake-imgui 门禁新增滑条范围精确检查，
确认范围为 `0.001..0.010`；语法、GUI Apply 与 `git diff --check` 均通过。

## 跳点稳定调参与宽范围 GUI（2026-08-13）

用户观察到少数物理粒子跳动并显著偏离整体，同时指出现有滑条范围不足以形成肉眼明显的
A/B。代码检查确认 persistent grip 的 direct/support 位置修正会在每个物理子步通过
`contact_projection_velocity_scale` 写回速度；如果四个 direct anchor 中少数节点先被拉走，
固定 `0.10 mm/substep` correction cap 与低柔顺度会形成局部速度尖峰。单独增加 shape 下限或
support radius 不能完整解决这个问题：前者可能把冲击传得更远，后者只让更多邻居一起运动。

### 新增跳点稳定折叠区

`Tip entry and grip -> Particle jump stabilization` 新增：

```text
Grip correction cap         0.005..2.000 mm/substep（启动 0.100）
Grip compliance             0.00..2.00 m/N          （启动 0.05）
Material/grip vel transfer  0.00..1.00               （启动 0.12）
Particle velocity damping   0..100 /s                （启动 10）
```

推荐先后次序和首轮明显档为：

```text
1. correction cap   0.10 -> 0.03..0.05 mm/substep
2. compliance       0.05 -> 0.15..0.30 m/N
3. velocity transfer 0.12 -> 0.03..0.06
4. damping          10   -> 25..40 /s
5. 若仍是局部尖刺，support radius 4 -> 6..8 mm
```

correction cap 越小，单个子步越不可能把 anchor 猛拉出去；compliance 越大，锚点越柔顺；
velocity transfer 越小，位置投影越少变成下一步速度；damping 越大，全局速度振荡衰减越快。
过度设置分别会造成夹爪跟随滞后、抓取打滑和组织像黏性
材料。capture max penetration 不属于止跳参数，增大它反而会更容易锁住深穿透坏状态。

`ContactGripGuiSettings` 新增 `particle_velocity_damping_per_second` 和
`material_projection_velocity_scale` 的启动抓取、校验、Apply 写回和 Restore；持续抓取在
contact displacement 记账之后投影，所以其位置修正实际走 material velocity scale 写回速度。
grip correction/compliance 原本已在 schema 和 projector 重建路径中，本轮才暴露到折叠 GUI。

### 肉眼明显的宽实验范围

```text
Distance lower bound     0.05..0.20   -> 0.01..1.00
Distance upper bound     0.20..4.00   -> 0.10..10.00
Shape lower bound        0.001..0.010 -> 0.0001..0.030
Shape upper bound        0.004..0.040 -> 0.001..0.100
Capture max penetration  0.20..4 mm   -> 0.10..10 mm
Grip support radius      0..10 mm     -> 0..30 mm
```

启动参数全部保持不变。tip entry 不能脱离 top-barrier 几何盲目放大，仍动态限制为有效 distal
length 减 `0.01 mm`；当前 jaw contact distal 为 `5.0 mm`，所以最大 `4.99 mm`。滑条增大 tip
时会同步扩展隐藏的 top-barrier distal length，仍保持 `tip < barrier <= jaw distal`。shape
`>0.020`、support radius `>10 mm`、capture
penetration 数毫米以上都属于故意保留的压力测试档，不代表推荐安全默认。

验证：

```text
python -m py_compile <GUI / fake-imgui gate>                         PASS
PYTHONPATH=src:examples python scripts/test_super_stiffness_gui_controls.py
  四个材料bound宽范围精确匹配                                      PASS
  entry/capture/support + correction/compliance/transfer/damping七项  PASS
  paused Apply写入projector+damping并清旧anchor/evidence               PASS
  playing Apply拒绝 / 非法几何与角度拒绝                              PASS
git diff --check                                                       PASS
```

补充默认回归：完整 CPU `paper_soft` 仍加载 `damping=10/s`、`grip compliance=0.05 m/N`、
`grip correction=0.10 mm/substep`、`support=4 mm x 1 generation`；20 步参考平衡态
`min J=0.99999857`、最大动态位移 `0`，全部场景 gate 通过。持续抓取独立回归的第三次双侧接触
capture、四 direct anchor、单层 support、共同腕部跟随、独立 jaw 安全缩放和 opening 全清空也
全部通过。在线刚度原有 20x 区域差异与 runtime reconfigure gate 通过；宽 GUI 没有修改启动默认
或在线更新公式。

## 刚度优化方案完成度审计与小白运行文档（2026-08-13）

本轮逐项对照 `刚度优化方案.md`、`CURRENT_SUPER_TISSUE_FRAMEWORK.md`、在线更新器、GUI 主循环、
视觉 residual 验收和现有测试。结论不是“全部完成”，而是分成两层：

```text
在线安全闭环主体                         已实现
  x_pred / x_acc / residual 分离
  u_t direct/support/一环排除
  质量、视觉监督和接触过渡门
  空间平滑 + EMA + log-space bounded candidate
  candidate / verified 双缓冲
  完整 rollout 状态快照/恢复
  最多4个历史快照的1-frame初版稳定门
  下一可用新观测 verified/candidate 双影子验收
  image/J/inversion/penetration/anchor/history 联合提交门

最终科学与工程验收                       未完成
  H=1/3/5/10 独立批量开放环
  完整轨迹指标持久化与动作阶段分组
  frozen CUDA fixed-k/residual-only/residual+stiffness A/B
  参数收敛、跨次重复性和真实材料解释
  完整闭环端到端墙钟性能
```

新增 `当前刚度优化方法与运行流程_小白公式版.md`。文档从 distance/volume/shape XPBD 公式开始，
依次解释左右目 Smooth-L1、节点 residual 总损失、正式 Gaussian 重蒙皮验收、`d/r` 两支箭头的
软硬判断、证据幅值门、物理图平滑、EMA、log-space candidate、历史质量加权 RMS、下一图像
prediction gap 和 commit/reject 联合门。运行部分明确说明：

1. 只有 `paper_pbd + residual + online update` 才创建更新器；
2. 视频请求约 30 FPS，物理配置 60 Hz/12 substeps，视觉每 3 物理帧请求但只消费新图像；
3. accepted residual 立即修改位置但不改速度；
4. candidate 不进入 live XPBD，pending 时记录真实器械命令并暂停新的 residual；
5. 下一新图像从同一快照跑 verified/candidate 两条影子，恢复 live 后再提交或拒绝；
6. GUI 启动 bounds 与宽压力测试滑条严格分开描述。

文档同时列出实际阈值：`J>=0.03` 材料证据门、`0.02/0.10 mm` residual/deformation 门、
`0.30/0.70` EMA、`±0.18` distance log step、历史 `+10%+0.01 mm`、prediction improvement
`max(1e-6,0.1%)`、逐相机 `0.5%` 容差、最小 J `-0.01/-2%` 容差、penetration `0.10 mm` 和
anchor `0.05 mm` 容差。这样可直接用文档核对 GUI 和 runtime，而不是把启发式参数误当真实
杨氏模量。

## 恢复版方案复核与小白公式文档重写（2026-08-14）

用户将 `刚度优化方案.md` 恢复为 402 行版本。本轮重新从正文提取验收项，没有沿用之前加入
方案页的实现状态表。恢复版仍明确包含第 5 节 `H=1/3/5/10` 多帧开放环和第 6 节四类指标及
压下/闭合/捕获/抬起/放下/松开阶段分组，所以准确边界为：

```text
在线运行方法（第1--4、7节）             已完成
  u_t direct/support/一环双向排除
  q_pred/q_acc/residual 分离
  局部质量、视觉监督和接触过渡门
  物理图平滑、EMA、log-space candidate
  candidate/verified 隔离
  最多4个历史状态的1-frame初版稳定门
  下一可用新图像的双影子联合验收
  commit/reject/完整状态恢复

整份方案离线交付（第5、6节）             尚未完成
  H=1/3/5/10 独立批量 evaluator
  全轨迹指标持久化
  六个动作阶段的自动分组报告
```

恢复版中的四处旧值与当前运行实现不一致，但不是功能倒退：distance bounds 从旧
`0.20..1.60` 扩到 `0.10..2.00`，shape 从 `0.004..0.020` 扩到 `0.003..0.020`，EMA 从
新证据/历史 `0.20/0.80` 改为 `0.30/0.70`，方向项从旧投影长度表达改为归一化
`-cos(d,r)+bias` 再独立乘 residual/deformation 强度。上述改动来自恢复文档之后的用户调参
要求，代码未回滚。

完整重写 `当前刚度优化方法与运行流程_小白公式版.md`，从向量长度、点积和单位换算开始，
新增以下逐步数字算例：

- 5.0 mm 边拉到 5.5 mm 时 distance constraint 的含义；
- `J=1/0.8/0.03/0/<0` 对应正常、压缩、危险、压扁和翻转；
- shape matching 的 `F=Ds Dm^-1`、SVD 和最近旋转；
- Smooth-L1 分段函数、左右目独立归一化和等权平均；
- 七项 residual 总损失及每一项防止的“图像作弊”；
- `d/r` 同向、反向和小 residual 三个软硬判断算例；
- `s=0.60` 与邻居均值 `0.20` 经 0.35 blend 得 `0.46`；
- 连续三次证据下 `0.30/0.70` EMA 的递推；
- `e=0.46` 如何得到 `ell=0.0828`，再得到 distance `0.20->0.2173` 和 shape
  `0.004->0.004529`；
- 三节点质量加权历史 RMS 和 `+10%+0.01 mm` 通过阈值；
- baseline visual loss `0.040` 时下一图像至少改善 `0.00004` 的计算；
- 最小 J、逐相机、穿透和 anchor 容差的数值解释。

运行章节改成完整状态机，明确普通物理帧、视觉更新、accepted residual、candidate、history
gate、pending 命令记录、下一图像两条影子、live 恢复和 commit/reject 的先后关系；GUI 章节
把 bounds、学习率、最大 log step、证据尺度和平滑范围分别解释，避免再把“允许最终走多远”
与“每次走多快”混为一谈。

验证：

```text
Markdown code fences / display math pairs                 62 / 127，全部闭合
git diff --check                                           PASS
python -m py_compile online_tissue_stiffness.py + GUI      PASS
PYTHONPATH=src:examples python scripts/test_super_online_tissue_stiffness.py
  candidate isolation / u_t exclusion / reject rollback   PASS
  current baseline bidirectional update                    PASS
  distance 0.10..2.00 = 20x regional contrast             PASS
  runtime reconfigure                                      PASS
PYTHONPATH=src:examples python scripts/test_super_stiffness_gui_controls.py
  all formula settings exposed / wide GUI bounds           PASS
  paused Apply / contact-grip controls / invalid gate       PASS
```

## 刚度方案第 5、6 节：多帧开放环与阶段指标（2026-08-14）

本轮继续完成恢复版 `刚度优化方案.md` 中尚缺的功能代码。

### 多 horizon 评测

`SuperPlaybackControls` 新增 `CommittedStiffnessEvaluation`。每个 candidate 只有先通过原有下一
新图像联合门并 commit，才创建多帧评测任务；任务持有提交前完整 rollout 快照、candidate、
起始视频帧、动作阶段和从起点以后真实执行的 `StiffnessToolCommand`。支持多个 committed
任务重叠等待各自 horizon。

默认 horizon：

```text
H = 1, 3, 5, 10 video frames
```

每个 H 在精确的 `start_frame+H` 读取真实双目图像；即使普通视觉更新没有恰好消费该帧，也会
临时切换 target observation，评测后恢复当前 observation、live 仿真、控制器索引和 Gaussian。
每个 H 执行四条影子：

```text
material_isolation old       frozen persistent-grip state machine
material_isolation candidate frozen persistent-grip state machine
end_to_end old               live contact/capture/release evolution
end_to_end candidate         live contact/capture/release evolution
```

材料隔离协议从快照恢复已有 direct/support 绑定，随后冻结新的 grip 状态判定，但器械 body
仍按真实记录命令移动，因此 `u_t` 目标随同一夹爪轨迹更新。端到端协议不冻结状态机，用于查看
材料变化是否导致捕获/释放分支改变。

每条影子记录双目 Gap、逐相机 Gap、distance/volume/shape loss、min J、翻转、穿透、anchor
RMS/max、grip active 和相对起点粒子 RMS。另对 old/new 最终状态各运行一次不写回的 visual
residual solve，记录 open-loop residual RMS/max；它衡量纯物理预测后还需要多大节点修正才能
重新对齐图像，不改变 live 或影子状态。

### 持久化和动作阶段

新增
`src/embodied_gaussians/physics_simulator/stiffness_evaluation.py`：

- `StiffnessActionPhaseClassifier` 结合世界 Z 位移、q7 motion、contact count、activation 和
  grip transition 分类 `press/close/capture/lift/place/release/idle`；
- capture/release 保持 3 个物理采样，防止短转换落不到视觉记录；
- `StiffnessMetricsRecorder` 使用 append-only `events.jsonl` 保存每个 reset、visual update、
  next-frame validation 和 open-loop horizon；
- `summary.json` 按 phase 和 image/physical/material/prediction 四类字段持续累计
  数值 count/mean/min/max，并对 accepted/status/protocol/rejection reason 等分类值计数；
- `metadata.json` 保存 dataset、horizon、protocol 和 schema；
- summary 通过同目录临时文件原子替换；已有结果文件时拒绝启动，不覆盖旧实验。

视觉事件现在持久化左右目 before/after、双目平均、accepted/rejected、residual RMS/max、有效
像素、mask coverage 和 solve time；物理事件记录 constraint loss、min J、翻转、局部冻结、
backtrack、接触穿透和 anchor；材料事件记录 candidate/commit/reject、刚度分布、active/
harden/soften、EMA、roughness、history 和全部在线配置；prediction 事件记录两协议的 old/new
Gap、residual RMS 和粒子 RMS。

评测为显式 opt-in，普通 GUI 默认零额外多 horizon 开销：

```bash
bash scripts/run_super_stiffness_evaluation.sh
```

或：

```text
--stiffness-evaluation-output <new-directory>
--stiffness-evaluation-horizons 1,3,5,10
```

GUI 只增加一行 `Open-loop eval: ON | active N | events M`，详细数据不重新塞回实时面板。
Pause 会把所有未完成任务写成 `open_loop_incomplete`；数据集最后一帧先补执行末条器械命令、
完成到期 horizon，再把确实越界的任务标记 unavailable/incomplete。

### 验证

```text
python -m py_compile evaluation module / runtime / test             PASS
bash -n scripts/run_super_stiffness_evaluation.sh                   PASS
git diff --check                                                    PASS
PYTHONPATH=src:examples python scripts/test_super_stiffness_evaluation.py
  horizon parse / invalid reject                                    PASS
  six required phases + idle                                        PASS
  JSONL + metadata + phase aggregate                                PASS
  exact target frame / command truncation                           PASS
  old/new x material-isolation/end-to-end                           PASS
PYTHONPATH=src:examples python scripts/test_super_online_tissue_stiffness.py
                                                                    PASS
PYTHONPATH=src:examples python scripts/test_super_stiffness_gui_controls.py
                                                                    PASS
PYTHONPATH=src:examples python scripts/test_super_visual_tissue_residual_mapping.py
                                                                    PASS
PYTHONPATH=src:examples python scripts/test_super_tissue_persistent_grip.py
                                                                    PASS
```

当前执行容器没有 CUDA device，无法在本轮实际生成完整双目真实轨迹报告。现在的边界已从“功能
缺失”变为“评测入口、schema 和状态隔离均已实现并通过 headless 回归，等待用户在有 CUDA/
Display 的运行环境执行完整轨迹并解读数据”。

## 抓取区 visual residual 小范围动态隔离（2026-08-14）

针对抬起阶段少数粒子被反复拉扯、跳动的问题，补齐抓取控制与视觉位置 residual 之间的动态
隔离。此前 direct/support 及其邻域只从在线刚度证据和刚度更新中排除，visual residual 本身仍
可移动这些节点；下一物理子步 persistent grip 再把它们投影回 (u_t)，可能形成周期性竞争。

当前实现：

- residual 求解新增 `dynamic_exclusion_mask`；它与资产 fixed mask 合并后，同时清零 residual
  梯度、Adam 后的 candidate、temporal reference 和最终 candidate；
- 抓取未激活时动态掩码为空；激活后种子为当前 `persistent_grip_particle_body>=0` 的 direct
  anchors 和实际 support nodes；
- 只收集直接包含任一种子的 tetra，并把这些 tetra 的四个节点加入掩码。扩张严格执行一次，
  新邻居不成为下一轮种子，因此不产生第二圈；
- live residual 与多 horizon 影子中的 open-loop residual measurement 使用同一掩码；
- 同一掩码继续提供给 online stiffness updater，避免位置修正和材料证据使用不同的控制区；
- result、持久化物理指标、控制台和折叠的 `Visual residual diagnostics` 新增
  `dynamically_excluded_particles`/`grip excluded` 数量。

小白公式文档和 `CURRENT_SUPER_TISSUE_FRAMEWORK.md` 已补充集合公式与运行边界。合成回归增加
两项门禁：动态排除节点的 residual/视觉梯度必须严格为零；两段共享节点的 tetra 链中，从第一
段种子扩张后不得误进入第二段，证明掩码确实只有一环。

验证：

```text
python -m py_compile residual mapper / simulator / runtime / test          PASS
PYTHONPATH=src:examples python scripts/test_super_visual_tissue_residual_mapping.py
  dynamic grip residual/gradient zero                                     PASS
  strict one-tet ring does not enter the second ring                       PASS
PYTHONPATH=src:examples python scripts/test_super_online_tissue_stiffness.py PASS
PYTHONPATH=src:examples python scripts/test_super_stiffness_evaluation.py    PASS
PYTHONPATH=src:examples python scripts/test_super_stiffness_gui_controls.py  PASS
PYTHONPATH=src:examples python scripts/test_super_tissue_persistent_grip.py  PASS
```

当前容器仍无 CUDA/display，因此本轮证明的是掩码边界、数据流和原有物理门禁无回退；实际抓取
轨迹中的跳点幅度变化需要在 GUI 重放中观察 `grip excluded`、anchor error 和 safe scales。

## 下压直接核心收紧到约 2 mm（2026-08-14）

用户确认当前下压观感仍传播太远，并明确要求大幅缩到约 `2 mm`、不得修改现有刚度优化方案。
本轮只修改接触局部性和投影速度重建：

```text
direct top-barrier physical-node core       disabled/full hit face -> 2.00 mm
material projection velocity feedback       0.12 -> 0.03
contact projection velocity feedback        0.35 -> 0.35（保持）
contact spread / pressure shoulder           0 / OFF（保持）
material iterations                          6（保持）
k_dist / k_volume / k_shape                  0.20 / 1e10 / 0.004（保持）
online distance / shape bounds               0.10..2.00 / 0.003..0.020（保持）
```

oriented top barrier 对每个真实最近面命中点计算三角形三个物理顶点的切平面距离，只给
`distance<=2 mm` 的节点写直接向下位移。若粗/瘦三角形三个顶点都略超半径，只选最近一个
节点兜底，避免单侧屏障突然消失，也不恢复整面三点共同下压。双侧 contact patch 建立后，
同一半径继续围绕实测抓取中心限制 direct jaw delta。范围外粒子仍可由固定的 distance/volume/
shape XPBD 自然响应，但没有显式接触位移源。

材料投影仍完整修改几何位置和执行原有 6 passes/体积保护，只将其进入下一子步惯性速度的比例
降至 `0.03`，用于压低 12 substeps 中不断携带的远端长尾。真实接触分量仍以 `0.35` 回写，
所以 q7 闭合推动没有随材料尾巴一起削弱。在线 residual→stiffness 信号、空间平滑、EMA、
candidate/verified、历史稳定门、下一图像双影子和 H=1/3/5/10 评测代码均未改动。

CPU `paper_soft` 完整配置/静态场景门禁已实际编译新的 Warp top-barrier kernel，并通过：

```text
top_barrier_patch_radius                         2.00 mm
material projection velocity feedback            0.03
20-step rest minimum J                            0.99999857
20-step maximum dynamic/fixed displacement        0 / 0 m
all paper_soft gates                              PASS
```

补充回归：

```text
PYTHONPATH=src:examples python scripts/test_super_paper_constraint_xpbd.py   PASS
  current 0.20 / 1e10 / 0.004, 6 passes unchanged; no inversion
PYTHONPATH=src:examples python scripts/test_triangle_skin_contact.py         PASS
PYTHONPATH=src:examples python scripts/test_super_tissue_persistent_grip.py  PASS
PYTHONPATH=src:examples python scripts/test_super_online_tissue_stiffness.py PASS
  candidate/verified, bounds, EMA and 20x distance contrast unchanged
PYTHONPATH=src:examples python scripts/test_super_stiffness_gui_controls.py   PASS
```

当前容器无 CUDA/display，`2 mm` 是“直接接触位移源”的严格配置范围，不冒充已经量出的最终
可见形变截止半径。固定体积和材料连接仍会产生衰减的自然响应；最终观感需 Reset 后连续 GUI
A/B，并可临时关闭 visual residual 判断剩余远端运动来自视觉状态修正还是 XPBD 材料响应。

### 2 mm 强局部档后的温和回调（2026-08-14）

GUI 观察仍有少数粒子乱飘，用户要求邻域稍微增大，并明确指定材料投影速度回灌改为 `0.05`。
当前生产默认值更新为：

```text
direct top-barrier physical-node core       2.00 -> 2.50 mm
material projection velocity feedback       0.03 -> 0.05
contact projection velocity feedback        0.35（保持）
```

局部筛选仍严格围绕实际命中点，粗三角形最近节点兜底、双侧中心 clipping、零 contact spread
和关闭 pressure shoulder 的结构均不变。扩大到 `2.5 mm` 让同一命中面的相邻节点更容易共同
承担向下位移，减少单节点承载造成的尖峰；`0.05` 在抑制远端惯性尾巴的同时，比 `0.03` 保留
更多连续材料速度。刚度优化方案、`k_dist/k_volume/k_shape`、6 passes、online bounds、EMA、
candidate/verified 和多 horizon 验收仍未修改。

回调后的 CPU 门禁再次通过：真实场景启动日志确认 `top_barrier_patch_radius=2.50 mm`，
20-step rest `min J=0.99999857`、零动态/固定漂移；persistent grip 全部门禁和在线刚度
candidate/verified、20x distance range 门禁均为 PASS。

### 材料投影速度回灌默认值改为 0.14（2026-08-14）

用户要求将最近讨论的连续标量默认值设为 `0.14`。由于视觉 residual 排除范围是离散拓扑
环数，不能取 `0.14`，本次明确修改的是 GUI 中的 `Material/grip velocity transfer` 启动值：

```text
material projection velocity feedback       0.05 -> 0.14
direct top-barrier core                      2.50 mm（保持）
visual residual grip exclusion               strict one tet ring（保持）
contact projection velocity feedback         0.35（保持）
```

刚度优化参数和验收流程保持不变。`0.14` 比 `0.05` 将更多材料投影位移转换成下一子步速度，
因此材料跟随更明显，但理论上也会比 `0.05` 更容易保留高频振荡和远端惯性尾巴；该取舍按用户
指定值执行。

验证：`python -m py_compile` 通过；CPU paper-soft 配置门禁通过，确认运行时默认值为
`0.14`，同时直接下压核心仍为 `2.50 mm`。当前环境没有 CUDA，因此本次未重复运行完整
GUI/CUDA 轨迹。

## Liang 论文实验评估方法小白教程与项目迁移方案（2026-08-16）

新增 `参考Liang论文的实验效果评估方法_小白公式版.md`。本文不把当前方法误写成论文严格复现，
而是从实验验收角度详细拆解 Liang et al. arXiv:2309.11656v2 的证据链：

```text
PBD vs PBD-RM                         当前 residual 是否改善对齐
PBD-RM vs PBD-RM-ON                   在线刚度是否改善未来预测
future residual gap, T=10             关闭未来观测后的开放环能力
15 keypoints every 10 frames          独立于点云目标的位置误差
three stiffness initializations       初值敏感性
four real manipulation trajectories   不同组织/轨迹覆盖
```

文档逐步解释双向 Chamfer、residual gap、历史稳定损失、空间刚度平滑、平均未来 gap 和关键点
位移误差，并为每个公式提供直观含义与数字算例；同时明确论文公式的 Chamfer 报告单位、shape
范围/初值和 smoothness 排版存在不宜直接照抄的边界。

迁移部分将论文评估逻辑替换为当前项目真实链路：双目 masked Smooth-L1 `D_vis`、accepted
physical-node residual、candidate/verified、`H=1/3/5/10`、`material_isolation/end_to_end`、
最小 `J`/翻转/穿透/anchor 和阶段 JSON recorder。推荐主对照为 `Fixed PBD / Residual-only /
Full improved`，其中刚度结论必须主要来自 Full 对 Residual-only，而不是只和无视觉 PBD 比。

文档还给出项目专用的 future Gap/相对改善率、2D/3D keypoint、粒子 jitter、commit/H-win-rate、
false reject、边界饱和、edge roughness、RTF 和配对统计公式；规划 soft/default/stiff 三种初值、
两层评测（单 candidate 因果影子 + Reset 后完整轨迹 A/B）、六动作阶段、重复性与最终表格。

当前能力审计结论保持不变：committed candidate 的双协议多 horizon 和持久化 recorder 已实现；
仍需 A/B/C 批量 runner、可复现初值 CLI、与 commit 无关的 scheduled checkpoints、关键点接线、
CUDA 分模块计时和自动聚合报告。仓库中尚无真实完整 `stiffness_evaluation_*` JSONL/summary，
因此该文档定义的是下一步实验规范，不冒充已经得到的效果结论。

## ThinLinc VirtualGL 随主机 NVIDIA 580 驱动自动匹配（2026-08-16）

用户运行 `scripts/run_demo_thinlinc.sh` 时，启动器报告仓库本地 NVIDIA OpenGL `535.104.12`
与已经加载的内核模块 `580.159.04` 不匹配。审计确认问题不在 GUI/仿真参数：旧启动器和
`tools/virtualgl/README.md` 把 `tools/virtualgl/vendor/nvidia-535.104.12` 固定为唯一 runtime，
而主机升级后已经提供完整匹配的系统 GLVND：

```text
kernel module                         580.159.04
/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.580.159.04
/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.580.159.04
/usr/lib/x86_64-linux-gnu/libnvidia-eglcore.so.580.159.04
/usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.580.159.04
/usr/share/glvnd/egl_vendor.d/10_nvidia.json
```

启动器现从 `/proc/driver/nvidia/version` 提取精确版本，优先选择同版本系统 GLVND；只有系统
runtime 不存在且仓库 bundle 与内核版本精确相同时才回退 bundle。启动前分别验证 EGL、GLX、
eglcore、glcore 和 EGL vendor manifest，不再使用 substring grep 或固定 `535`。仍保留
`NVIDIA_GL_ROOT/NVIDIA_GL_VERSION/NVIDIA_EGL_VENDOR_JSON` 显式覆盖，但任何内核/用户态版本
不一致都会拒绝启动。

新增无需 DISPLAY 的预检：

```bash
bash scripts/run_demo_thinlinc.sh --check-nvidia-runtime
```

真正 GUI 验收仍使用 `--check-virtualgl`，renderer 必须为 NVIDIA A800 且不能是 llvmpipe。
若用户自己的 ThinLinc shell 中 `nvidia-smi` 失败或 `/dev/nvidia*` 缺失，则属于主机设备节点/
驱动模块问题，需要管理员处理；仅切换 OpenGL runtime 无法修复。

静态和 runtime 选择验证：

```text
bash -n scripts/run_demo_thinlinc.sh                         PASS
bash scripts/run_demo_thinlinc.sh --check-nvidia-runtime    PASS
  kernel/OpenGL 580.159.04, system GLVND, system EGL JSON
显式请求旧 535 runtime                                      正确拒绝
git diff --check                                             PASS
```

当前 `/tmp/.X11-unix` 只有用户 `jwshan` 当次会话的 `X10`，没有 `X14`；ThinLinc display 号属于
会话动态分配，旧会话曾使用 `:14` 不代表本次也必须使用。启动器继续优先保留已有 `DISPLAY`，
为空时才自动寻找当前用户可访问的 ThinLinc display。受 Codex 沙箱的 Xauthority/GPU device
隔离影响，最终 `--check-virtualgl` renderer 验收需要用户在自己的 ThinLinc terminal 中执行。

## Tip entry 3.3 mm / capture penetration 2.9 mm 与初步评测数据审计（2026-08-16）

按用户指定更新纸式软组织模式的启动默认值：

```text
tip entry allowance                         2.50 -> 3.30 mm
hidden top-barrier distal length             3.00 -> 3.31 mm
maximum capture penetration                  1.00 -> 2.90 mm
jaw contact distal length                    5.00 mm（保持）
direct downward particle core                2.50 mm（保持）
stiffness excessive-penetration pause gate   1.00 mm（保持）
```

`3.31 mm` 不是新的用户旋钮，而是为了在 tip 放宽到 `3.30 mm` 后仍满足
`tip < barrier <= 5 mm jaw band` 的隐藏几何末端。这与 GUI 原有的自动扩展规则一致，保留
`0.01 mm` 最窄 top-barrier band。`2.90 mm` capture 值包含原有 `0.40 mm` contact margin，
因此对真实 signed overlap 的容忍上限约为 `2.50 mm`；它会更容易在深的双侧接触上建立
persistent grip，也更容易锁住错误穿透状态。在线刚度的过深穿透暂停门仍为 `1.00 mm`，
不与 capture 门限混合放宽。

CPU 实场景回归结果：

```text
py_compile                                  PASS
paper-soft full-scene configuration         PASS
particles / tetrahedra                      4059 / 15830
reference-equilibrium steps                 20
minimum tetrahedron volume ratio J_min      0.9999985695
maximum dynamic displacement                0.0 m
maximum anchor drift                        0.0 m
persistent-grip behavioral gates            13 / 13 PASS
GUI stiffness/contact control gates          8 / 8 PASS
evaluation recorder/formula gates            5 / 5 PASS
```

这些数据只能证明新默认值已真正进入 runtime，初始平衡没有因此自发翻转/漂移，且抓取与
GUI 状态机仍通过。它们不是完整轨迹 A/B 的图像误差结论。

现有 `outputs/stiffness_eval_full_default_r01` 已指向正确数据集，并配置
`H=1/3/5/10`、`material_isolation/end_to_end`、residual 反馈和在线刚度，但数据审计为：

```text
total events                                1
reset events                                1
events with image metrics                   0
events with material metrics                0
events with physical metrics                0
events with prediction metrics              0
```

因此目前真实轨迹的 baseline/candidate gap、相对改善率、分阶段误差和 horizon 曲线都是
`无可计算样本`，不能报一个虚假数字。评测回归中的 `baseline_gap=0.10` / `candidate_gap=0.08`
仅是手写合成 fixture（绝对改善 `0.02`，相对 `20%`），只用来证明 recorder 和双协议多 horizon
公式链路会计算，明确不作为本项目实验效果。下一份有效输出必须使用新目录，在 GUI 点击
Play 并让完整轨迹产生 visual update/open-loop prediction；同一 A/B 的两组都必须固定本次
`3.30/2.90 mm` 接触配置。

## 默认接触值改为 3.0/2.8 mm，并完成 frame 350..560 三组初步对照（2026-08-16）

> 本节的默认值明确取代上一节临时设置的 `3.30/2.90 mm`。

按用户最新要求更新为：

```text
tip entry allowance                         3.30 -> 3.00 mm
hidden top-barrier distal length             3.31 -> 3.01 mm
maximum capture penetration                  2.90 -> 2.80 mm
stiffness excessive-penetration pause gate   1.00 mm（保持）
```

`3.01 mm` 继续保留 `0.01 mm` 隐藏 barrier band，满足
`tip < barrier <= 5 mm jaw band`。`2.80 mm` capture 值包含 `0.40 mm` contact margin，
因此允许的真实 signed overlap 约为 `2.40 mm`。直接下压粒子核心、刚度上下界、
EMA、candidate/verified 和 volume 固定策略均未改变。

用户所说的三组对照被明确定义为：

```text
fixed_pbd                    visual feedback OFF, online stiffness OFF
residual_only                residual ON, online stiffness OFF
residual_online_stiffness    residual ON, online stiffness ON
```

新增 `--evaluation-headless`、起始帧/帧数/每帧物理步数 CLI，评测时不构建 OpenGL GUI。
三组都在每个新视频帧的物理步之后、当前帧 residual 写回之前，用同一 mapper、左右目
mask 和 Smooth-L1 记录 `trajectory_observation.image.prediction_loss`。这是三组的公平主指标；
residual 的 `mean_loss_after` 只作为“当前图像拟合能力”辅助指标。

新增：

```text
scripts/run_super_three_way_stiffness_comparison.sh
scripts/summarize_super_three_way_comparison.py
```

前者顺序跑三个独立进程和独立初始状态；后者只在三组共同 frame id 上做配对汇总，
输出 `comparison.json` 和 `comparison.md`。

实际 CUDA 初步实验固定：

```text
video frames                    350..560 inclusive
common observations             211 per mode
physics steps per video frame   3
formal physics steps            633 per mode
tip / capture defaults          3.00 / 2.80 mm
residual iterations             8
visual update interval          3 physics steps
```

主结果（平均 pre-feedback 双目 masked Smooth-L1 prediction loss）：

| 模式 | 平均损失 | 中位数 | 相对 fixed | 最小 `J` | 最大穿透 | 翻转 tet |
|---|---:|---:|---:|---:|---:|---:|
| Fixed PBD | `0.02412187` | `0.02438195` | baseline | `0.010093` | `2.583 mm` | `0` |
| Residual-only | `0.02111430` | `0.02124880` | `12.468%` 改善 | `0.010069` | `2.328 mm` | `0` |
| Residual + online stiffness | `0.02111612` | `0.02117986` | `12.461%` 改善 | `0.010042` | `2.307 mm` | `0` |

Residual-only 相对 fixed 的分阶段改善为：

```text
press       0.02290677 -> 0.02025266    11.587%
capture     0.02456509 -> 0.02120908    13.662%
lift        0.04094047 -> 0.03300544    19.382%
place       0.04153939 -> 0.03436061    17.282%
```

Residual-only 的 211 次 residual 全部 accepted，单次更新前/后损失均值为
`0.02111430 -> 0.01942191`，平均当帧降幅 `7.826%`。这和下一帧 prediction loss 的
`12.468%` 是两个不同问题，不得混在一起报。

最重要的负结果：在线组 211/211 次 visual update 都被刚度全局门暂停，其中每次理由
都包含 `excessive_penetration`；`stiffness_validation=0`、`open_loop_started=0`、commit=0。因此这一轮
只证明 residual 相对 fixed 有改善，**不能证明实时刚度额外有效**。在线组相对
residual-only 反而高 `0.0086%` 的极小差别属于 GPU 原子投影重复性噪声，不归因于材料。

结果文件：

```text
outputs/stiffness_three_way_350_560_prelim_20260816/comparison.md
outputs/stiffness_three_way_350_560_prelim_20260816/comparison.json
outputs/stiffness_three_way_350_560_prelim_20260816/{fixed_pbd,residual_only,residual_online_stiffness}/
```

下一轮若要真正比较 online stiffness 与 residual-only，必须先处理“全程穿透超过 `1.0 mm`”：
优先检查 frame 350 起步的接触初态/预热路径，或为实验显式标记一个更合理的学习穿透门；
不应把 capture `2.8 mm` 直接等同于刚度学习安全门，也不应在 0 次 commit 时宣称在线优化有效。

回归验证：`py_compile` / shell syntax / `git diff --check` 通过；原有评测记录器
`H=1/3/5/10` 双协议 5/5 门禁、GUI 刚度/接触 8/8 门禁、paper-soft 完整 CPU 场景
门禁均 PASS。CPU 启动日志确认 `grip_capture_max_penetration=2.8mm`、
`top_barrier_band=3.00..3.01mm`，20 步平衡的 `J_min=0.9999985695`、最大自发位移和锚点漂移均为 0。

## 在线刚度穿透暂停门 1.0 -> 2.0 mm 与重跑结果（2026-08-16）

按用户指定将：

```text
STIFFNESS_MAXIMUM_PENETRATION_M              0.0010 -> 0.0020 m
grip maximum capture penetration             2.8 mm（保持）
local stiffness-learning J floor             0.03（保持）
tip entry allowance                           3.0 mm（保持）
```

新门限只放宽 online stiffness 全局 gate，不改变 contact correction、residual、capture 或物理材料初值。
评测 metadata 新增 `stiffness_maximum_penetration_m`，避免后续报告无法追溯门限。

在上一轮 online 穿透序列上的静态反算预计 `2.0 mm` 可使 186/211 帧通过门限；
实际重跑 frame `350..560`、每帧 3 物理步后得到：

```text
visual updates                              211
gate eligible                               185
gate paused                                  26
  excessive penetration                     17
  transition cooldown                         4
  rapid q7                                    4
  capture/release + rapid q7                  1
stiffness validation                        184
validation rejected                         184
open-loop started / committed                 0 / 0
```

26 次暂停而不是静态估计的 25 次，来自实际重跑的 GPU 投影微小差异和三次过渡冷却。
门限已成功解决“211/211 全程不学习”问题，但未产生任何 verified 刚度改变。

184 个 validation 中：

```text
complete baseline/candidate shadow pairs     181
rejected before rollout                        2
cancelled at playback end                       1
prediction_gap_not_improved                   181
volume_quality_regression                      51
positive but sub-threshold gap                106
gap improvement meeting required margin         0
mean candidate gap improvement         0.0000009589
mean required improvement              0.0000210533
best / worst gap improvement            +0.0000285357 / -0.0000087172
```

51 个 volume rejection 与 181 个 gap rejection 有重叠：130 次只因 gap 不足，51 次同时 gap 不足且体积退化。
由此可以明确：当前的主阻塞已不是穿透门，而是刚度 candidate 对下一帧的改善量普遍小于提交门，
并且部分 candidate 会使最小体积比退化。

使用原 Fixed PBD / Residual-only 与新 `2.0 mm` online 组按 211 个共同帧配对：

| 模式 | 平均 prediction loss | 相对 fixed | 最小 `J` | 最大穿透 | 翻转 tet |
|---|---:|---:|---:|---:|---:|
| Fixed PBD | `0.02412187` | baseline | `0.010093` | `2.583 mm` | `0` |
| Residual-only | `0.02111430` | `12.468%` 改善 | `0.010069` | `2.328 mm` | `0` |
| Residual + online, 2.0 mm gate | `0.02112122` | `12.440%` 改善 | `0.010015` | `2.335 mm` | `0` |

在线组相对 residual-only 损失高 `0.0328%`。因为 commit=0、verified 材料场从未改变，
这个微小差异不能归因于刚度优化；本轮仍只能确认 residual 相对 fixed 的约 `12.47%` 改善。

结果：

```text
outputs/stiffness_online_2mm_350_560_prelim_20260816/
outputs/stiffness_three_way_2mm_350_560_prelim_20260816/comparison.json
outputs/stiffness_three_way_2mm_350_560_prelim_20260816/comparison.md
```

验证：在线刚度 updater 20/20 独立门禁 PASS，candidate/verified 隔离、拒绝回滚、局部质量屏蔽、
20x distance 区域范围和 GUI reconfigure 均未被这次全局穿透门改动破坏；原有
`H=1/3/5/10` 记录器 5/5 门禁、`py_compile` 和 `git diff --check` 也通过。

## 保留原刚度安全闭环，并增加分阶段诊断（2026-08-16）

用户要求不要为了“接触时都放开”而随意改变既有刚度优化方案。已撤回尚未验证的
“捕获/快速 q7/接触片突变/过度穿透只记录而不暂停”尝试；当前正式路径继续保留：

```text
capture_or_release / rapid_q7 / contact_patch_switch / excessive_penetration
    -> 触发 3 次 visual update cooldown
    -> 必要时清空不再可比的 signal history
    -> candidate 仍必须通过 next-frame shadow validation 才可 commit
```

已确认的 `STIFFNESS_MAXIMUM_PENETRATION_M=2.0 mm` 保留。也就是说，接触本身不会永久关闭
刚度更新；只在接触状态发生突变或穿透超过 2.0 mm 时短暂停顿。frame 350..560 的真实数据为
211 次 visual update、185 次 gate eligible、26 次 paused、184 次 stiffness validation，足以直接
反证“只要接触就放弃刚度更新”。

`scripts/summarize_super_three_way_comparison.py` 的报告 schema 更新为 v2，新增：

```text
按动作阶段的三组 prediction loss、样本数和相对改善
从 frame 350 起对齐的每 50 帧时间窗口
按动作阶段/时间窗口的 baseline-candidate 影子验证统计
候选降损次数、达到 required margin 次数和体积退化次数
```

三组分阶段 prediction loss：

| 阶段 | 帧数 | Fixed | Residual-only | Residual 相对 Fixed | Online | Online 相对 Residual |
|---|---:|---:|---:|---:|---:|---:|
| press | 195 | `0.02290677` | `0.02025266` | `+11.587%` | `0.02025490` | `+0.0111%`（略差） |
| capture | 2 | `0.02456509` | `0.02120908` | `+13.662%` | `0.02088367` | `-1.5343%`（数值更低） |
| lift | 13 | `0.04094047` | `0.03300544` | `+19.382%` | `0.03312422` | `+0.3599%`（略差） |
| place | 1 | `0.04153939` | `0.03436061` | `+17.282%` | `0.03448858` | `+0.3724%`（略差） |

capture/place 只有 2/1 帧，不能用来调材料；更重要的是本轮 `commit=0`，所以 Online 与
Residual-only 的这些小差异都属于不同 CUDA 运行的数值波动，不能声称为刚度收益。

原安全门按阶段的实际放行情况：

| 阶段 | visual updates | eligible | paused | 主要暂停原因 |
|---|---:|---:|---:|---|
| press | 195 | 175 | 20 | excessive penetration 17、cooldown 2、rapid q7 1 |
| capture | 2 | 0 | 2 | capture/release + rapid q7 |
| lift | 13 | 9 | 4 | rapid q7 2、cooldown 2 |
| place | 1 | 1 | 0 | - |

因此当前行为是“接触过渡阶段短暂停，稳定接触/按压/抬升时继续提候选”，不是“一接触就永久停更”。
capture 两帧都暂停是因为它恰好就是闭合状态切换段；若后续希望评估稳定夹持阶段，应延长该阶段
或细分 `capture transition` 与 `stable grip`，而不应直接删除所有安全门。

完整影子验证按阶段统计：

| 阶段 | validation events | 完整 shadow pairs | candidate loss 更低 | 达到 required | 平均降损 | 平均 required | volume regression |
|---|---:|---:|---:|---:|---:|---:|---:|
| press | 174 | 172 | 97 | 0 | `0.442e-6` | `20.460e-6` | 46 |
| lift | 10 | 9 | 9 | 0 | `10.828e-6` | `32.394e-6` | 5 |

这说明候选在抬升阶段表现出比按压阶段强得多的正趋势：9/9 次都让下一帧 loss 下降，平均达到
required 的约 `33.4%`，最佳单次为 `28.536e-6`，但仍低于对应的 required margin，且 5/9 次
伴随局部体积质量退化。因此当前不应删除接触门禁或强行 commit；下一轮应优先延长 capture/lift/place
各阶段样本并重复运行，确认这种阶段差异是否可重复，再决定是否只调整验证 margin、candidate step
或阶段 horizon。

更新后的结果文件：

```text
outputs/stiffness_three_way_2mm_350_560_prelim_20260816/comparison.md
outputs/stiffness_three_way_2mm_350_560_prelim_20260816/comparison.json
```

本次复核：online updater `20/20` 门禁 PASS；stiffness evaluation `5/5` 门禁 PASS；
主例程与汇总脚本 `py_compile` PASS；相关文件 `git diff --check` PASS。Warp 在该测试进程中
无法初始化 CUDA 后自动使用 CPU，不影响这两组确定性逻辑门禁；本节实验数值仍来自前述 A800 CUDA 运行。

## 近全开放刚度放行与整体效果评估（2026-08-16）

按用户要求将 candidate admission 尽可能放宽，但没有删除 candidate/verified 隔离及影子回滚：

```text
global rapid-q7 pause                 0.50 rad/s threshold -> disabled
transition cooldown                   3 -> 1 visual update
maximum learning penetration          2.0 -> 2.8 mm（与 grip capture 对齐）
prediction relative margin            0.1% -> 0
prediction absolute margin            1e-6（保持）
```

因此稳定接触、高速 q7、按压和抬升均可提出 candidate；仅 capture/release 状态翻转当帧、
contact-patch 突变当帧或穿透超过 2.8 mm 时暂停。快速 q7 仍不会写入最多 4 个的准静态
history snapshot。候选提交仍需满足：总体 loss 至少下降 `1e-6`、左右相机不过度退化、
局部体积不过度下降、无翻转、穿透和锚点误差不过门。volume 继续固定为 `1e10`。

旧 2.0 mm 日志静态反算得到 181 个完整 shadow pairs 中约 44 个同时通过新 loss floor 和
原 volume gate；因为真正 commit 会改变后续刚度场和轨迹，最终以全新 CUDA 重跑为准。

实际 A800 CUDA 运行：frame `350..560` inclusive，211 个视频帧，每帧 3 个正式 physics steps：

```text
visual updates                         211
gate eligible                          210
gate paused                              1  (capture_or_release)
stiffness validations                  209
committed                               33
rejected                               176
open-loop started/completed/incomplete  33 / 27 / 6
```

33 次 commit 按阶段为 press 24、lift 9；没有 capture/place commit。validation 的主要拒绝原因：

```text
prediction_gap_not_improved            160
volume_quality_regression                76
grip_anchor_regression                    1
```

各阶段候选：

| 阶段 | 完整 shadow | 候选 loss 更低 | 过 `1e-6` loss 门 | 最终 commit | 平均 gap 改善 |
|---|---:|---:|---:|---:|---:|
| press | 193 | 65 | 35 | 24 | `-0.982e-6` |
| capture | 0 | 0 | 0 | 0 | - |
| lift | 13 | 12 | 11 | 9 | `+2.662e-6` |
| place | 1 | 1 | 1 | 0 | `+260.308e-6`，但 volume regression |

三组总体公平指标仍为当前帧 residual 写回前的双目 masked Smooth-L1 prediction loss：

| 模式 | 211 帧平均 | 中位数 | 相对 Fixed | 最小 J | 最大穿透 | inverted |
|---|---:|---:|---:|---:|---:|---:|
| Fixed PBD | `0.02412187` | `0.02438195` | baseline | `0.010093` | `2.583 mm` | 0 |
| Residual-only | `0.02111430` | `0.02124880` | `12.468%` 改善 | `0.010069` | `2.328 mm` | 0 |
| Residual + loose online | `0.02114035` | `0.02118084` | `12.360%` 改善 | `0.010003` | `2.349 mm` | 0 |

宽松 online 相对 residual-only 的平均 loss 高 `0.1233%`，相对旧 2.0 mm/0-commit online
高 `0.0906%`。逐帧比较 online 更低 110 帧、residual-only 更低 100 帧、相同 1 帧；差值
中位数为 `-1.202e-6`，但均值为 `+26.044e-6`，说明少量明显退化帧把平均值拉坏。
退化主要集中在 frame 547..555 的闭合后抬升过渡；frame 557..560 又明显优于 residual-only。

分阶段主损失：

| 阶段 | N | Residual-only | Loose online | Online 相对 residual |
|---|---:|---:|---:|---:|
| press | 195 | `0.02025266` | `0.02024685` | `-0.0287%`（略好） |
| capture | 2 | `0.02120908` | `0.02094428` | `-1.2485%`（样本不足） |
| lift | 13 | `0.03300544` | `0.03335511` | `+1.0594%`（变差） |
| place | 1 | `0.03436061` | `0.03697214` | `+7.6004%`（样本不足） |

33 个已提交 candidate 的多 horizon 结果：

| Protocol | H | N | 正改善 | 平均 gap 改善 | 中位数 |
|---|---:|---:|---:|---:|---:|
| end-to-end | 1 | 33 | 33 | `+4.534e-6` | `+3.049e-6` |
| end-to-end | 3 | 32 | 21 | `-9.022e-6` | `+1.899e-6` |
| end-to-end | 5 | 31 | 14 | `-19.063e-6` | `-0.130e-6` |
| end-to-end | 10 | 27 | 18 | `-25.672e-6` | `+0.503e-6` |
| material-isolation | 1 | 33 | 33 | `+4.534e-6` | `+3.049e-6` |
| material-isolation | 3 | 32 | 21 | `-8.968e-6` | `+1.899e-6` |
| material-isolation | 5 | 31 | 16 | `+4.832e-6` | `+0.043e-6` |
| material-isolation | 10 | 27 | 18 | `+3.877e-6` | `+0.503e-6` |

所有 horizon 的 inverted tet 最大值均为 0；只有一个 H=10 pair 的 candidate 穿透相对 baseline
增加超过 0.1 mm。material-isolation 的 H=5/10 平均仍略正，但 end-to-end H=3/5/10 平均转负，
说明局部材料更新本身未必错误，主要问题是 H=1 提交标准没有覆盖之后的 residual/接触闭环耦合。

最终 verified field（第 33 次 commit，frame 559）：

```text
distance min / median / mean / max    0.100000 / 0.200000 / 0.199086 / 0.530754
shape    min / median / mean / max    0.003000 / 0.004000 / 0.004029 / 0.017292
```

中位数保持初值，证明更新仍是局部的；局部节点已经触及软化下限，同时局部硬化到 distance
`0.531`、shape `0.0173`，尚未触及配置上限 `2.0/0.020`。

结论：近全开放成功解决了“没有 commit、无法评估”的问题，且没有产生翻转或显著穿透恶化；
但目前 **不能证明 online stiffness 有整体收益**，211 帧主指标反而比 residual-only 差 `0.1233%`。
下一步不应继续放松 volume/翻转门；更合理的是把 commit 判据从只看 H=1 改为至少 H=3，或对
press/lift 使用不同 horizon，并在完整 capture/lift/place/release 轨迹上重复运行。

结果：

```text
outputs/stiffness_online_loose_350_560_20260816/
outputs/stiffness_three_way_loose_350_560_20260816/comparison.json
outputs/stiffness_three_way_loose_350_560_20260816/comparison.md
```

验证：online updater `20/20` PASS；扩展后的 stiffness evaluation/admission `10/10` PASS；
报告关键计数 assertions PASS；主例程、评测和汇总脚本 `py_compile` PASS；相关文件
`git diff --check` PASS。211 帧正式实验在沙箱外使用 A800 CUDA 完成；逻辑单测在 CPU 完成。

## H=5 真实闭环、极松门诊断与固定滞后状态接回（2026-08-17）

### 原因复核

近全开放 H=1 版本的 33 次 commit 全部通过单帧检查，但 end-to-end H=3/5/10 平均变负，
总体比 residual-only 差 `0.1233%`。进一步发现旧 `end_to_end` 影子只推进物理/抓取，没有在
中间帧重放视觉 residual；而 admission 只看 horizon 终点，不对应三组报告使用的逐帧平均
pre-residual prediction loss。

已实现：

- commit horizon 固定为 5 个视频帧；pending 期间 live residual 继续工作；
- 影子使用真实 capture/release 状态机，并在候选帧之后、终点之前逐帧重放 residual；
- 保存候选创建时的 temporal residual，候选帧不重复写回；
- admission 目标改为 H=1..5 的平均 pre-residual prediction loss；
- 候选 log-step 倍率 line search 为 `0.5/1.0/1.5/2.0`；
- metadata/validation 记录累计目标、终点目标、四个倍率及真实学习率；
- 每个影子分支结束恢复 live 物理、控制器、当前数据帧和 Gaussian。

H=5 endpoint 旧版、较大 lr、line search 与累计目标调参的主要结果：

| 版本 | commit | Online 相对 residual-only |
|---|---:|---:|
| H=5 endpoint，lr=0.18 | 11 | `-0.2280%` |
| H=5 endpoint，lr=0.36 | 16 | `+0.0906%`（退化） |
| H=5 endpoint + line search | 20 | `-0.1425%` |
| H=5 cumulative，margin `1e-6` | 20 | `+0.0523%`（退化） |
| H=5 cumulative，margin `1e-5`，仅材料 | 8 | `-0.1685%` |
| H=5 cumulative，margin `5e-6` | 10 | `+0.0579%`（退化） |

负号表示 loss 更低。`1e-5` 最可靠：8 个 commit 的后验 end-to-end H=5 全为正，逐帧
`130:80`，0 翻转、0 次 >0.1 mm 穿透退化。因此正式证据门恢复并保留 `1e-5`。

### 极松门禁试验并恢复

按要求运行一次独立极松进程：

```text
local J learning floor                 0.03 -> 0.005
maximum learning penetration           2.8 -> 10 mm
transition cooldown                    1 -> 0
prediction margin                      1e-5 -> -1
camera abs/relative regression         1e-6/0.005 -> 0.1/1.0
volume abs/relative drop               0.01/0.02 -> 1.0/1.0
penetration/anchor tolerance           0.1/0.05 mm -> 10/10 mm
history relative/absolute tolerance    0.10/0.01 mm -> 10/10 mm
```

仍保留有限数值、不新增翻转、材料绝对上下界和 maximum log-step。metadata 确认 profile 为
`extreme_loose_trial`。结果为 41 commit、总体 `0.02112722`，比 residual-only 高 `0.0612%`，
逐帧 `89:121`；press/capture 分别退化 `0.2149%/0.7442%`，H=5 end-to-end 只有 `17/41`
正改善。临时环境开关和条件常量随后从源码删除，默认已恢复正式值。

诊断结果：门禁不是收益瓶颈；过度放开会把少量大幅退化帧和长期材料漂移放进 live。

### 固定滞后状态接回

H=5 candidate 是从 5 帧前快照用已经观察到的命令和图像重放得到的因果固定滞后估计。
旧实现通过验证后只 commit material，却丢弃这条更优状态。现在 commit 时同时接回候选的
physics/Gaussian/contact/material 辅助状态和 temporal residual，并设置：

```text
all-particle state RMS cap       0.25 mm
single-particle state cap        1.00 mm
```

超限时仍可提交材料，但不接回状态。CPU fake-sim 门禁覆盖允许/拒绝两条路径。

只跑 H=5 的完整 211 帧调参轮得到：Online `0.02096992`，相对 residual-only 改善
`0.6838%`，逐帧 `187:23`，所有动作阶段与 50 帧窗口均改善；8 次 commit 中 7 次接回，
接回 RMS `0.048..0.057 mm`，最后一次因单粒子 `3.117 mm` 被拒绝接回。0 翻转，最大穿透
`2.284 mm`。配对 t 检验 `p=2.07e-18`。

### 最终 H=1/3/5/10 完整复核

为避免只报告 H=5 最好结果，最终另跑完整多 horizon 诊断；共同配置为 frame `350..560`、
211 帧、每帧 3 物理步、tip entry `3.0 mm`、capture max `2.8 mm`。

| 模式 | mean prediction loss | 相对 Fixed | 相对 residual-only |
|---|---:|---:|---:|
| Fixed PBD | `0.02412187` | baseline | - |
| Residual-only | `0.02111430` | `-12.468%` | baseline |
| Residual + online + fixed-lag | `0.02105285` | `-12.723%` | `-0.2911%` |

配对逐帧胜负 `169:41:1`；paired t-test `t=-3.0521, p=0.0025655`；Wilcoxon 单侧
`p=6.62e-18`。所以相对前两组的主指标改善在这条轨迹上达到统计显著。

分阶段相对 residual-only：press `-0.3559%`、capture `-2.0119%`、lift `-0.0087%`、
place `+5.7615%`；capture/place 仅 2/1 帧。分窗口：350--399 `-0.4042%`、400--449
`-0.5027%`、450--499 `-0.0794%`、500--549 `-0.3620%`、550--560 `+0.1932%`。

安全：0 inverted tet、minimum J `0.0100099`、maximum penetration `2.3529 mm`；所有
horizon 均 0 次 >0.1 mm 穿透退化。11 次 commit 中 9 次接回；最大接回 RMS `0.2112 mm`，
2 次因单粒子 maximum 超 `1 mm` 拒绝接回。

end-to-end horizon：

| H | N | positive | mean gap improvement | median |
|---:|---:|---:|---:|---:|
| 1 | 11 | 8 | `+1.641e-6` | `+3.859e-6` |
| 3 | 11 | 9 | `+31.289e-6` | `+17.591e-6` |
| 5 | 11 | 10 | `+29.215e-6` | `+20.342e-6` |
| 10 | 10 | 7 | `-15.352e-6` | `+13.845e-6` |

H=10 均值仍受 3 个负样本影响；完整 Reset 到 release、独立重复、关键点误差和 CUDA FPS
仍未完成，不能把单轨迹统计显著等同于跨轨迹材料辨识完成。

正式结果：

```text
outputs/stiffness_online_h13510_fixed_lag_final_350_560_20260817/
outputs/stiffness_three_way_h13510_fixed_lag_final_350_560_20260817/comparison.json
outputs/stiffness_three_way_h13510_fixed_lag_final_350_560_20260817/comparison.md
```

验证：`py_compile` PASS；stiffness evaluation/admission/fixed-lag `13/13` PASS；online updater
`20/20` PASS；A800 CUDA 完整轨迹 PASS；极松试验后正式常量复核通过。

## SuFIA 组织数据：低高光纹理、全视角初始化与平滑牵拉（2026-08-18）

所有修改只位于 `embodied_gaussians_fixed_super_best_sim`，原始
`Embodied_gaussians_fixed_super_best` 未改动。

- 组织外观继续使用官方 Body 4096×4096 diffuse/normal；改为 roughness `0.88`、
  specular color `0.012` 的受控低高光材质。UV 窗口缩小到原来的约 53%，天然斑驳
  纹理尺度约放大 1.9 倍；三个颜色通道使用相同的 `1.55` 对比度增益，再以固定
  bias 调成中等亮度暖肉色，不生成随机纹理，也不把十区域 ID 编进颜色。
- 新增 `canonical_scan/`：任务前隐藏 PSM、腹部和桌面，默认使用 12 个方位角×
  4 个仰角并加正上/正下极点，共 50 个静止组织 RGB-D/mask/标定视角，覆盖顶面、
  底面和侧缘。该目录是允许的重建初始化输入，不含区域刚度或未来运动。
- 旧版把一整块矩形组织设成刚性夹持位移，导致局部近 90° 硬折线。新版只保留
  约 8×8 mm 夹持核心；外围 34×30 mm 椭圆支持域使用余弦连续衰减的柔顺力，
  材料感知 Neo-Hookean XPBD 和十个隐藏刚度区域仍参与最终形变。
- 最终外观低分辨率 A800 预览位于
  `data/sim/tissue_retraction_warm_texture_preview_006/`。任务相机组织像素均值约为
  RGB `(164,107,94)`，色度差中位数约 `76`；既不使用上一版的橙色，也不使用
  粉白低对比色。硬夹持版最大/p99 相邻边
  位移跳变为 `41.24/37.73 mm`，平滑版降为 `12.29/7.19 mm`；平滑耦合包含
  16 个核心节点和 532 个连续过渡节点。渲染中牵拉曲面由硬直立板变为连续拱形。
- 数据验证器新增全视角 RGB-D 完整性、扫描不含刚度标签、组织颜色/高光以及平滑夹持
  权重门禁。正式评估仍只报告 3D/2D 轨迹误差和 PSNR/SSIM/LPIPS，不报告刚度误差。

用户确认 180 帧在 30 fps 下仅 6 秒、运动偏快；正式包装脚本已改为默认 300 帧
（10 秒）。牵拉幅度仍约 `28.6 mm`，但各阶段按归一化进度展开到更长物理时间；
训练/未来预测固定为前 240 帧/后 60 帧。64 spp 单 A800 实测任务帧耗时过长，
正式包装脚本改为 16 spp；双 A800 试跑在 Replicator 语义分割回读处触发
`CUDA illegal address`，已明确拒绝并恢复经验证稳定的单 A800 PathTracing。仍保持
1536×1536、4K 原生纹理和无损 PNG，物理/相机/深度/轨迹真值不受渲染吞吐设置影响。
当前只完成预览验收，尚未生成最终 300 帧 1536×1536 正式数据集。

## 仿真三组评估与视觉/刚度修正（2026-08-20）

初版正式消融发现旧强档失败：PBD 全序列 3D/2D Mean 为 `0.932 mm / 4.785 px`，
旧视觉残差为 `1.883 mm / 9.979 px`，旧视觉+刚度为 `2.331 mm / 11.826 px`。
原因不是迭代不足：绝对 RGB 把首帧固定外观差异解释成位移，增量时间项又重复使用上一帧
已经回写的增量；旧刚度在 frame 100 已把中位数从 `0.31/0.0058` 推到接近
`3.0/0.05`，未来段因此发散。

修正内容：

- 第 0 帧缓存观测/渲染外观差，后续优化相对 RGB 变化；每帧增量的时间参考归零；
- 第 0 帧只标定，随后只在 `grasped` 区间启用视觉反馈；frame 240 后仍完全开环；
- 低于 `1e-4` 的校准视觉损失不回写，精确双目重渲染要求两目都严格不变差；
- 参数筛选确定视觉为 12 Adam、`0.06 mm/轮`、单帧上限 `1.0 mm`；更强的
  16 Adam、`0.12 mm/轮`、`1.5 mm` 已实测开始损伤 3D；
- 刚度最终使用 `log_lr=0.03`、单步 log cap `0.02`、EMA decay `0.90`、三轮空间
  平滑，distance `0.18..0.65`、shape `0.003..0.012`。

无渲染 300 帧筛选中，修正后视觉+刚度相对同一 PBD：全序列 3D Mean
`0.9323 -> 0.9112 mm`、2D Mean `4.7850 -> 4.5244 px`；后 20% 开环 3D Mean
`2.5164 -> 2.4167 mm`、2D Mean `13.8819 -> 12.8471 px`、2D RMSE
`18.1624 -> 16.9999 px`。frame 239 的局部 distance 为 `0.18/0.310/0.417`
（min/median/max），shape 为 `0.00319/0.00580/0.00724`，frame 240--299 完全冻结。
这些筛选只用于确定参数；正式 PSNR/SSIM/LPIPS 三组需要用修正后配置重新导出。

## RGB-only 多初值材料分支选择（2026-08-20）

在不改变现有 `PBD + visual residual + online stiffness` 内部算法的前提下，新增一个
纯 RGB 外层。依据 Liang 等人（https://arxiv.org/abs/2309.11656）对初始化敏感性的
讨论和“最近 20 帧均匀抽 4 个历史快照”的设置，同时运行三组**全局均匀**材料初值：

```text
soft     distance=0.22, shape=0.0042
nominal  distance=0.31, shape=0.0058
firm     distance=0.48, shape=0.0085
```

每一支都继续运行原来的视觉残差和局部在线刚度更新；没有读取十区域材料标签。分支选择只
使用 frame `220/226/233/239` 的双目 masked robust RGB loss，完全不读取深度、2D/3D
轨迹真值或材料真值。RGB 分数依次为 `0.0268170 / 0.0281354 / 0.0300331`，因此因果地
选择 `soft`。后 20% 仍在 frame 240 冻结视觉残差和刚度，开环预测 60 帧。

相对原正式 `PBD + visual residual + stiffness`：

| 区间 | 3D Mean | 3D RMSE | 3D P95 | 双目 2D Mean |
|---|---:|---:|---:|---:|
| 全 300 帧 | `0.9060 -> 0.8495 mm` (`-6.24%`) | `1.6213 -> 1.4757 mm` (`-8.99%`) | `4.1530 -> 3.6978 mm` (`-10.96%`) | `4.5061 -> 4.3332 px` (`-3.84%`) |
| 后 20% 开环 | `2.3911 -> 2.0339 mm` (`-14.94%`) | `2.9900 -> 2.5533 mm` (`-14.61%`) | `5.8476 -> 5.1672 mm` (`-11.63%`) | `12.7640 -> 11.0033 px` (`-13.79%`) |

相对单纯 PBD，后 20% 的 3D/2D Mean 分别降低 `19.17% / 20.74%`。未来渲染也一致
改善：相对原 visual+stiffness，PSNR `+0.035 dB`、SSIM `+0.00317`、LPIPS
`-1.77%`；相对 PBD 为 `+0.470 dB / +0.00445 / -3.09%`。

必须同时保留负面结果：soft 在前 80% 同化段的 3D/2D Mean 相对原 visual+stiffness
变差 `3.49% / 7.87%`，所以不能声称所有区间全指标提升；它的收益集中在论文要求的未来
开环预测和全序列总体误差。直接把 RGB 数据项放大并增加到 30 次 Adam 的探针让 3D Mean
变差，因此该失败参数入口已移除。Farneback/RAFT 的纯 RGB 光流在大形变和长时序中产生
明显漂移，且双目独立流经三角化会放大视差误差，也没有接入正式方法。

实现与产物：

```text
scripts/run_sim_rgb_material_multistart.sh
scripts/select_sim_rgb_material_hypothesis.py
outputs/sim_rgb_material_multistart_20260820/rgb_material_selection.md
outputs/sim_rgb_selected_soft_formal_20260820/trajectory_metrics.json
outputs/sim_rgb_selected_soft_formal_20260820/render_metrics.json
```

`py_compile`、shell 语法、2 帧 headless 冒烟、三支完整 300 帧、selected-soft 完整
300 帧组织渲染以及 600 张 PSNR/SSIM/LPIPS 均已通过。RGB 诊断文件记录在每支
`artifacts/rgb_alignment_diagnostics.npz`，元数据明确标记 no depth/no GT。

## SUPER式局部零均值刚度更新（2026-09-01）

在 FoundationStereo RGB 深度 + CoTracker3 轨迹校正的上一轮 C-R 探针上继续开发，数据集、
203 条优化轨迹（其中 200 条有效绑定）、30 个非夹持评估点、7:1 重建和 80/20 未来预测协议
均未改变。上一轮全局 distance+damping 更新虽然改善重建 Mean，却让未来少数点漂移；本轮只
替换 C 组材料更新器：

- 保留视觉状态校正，但材料损失只读取 residual 写入前的 PBD rollout；
- 从最近 5 个已经发生的连续帧做一次开放环重放，只在 `H=1/3/5` 计损失，权重固定为
  `1.5:2:3`；不访问未来帧，不做候选倍率选择、分支状态选择或状态接回；
- 使用 CoTracker 邻接轨迹的 Cauchy 相对形变、绝对分布项和权重 `0.2` 的边应变项；
- distance 由 6 个确定性图平滑区域系数表示。区域系数先映射到粒子 log 场，再按动态粒子
  逆质量精确去均值；固定点和已知夹持控制区 log 偏移恒为零。因此全局几何平均刚度保持
  reset 初值，shape、volume、阻尼和夹持耦合均固定；
- 每满 5 帧才尝试一次更新。Torch 通过完整 distance+volume+shape PBD 给出方向，正式更新
  使用 Warp 中心有限差分梯度；方向余弦必须 `>=0.95`。Warp 局部探针 epsilon 使用 `0.05`
  以越过平滑局部基下的 GPU 重放底噪，但 Adam 学习率/单步上限仍为 `0.03/0.02`；
- 修复 Warp 有限差分结束后没有显式恢复 live distance/shape 数组的问题；失败或被余弦门拒绝
  的探针现在不会把最后一个 FD 扰动泄漏进 live 材料场。

真实 Warp 冒烟覆盖 frame `118..131`、首个夹持窗口。12 区域在 epsilon `0.005/0.05` 下的
方向余弦分别仅 `0.54/0.8766`，说明局部自由度超过当前运动可辨识性；合并为 6 个平滑区域后
余弦为 `0.98647`，超过固定 `0.95` 门并真实提交 1 次。CPU 门同时验证：H1/H3/H5 目标、
加权 log 均值为零、固定/夹持点材料不变、shape/阻尼固定，以及 Warp 梯度实际驱动 Adam；
原全局与旧低维测试继续通过。

实现与入口：

```text
src/embodied_gaussians/physics_simulator/online_tissue_stiffness.py
examples/example_embodied_super_offline.py
src/embodied_gaussians/physics_simulator/sim_benchmark.py
scripts/test_sim_differentiable_local_relative.py
scripts/run_sim_foundation_local_relative_probe.sh
scripts/summarize_sim_local_relative_diagnostics.py
```

完整轨迹探针位于 `outputs/sim_foundation_local_relative_super_v1/`，重建与未来预测退出码
均为 0。30 个非夹持评估点的结果如下（本探针未计算 PSNR/SSIM/LPIPS）：

| 能力 | 方法 | 3D Mean (mm) | 3D Median (mm) | 3D P95 (mm) | 3D Max (mm) | 2D Mean (px) |
|---|---|---:|---:|---:|---:|---:|
| 7:1 重建 | A: PBD | 2.2596 | 0.9975 | 7.9727 | 17.5129 | 19.8904 |
| 7:1 重建 | B: PBD + Foundation/CoTracker 校正 | 0.4778 | 0.2658 | 1.4814 | 6.3274 | 4.6277 |
| 7:1 重建 | C: B + SUPER式局部零均值刚度 | 0.4711 | 0.2411 | 1.5356 | 5.6193 | 4.3051 |
| 80/20 未来 | A: PBD | 5.9556 | 4.8389 | 14.4024 | 17.5183 | 53.9573 |
| 80/20 未来 | B: PBD + Foundation/CoTracker 校正 | 1.0134 | 0.8144 | 3.4690 | 3.8520 | 9.5249 |
| 80/20 未来 | C: B + SUPER式局部零均值刚度 | 1.2411 | 0.9233 | 3.8664 | 8.6179 | 10.1408 |

C 相对 B：重建 3D Mean/Median/Max 分别改善 `1.41%/9.30%/11.19%`，2D Mean 改善
`6.97%`，但 3D P95 变差 `3.66%`；未来 3D Mean/Median/P95/Max 分别变差
`22.46%/13.38%/11.46%/123.73%`，2D Mean 变差 `6.47%`。因此该版本不能作为正式 C 组。

两种协议各有 5 次真实 commit，加权粒子 log 均值误差均小于 `1e-9`，证明局部零均值约束
和材料写入均实际生效。重建最终 distance 为 `0.19048/0.199997/0.20666`，未来训练结束
为 `0.18726/0.20054/0.20988`（min/median/max）。但是事后与隐藏十区域真值比较显示：
重建最终粒子/区域 Spearman 仅 `0.0194/-0.1273`，未来仅 `0.0217/-0.0909`；真实最软/最硬
区域为 `6/7`，两次估计却均为 `0/2`。这说明短时 H1/H3/H5 轨迹损失虽然能降低同化误差，
却没有辨识出正确的材料空间排序；错误的局部软硬补偿在 72 帧开环中被放大。下一版不应继续
放宽局部更新幅度，而应先增加局部材料的可辨识性/持久激励约束，并将多步开环尾部稳定性作为
因果提交门，而不是仅以短时梯度一致性决定 commit。
