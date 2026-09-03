# SUPER PSM 原始数据运动学主干

这条新路径只解决器械位姿的第一层：从根数据建立稳定、可复现的 LND 运动学
主干。它暂时不接 CAD/URDF 注册、图像矫正、双目深度矫正、手工偏移、table
坐标或 GUI。

## 为什么重新建立

历史器械链已经经历了多代产物：

- bag 被部分提取成左目 1441 张、右目 1443 张 PNG，正式回放又截成 1441 对；
- q7/LND 先生成 strict driver；
- 后续又叠加 registered LND、三部件图像 corrected、FoundationStereo 深度修正、
  轮廓修正和运行时 `-27°` 自旋；
- 多个 pose driver 并存，难以证明某一中间结果没有读取更早的派生产物。

原始 bag 实际包含左右目各 1644 条图像消息和 5458 条 q7，时长约 55 秒。因此
本路径不把历史 1441 帧作为新的数据边界。

## 唯一输入白名单

```text
data/grasp5/grasp5.bag
data/camera_calibration.yaml
data/handeye.yaml
data/LND.json
```

脚本会主动拒绝任何位于 `data/super/` 下的输入。相机标定、hand-eye 和 LND
还会复制到输出目录，作为本次构建的只读快照。

## 坐标链

所有三维长度统一导出为米，所有角度为弧度，变换采用
`T_A_B @ p_B = p_A`。

```text
q7(t)
  -> LND Craig Modified-DH
  -> T_PSM_base_LND_link(t)

T_rect_left_PSM_base
  = T_rect_left_raw_left
  @ T_raw_left_PSM_base

T_rect_right_PSM_base
  = T_rect_right_raw_right
  @ T_raw_right_raw_left
  @ T_raw_left_PSM_base
```

LND link 0 是 PSM base；1..6 是六个串联 DH link；7 和 8 是 link 6 的并联
子节点，分别旋转 `+jaw/2` 和 `-jaw/2`。

## 构建

```bash
python scripts/build_super_raw_psm_kinematics.py
```

如果需要同时为 29 GB bag 计算完整 SHA256：

```bash
python scripts/build_super_raw_psm_kinematics.py --hash-bag
```

输出目录：

```text
data/super/psm_raw_kinematics_v1（纯机器人学版本）/
├── model.json
├── kinematics.npz
├── report.json
├── root_inputs_snapshot/
└── validation/
```

`kinematics.npz` 以原生 q7 时间轴保存 PSM base、rectified 左相机和 rectified
右相机中的 9 个 LND link 变换。左右目各自保留绝对 ROS 时间戳，同时显式导出
nearest-q 与因果 zero-order-hold 索引；双目配对用真实时间戳做单调匹配并要求
时间差不超过 20 ms，不再假设左右同下标。主干本身不把图像重采样成 q7，也不
把 q7 插值成图像帧。

`validation/` 的首、中、末帧图像直接从 bag 解码，再用根标定做 rectification
和 LND 投影。它们只用于验证坐标链，不参与求解。

## 2026-07-25 本次构建结果

- 原始 bag SHA256：
  `1ec637d3037fd704778d647dee3fc85cf21131a4636d2e69eaf64bb726e26bd6`；
- 左目 `1644` 条、右目 `1644` 条、q7 `5458` 条；
- 在 `|dt| <= 20 ms` 门限下得到 `1631` 对；左右各 `13` 条保留为未配对消息；
- 双目对时间差 P50/P95/max 为 `10.589/16.286/19.956 ms`；
- 图像到最近 q7 的时间差 P95：左目 `4.784 ms`，右目 `4.780 ms`；
- rectified 右相机到左相机是纯 `-5.316137 mm` x 平移，数值误差
  `2.69e-15`；
- 全部 `5458 x 9` 个 link 变换 finite，旋转正交/行列式误差约 `1e-15`；
- link 单步平移最大 `0.204 mm`，串联 link 单步旋转最大 `2.526°`；
- jaw 快速动作的半夹爪单步旋转最大 `13.207°`。该段原始 jaw 速度为
  `-34.8/-43.0 rad/s`，全序列 jaw 有限差分与报告速度相关系数 `0.992`，因此
  保留为真实快速动作，不做隐式滤波；
- 首、中、末帧左右目各 12/12 个 LND point feature 均在画面内，人工查看投影
  跟随真实器械。

## GUI 坐标与 CAD 接入

主干输出的 link 是 LND 数学坐标系。GUI 接入另用一个隔离的构建步骤，从根
`data/dvrk_model` xacro/mesh 重新生成 URDF，并在 `q7=0` 时求一次固定
LND-to-CAD 注册。它不会读取历史 `psm.urdf`、surface Gaussian、注册报告或
任何 strict/registered/corrected/depth pose driver。

GUI 使用现有 `table_frame.json` 定义的 dense-ground table/world 坐标，完整
变换链为：

```text
T_GUIworld_CADlink(t)
  = X_GUIworld_rectifiedLeftCamera
  @ T_rectifiedLeftCamera_LNDlink(t)       # 原始主干
  @ T_LNDlink_CADlink                      # 根模型 q7=0 固定注册
```

`cameras.json` 的 Blender 相机坐标还会作为硬门限检查：

```text
X_WC_blender
  = X_GUIworld_rectifiedLeftCamera
  @ diag(1, -1, -1, 1)
```

构建 GUI 资产：

```bash
PYTHONPATH=/home/jwshan/miniconda3/envs/endo3r/lib/python3.11/site-packages \
python scripts/build_super_raw_psm_gui_driver.py

python scripts/build_psm_surface_gaussians.py \
  --urdf "$PWD/data/super/psm_raw_kinematics_v1（纯机器人学版本）/gui_v1/psm_raw.urdf" \
  --output "$PWD/data/super/psm_raw_kinematics_v1（纯机器人学版本）/gui_v1/psm_raw_surface_gaussians.npz" \
  --report "$PWD/data/super/psm_raw_kinematics_v1（纯机器人学版本）/gui_v1/psm_raw_surface_gaussians_report.json"

python scripts/validate_super_raw_psm_gui_driver.py
```

输出保存在同一纯机器人学目录的 `gui_v1/`，包括 fresh URDF/mesh、mimic map、
5458 状态的 GUI-world pose driver、1508 个 surface Gaussian、注册报告和验证
报告。

按当前 GUI 的原有配置启动（组织、接触、相机、器械显示范围等均保持
`run_demo_thinlinc.sh` 原设置）：

```bash
bash scripts/run_demo_thinlinc.sh --visual-force-iterations 1
```

该脚本现在默认选择 `raw_kinematics` 且 `roll=0°`。GUI 视频仍使用当前离线
数据集的时间范围；器械状态则按 raw 主干自己的 5458 个 q7 时间戳做
zero-order-hold，不再读取旧 `robots.json` 作为位姿源。需要临时切换位姿版本
时，在命令末尾覆盖即可：

```bash
bash scripts/run_demo_thinlinc.sh \
  --visual-force-iterations 1 \
  --psm-pose-driver raw_kinematics \
  --psm-roll-offset-deg 0
```

把 `raw_kinematics` 换成 `strict`、`registered_lnd`、`paper`、
`paper_robust`、`hybrid`、`corrected` 或 `depth_then_visual` 即可切换版本；
历史视觉修正版若需要原先的目视自旋，同时将最后一项改成 `-27`。

## 2026-07-25 GUI 接入验证

- 根 dVRK/LND canonical 注册最大残差 `0.794 µm`；
- 当前 GUI 左相机矩阵与 `cameras.json` 最大误差 `0`；
- 抽查全序列 17 个时刻，坐标链最大平移误差 `1.65e-8 m`、最大旋转误差
  `4.49e-6°`；
- runtime LND FK 与保存的 GUI pose driver 达到相同误差；
- raw 时间戳和 q7 与主干逐元素完全一致；
- CPU 环境构建成功加载 fresh `psm_raw.urdf`、7 个驱动 CAD link 和 1508 个
  Gaussian，并在首/中/末状态逐项验证写入的 `body_q`。
