# SUPER 原始数据到 PSM 器械位姿与 GUI 接入

> **当前入口提醒（2026-07-25）：** 本文主体记录的是较早的 `corrected`
> 流程。当前默认已经更新为 `depth_then_visual`。第一次阅读请先看
> [`器械矫正流程_小白版.md`](器械矫正流程_小白版.md)；本文继续作为
> corrected 基线、CAD 注册、URDF 远端链和历史复现细节的补充资料。

本文面向第一次接触这套工程的人，说明 `grasp5` 的原始 SUPER 数据如何一步步变成 GUI 中运动的 PSM 器械。

本文记录的是截至 **2026-07-17** 的当前正式路径。PSM 位姿以 `corrected` 驱动为准；`strict`、`registered_lnd`、`paper`、`paper_robust` 和 `hybrid` 仅保留作调试、复现或消融对照。本文取代 `SUPER_COMPLETE_ADAPTATION_GUIDE.md` 中“只用严格 LND、不做图像修正”的旧 PSM 结论，但不取代其中关于组织、桌面和相机坐标系的说明。

## 1. 先说最终答案

现在 GUI 使用的最终位姿文件是：

```text
data/super/psm_tracking/psm_part_corrected_pose_driver.npz
```

它包含：

```text
5458 个机器人时间点
× 7 个可见器械 link
× 7 个位姿数值 [x, y, z, qx, qy, qz, qw]
```

推荐且唯一需要日常使用的 GUI 命令是：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

bash scripts/run_demo_browser_12.sh \
  --psm-pose-driver corrected \
  --psm-visual-mode tip \
  --psm-roll-offset-deg -27
```

其中：

- `corrected`：加载最终矫正后的 driver。
- `tip`：只显示腕部和夹爪，隐藏长杆；不会关闭夹爪的物理接触。
- `-27`：在 GUI 运行时附加器械沿杆轴的自旋修正。它没有写进 `corrected` 文件，也没有修改 `robots.json`。

最重要的一句话是：

> 当前方法不是“只相信图像”，也不是“只相信机器人编码器”，而是用编码器/LND 提供稳定运动主干，用图像提供有限的小修正，再用 URDF 强制夹爪遵守真实父子关节关系。

## 2. 小白需要先懂的五个词

### 2.1 q7

`q7` 是 PSM 每个采样时刻的 7 个关节状态：

```text
outer_yaw
outer_pitch
outer_insertion
outer_roll
outer_wrist_pitch
outer_wrist_yaw
jaw
```

可以把它理解为机器人自己报告的“7 个旋钮现在各转了多少”。`robots.json` 和 `joints.json` 记录的主要就是它。

### 2.2 前向运动学（FK）

只有关节角还不能直接画出器械。前向运动学会把 q7 和器械的长度、关节轴、父子关系组合起来，算出每个零件在三维空间中的位置和朝向。

这里的稳定运动主干来自 SUPER 的 `LND.json`。它使用 Modified-DH 参数描述器械运动学。

### 2.3 位姿（pose）

一个刚体位姿回答两个问题：

- 它在哪里：`x, y, z`，单位米。
- 它朝向哪里：四元数 `qx, qy, qz, qw`。

所以一个 link 的位姿有 7 个数。

### 2.4 link

器械不是一整块不变形的模型，而是多个刚体零件通过关节连接。当前 GUI driver 记录下面 7 个可见 link：

| 序号 | link | 通俗含义 |
|---:|---|---|
| 0 | `PSM1_tool_main_link` | 长器械杆主体 |
| 1 | `PSM1_tool_wrist_link` | 末端腕部的空间锚点 |
| 2 | `PSM1_tool_wrist_shaft_link` | 腕部固定连接段 |
| 3 | `PSM1_tool_wrist_sca_link` | wrist-pitch 之后的零件 |
| 4 | `PSM1_tool_wrist_sca_shaft_link` | wrist-yaw 之后、两夹爪共同父节点 |
| 5 | `PSM1_tool_wrist_sca_ee_link_1` | 第一片夹爪 |
| 6 | `PSM1_tool_wrist_sca_ee_link_2` | 第二片夹爪 |

### 2.5 pose driver

pose driver 是一个已经算好的“逐时刻动作表”。GUI 不需要重新做分割和优化，只要在当前时间读取 7 个 link 的位姿并写进仿真即可。

## 3. 整条流水线鸟瞰

```text
原始 ROS bag、标定、LND、CAD
  │
  ├─ 提取和双目校正
  │    ├─ 左目视频：1441 帧，约 29.77 Hz
  │    └─ q7：5458 个状态，约 99 Hz
  │
  ├─ q7 + LND + hand-eye
  │    └─ 稳定但可能与真实图像略有偏差的运动学先验
  │
  ├─ 论文原始视频跟踪
  │    └─ 1441 帧 paper exact 结果，保留作图像基线
  │
  ├─ body / jaw_left / jaw_right 三部分观测
  │    ├─ 4 帧人工关键帧验证
  │    └─ SurgicalSAM2 传播到 1441 帧，并记录置信度
  │
  ├─ 有界位姿矫正
  │    ├─ LND 运动是主干
  │    ├─ 图像只优化小残差
  │    └─ 时间平滑并保留人工关键帧
  │
  ├─ paper CAD 到 GUI CAD 的一次固定注册
  │
  ├─ 按真实 URDF 重建 wrist / jaw 父子链
  │    └─ 两片夹爪同铰点、同铰轴、对称开合
  │
  ├─ 1441 个视频位姿重采样到 5458 个机器人时间点
  │
  └─ corrected pose driver → GUI 按时间播放
```

## 4. 原始输入分别负责什么

| 输入 | 作用 | 是否直接决定最终位姿 |
|---|---|---|
| `data/grasp5/grasp5.bag` | 左右目图像和 PSM1 关节状态的原始来源 | 是，原始数据源 |
| `data/camera_calibration.yaml` | 双目标定、去畸变和 rectification | 是，决定像素投影 |
| `data/handeye.yaml` | PSM base 到原始左相机的外参 | 是，决定机器人进入相机坐标系的位置 |
| `data/LND.json` | Modified-DH、特征点和器械骨架 | 是，提供稳定运动学主干 |
| `data/dvrk_model/`、`data/super/psm_robot/psm.urdf` | GUI 的 CAD 外形和真实父子关节关系 | 决定显示外形和机械约束 |
| 左目视频 | 观察真实器械轮廓和夹爪尖端 | 只提供有限图像修正 |

这里要避免一个常见误解：`robots.json` 虽然记录了每一帧附近的 q7，但它不是相机中器械的绝对真值。编码器零点、hand-eye 残差、线缆传动、实际工具型号和 CAD 坐标约定都会让理想 FK 与视频有差别。

## 5. 第一步：从 bag 得到视频、时间戳和 q7

原始提取脚本是：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/extract_grasp5_bag.py
```

当前仓库已经有完整产物，不需要为了运行 GUI 再执行这一步。当前使用的是连续成对的 frame `0..1440`：

```text
data/super/grasp5_native/
├── rgb/                         # rectified 左右目 PNG
├── calib_rectified.json         # rectified 相机参数
└── joints.json                  # 5458 个 q7 和时间戳

data/super/grasp5_offline_demo/
├── robots.json                  # DatasetManager 使用的 q7
├── cameras.json                 # 相机入口和外参
└── videos/
    ├── stereo_left.mp4          # 1441 帧真实观测
    ├── stereo_left.json         # 左目 K、分辨率和逐帧时间戳
    ├── stereo_right.mp4
    └── stereo_right.json
```

如果 PNG 已经存在，只想重新整理连续双目帧、视频和 JSON，可运行：

```bash
/Media_HDD/jwshan/conda_envs/eg_codex/bin/python \
  scripts/finalize_grasp5_partial_extract.py
```

### 为什么是 1441 和 5458 两个数字

相机约每秒拍 30 帧，机器人关节约每秒采样 99 次，所以同一段时间内机器人状态更多：

```text
视频时间轴：1441 个点
机器人时间轴：5458 个点
```

图像分割和轮廓优化先在 1441 个视频时刻进行；GUI/物理仿真最后使用 5458 个机器人时刻。因此后面必须按时间戳重采样，绝不能简单用“第 100 张图对应第 100 个 q7”。

## 6. 第二步：用 q7、LND 和 hand-eye 建立稳定先验

### 6.1 先生成 LND 中间结果

```bash
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/build_super_psm_lnd_intermediates.py
```

脚本做三件事：

1. 用 q7 和 `LND.json` 做 Modified-DH 前向运动学。
2. 用 `handeye.yaml` 把 PSM base 放到原始左相机坐标系。
3. 用 rectification 的 `R1` 转到 rectified 左目相机坐标系。

核心坐标链是：

```text
T_rect_camera_psm_base
  = T_rect_camera_raw_camera
  × T_raw_camera_psm_base
```

输出为：

```text
data/super/grasp5_offline_demo/instruments/
├── psm1_lnd_model.json
├── psm1_lnd_motion.json
└── psm1_lnd_generation_report.json
```

### 6.2 把 LND link 映射到当前 URDF 可见 link

```bash
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/calibrate_psm_lnd_pose_driver.py
```

LND link 和 GUI URDF link 名称、局部坐标轴并不相同。这个脚本在 canonical `q=0` 状态下求一次固定映射，然后把 5458 个 LND 状态变成 7 个 GUI 可见 link 的位姿：

```text
data/super/psm_robot/psm_lnd_pose_driver.npz
```

这就是 `strict` 驱动。它很稳定，但没有利用视频逐帧纠正，因此主要作为运动学参考，不是当前最终显示结果。

## 7. 第三步：复现纯论文视频跟踪

纯论文跟踪需要第一帧的前景/背景提示、参考 mask 和两个夹爪尖端：

```text
data/super/psm_tracking/online_videos/grasp5/
├── PSM1_prompts.txt
├── PSM1_ref_mask.png
└── PSM1_keypoints.txt
```

当前文件已经准备好。需要重新标注第一帧时可运行：

```bash
bash scripts/run_super_psm_annotation_browser.sh
```

论文完整跟踪命令为：

```bash
MPLCONFIGDIR=/tmp/matplotlib-online-dvrk \
XDG_CACHE_HOME=/tmp/online-dvrk-cache \
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/track_super_psm_paper_exact.py \
  --checkpoint-every 25
```

主要产物：

```text
tracking_states_paper_exact.npz
├── pure_ctr       (1441, 6)    # 论文 frame-4 的相机位姿
├── pure_joints    (1441, 4)    # wrist 两维和左右半夹爪
├── pure_losses    (1441,)
└── masks_packbits (1441, ...)

psm_paper_exact_pose_driver.npz # 适配后的纯论文 GUI 对照 driver
```

这一路严格保留论文的 SurgicalSAM2、BO 初始化、CMA-ES、ContourTipNet 和 Kalman 设置。它的意义是忠实复现论文方法，而不是保证每帧可靠。

为什么不能直接把它当最终结果：原始 union mask 不区分器械主体和两片夹爪；一片夹爪漏分或 mask 黏到器械杆后，优化器仍会认真拟合错误观测。frame 160、320 及后段已经出现明显漂移和跳变。

## 8. 第四步：把器械拆成三个可识别部分

我们把图像观测拆成：

```text
body
jaw_left
jaw_right
```

先在 frame `141,142,160,320` 做关键帧诊断：

```bash
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/experiment_super_psm_part_segmentation.py \
  --frames 141,142,160,320
```

提示点保存在 `scripts/experiment_super_psm_part_segmentation.py` 的 `FRAME_PROMPTS` 中。每帧输出的 `frameXXXXXX_masks.npz` 包含：

```text
body
jaw_left
jaw_right
original_union
jaw_left_tip_xy
jaw_right_tip_xy
```

下图左列是 RGB，中列是论文原 union mask，右列是三部分结果。橙色是 body，紫色和红色是两片夹爪，黄色点是尖端：

![三部分关键帧分割](data/super/psm_tracking/part_segmentation_experiment/contact_sheet.png)

然后验证这四帧是否真的能改善位姿：

```bash
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/refine_super_psm_part_keyframes.py
```

四个关键帧中，两片夹爪尖端最终误差均小于 `0.78 px`。这说明“三部分观测”有用，之后才传播到全序列。

## 9. 第五步：把三部分 mask 传播到 1441 帧

```bash
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/propagate_super_psm_part_masks.py
```

传播不是完全盲目的：

- frame `141,142,160,320` 注入人工验证过的 mask。
- 每 25 帧用 LND 投影提示保持 body/左右夹爪身份，减少互换。
- 每一部分都保存置信度；低可信观测不会假装成真值。

输出：

```text
data/super/psm_tracking/part_masks_full_sequence/part_masks_full.npz
├── masks_packbits  (1441, 3, ...)
├── confidence      (1441, 3)
├── jaw_tips_xy     (1441, 2, 2)
├── areas_px        (1441, 3)
└── prompt_source   (1441,)
```

这一步仍可能出错。尤其 frame `800` 和 `1120` 的 jaw mask 不可靠，所以后面的优化不能无条件相信分割。

## 10. 第六步：以 LND 为主干做有界图像矫正

完整矫正命令：

```bash
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/track_super_psm_part_corrected.py \
  --output-dir \
  data/super/psm_tracking/part_pose_correction_expanded_3mm_8deg_12deg_25deg
```

每帧不是从随机位置开始，而是先取该视频时间戳附近的 q7，通过 LND 得到稳定先验。优化器只允许 9 个小残差：

```text
整体旋转      3 个数
整体平移      3 个数
wrist 残差    2 个数
共同 jaw 残差 1 个数
合计          9 个数
```

当前上限为：

| 修正项 | 最大值 |
|---|---:|
| 整体平移 | `3 mm` |
| 整体旋转 | `8°` |
| wrist | `12°` |
| jaw | `25°` |

这些限制的作用类似“护栏”：图像可信时允许模型靠近轮廓；mask 错误时，不让单帧观测把器械拉飞。优化后还会做时序平滑，并把四个人工关键帧作为固定 anchor 保留下来。

当前 1441 帧全部得到更低的优化目标。loss P50 从 `1.2764` 降到 `0.3228`；实际平移修正 P50/P95 为 `0.874/1.120 mm`，整体旋转修正 P50/P95 为 `2.820/4.124°`。

下图每行是一帧，从左到右依次是三部分观测、纯论文结果、registered LND prior、最终图像矫正结果：

![纯论文、LND 先验和最终矫正对比](data/super/psm_tracking/part_pose_correction_expanded_3mm_8deg_12deg_25deg/contact_sheet.png)

主要状态文件为：

```text
tracking_states_part_corrected.npz
├── prior_ctr / prior_joints
├── raw_corrected_ctr / raw_corrected_joints
├── corrected_ctr     (1441, 6)
├── corrected_joints  (1441, 4)
├── raw_correction_parameters
├── smoothed_correction_parameters
└── confidence / losses / prompt_source
```

纯论文结果不会被覆盖；当前矫正会生成一套新的资产。

## 11. 第七步：把论文状态变成 GUI 能用的机械结构

这是最容易被忽略、但对夹爪最重要的一步。

### 11.1 为什么论文位姿不能直接塞进 GUI

论文 renderer 的 CAD、局部坐标轴和当前 GUI 的 URDF/Gaussian 不是同一套定义。即使论文投影正确，把每个论文零件的绝对位姿分别套到 GUI 零件上，也可能出现：

- 两片夹爪的圆形铰链中心不在一起。
- 两个夹爪铰轴不平行。
- 夹爪看似闭合，但前后深度相差很大。
- 夹爪开合平面与器械杆不一致。

因此适配不是简单改文件名，而是下面两层处理。

### 11.2 一次固定的 paper-CAD 到 GUI-CAD 注册

使用当前 tip Gaussian 与 registered paper CAD 做一次固定 SE(3) 拟合。当前得到约：

```text
平移 [-1.278, +6.734, -9.346] mm
旋转 3.518°
```

这个变换整段视频只求一次。`registered_lnd` 和 `corrected` 共用它，避免把每帧运动误差和 CAD 固定坐标差混在一起。

### 11.3 用真实 URDF 重建完整远端链

当前以已经跟踪好的 `tool_wrist_link` 作为空间锚点，然后完全按照 `psm.urdf` 重建它后面的 link：

```text
tool_wrist_link                 # 跟踪锚点
  └─ tool_wrist_shaft_link      # fixed
      └─ wrist-pitch
          └─ wrist-yaw
              ├─ jaw 1: +jaw/2
              └─ jaw 2: -jaw/2
```

最终角度来源是：

- wrist-pitch `q4`：严格使用真实 encoder。
- wrist-yaw `q5`：encoder 加图像估计的 q5 残差。
- jaw：encoder jaw 加图像估计的共同开合残差。

没有把论文的绝对 wrist-pitch 直接送入 URDF。原因是论文第一帧 wrist-pitch 约 `-29.09°`，同帧 encoder/URDF 约 `+1.64°`；约 `30.7°` 的静态坐标零点差已被固定注册吸收，再加一次就会把夹爪平面扭离器械杆。

两片夹爪最终严格共享：

```text
同一个 parent
同一个 origin = (0, 0, 0)
同一条 axis = (0, 0, 1)
对称角度 +jaw/2 和 -jaw/2
```

稳定闭合段 frame `548..1246` 的共同铰点误差从旧方法约 `5.006 mm` 降到数值 `0 mm`；铰轴不平行误差 P50 约 `1.9e-6°`。杆线离开夹爪平面的全序列 P50/P95 从 `16.28/19.36°` 降到 `1.87/2.19°`。

下图上排是旧的独立夹爪适配，下排是当前共享 URDF 父子链：

![夹爪共同铰点和 URDF 远端链验证](data/super/psm_tracking/urdf_distal_chain_validation/jaw_shared_pivot_comparison.png)

## 12. 第八步：从 1441 帧重采样为 GUI 的 5458 个状态

视频优化得到的是 1441 个时刻，GUI 的机器人时间轴有 5458 个时刻。`save_runtime_driver()` 会按真实时间戳重采样：

- 平移：线性插值。
- 旋转：四元数 SLERP。

最终写出：

```text
data/super/psm_tracking/psm_registered_lnd_pose_driver.npz
data/super/psm_tracking/psm_part_corrected_pose_driver.npz
```

二者格式相同：

```text
timestamps                         (5458,)
link_names                         (7,)
poses_rect_camera_xyz_xyzw         (5458, 7, 7)
```

如果已经有 `tracking_states_part_corrected.npz`，只修改了 CAD 注册或 URDF 适配代码，不想重跑分割和优化，可快速重建 driver：

```bash
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/rebuild_super_psm_registered_drivers.py \
  --states \
  data/super/psm_tracking/part_pose_correction_expanded_3mm_8deg_12deg_25deg/tracking_states_part_corrected.npz \
  --output-dir \
  data/super/psm_tracking/part_pose_correction_expanded_3mm_8deg_12deg_25deg \
  --registered-driver \
  data/super/psm_tracking/psm_registered_lnd_pose_driver.npz \
  --corrected-driver \
  data/super/psm_tracking/psm_part_corrected_pose_driver.npz
```

## 13. 第九步：GUI 到底怎样消费这个文件

GUI 入口是：

```text
examples/example_embodied_super_offline.py
```

环境实现是：

```text
examples/embodied_environments/super_embodied/super_embodied.py
```

`--psm-pose-driver corrected` 会映射到：

```text
data/super/psm_tracking/psm_part_corrected_pose_driver.npz
```

运行时按下面的顺序工作：

1. 加载 driver 中 rectified 左目相机坐标系下的 7 个 link 位姿。
2. 用统一的 `X_table_camera` 把它们变换到 GUI/table 世界坐标系。
3. 当前视频时间变化时，在 5458 个时间戳中取对应机器人状态。
4. 把所选 7 个 link 位姿同时写入 Warp 的 `state_0` 和 `state_1`。
5. 每次 physics step 后重新施加，防止物理解算把外部驱动的器械拉离记录位姿。
6. surface Gaussian 绑定在各自 link 的局部坐标中，因此 link 一动，器械外观一起动。

GUI 并不会重新运行 SAM、CMA-ES 或位姿优化。它只是播放器。

### 为什么 GUI 仍然读取 robots.json

即使已经有 corrected driver，q7 仍有三项用途：

- 维持完整 Warp articulation 和 mimic 关节状态。
- 提供物理器械/碰撞相关的机器人时间状态。
- 在 selected driver 上按真实 URDF 关节轴叠加手动 roll 和 jaw 增量。

但 q7 不会再通过 strict FK 覆盖 corrected 的 7 个可见 link 位姿。

## 14. GUI 中还能手动改什么

`SUPER Playback` 面板目前提供：

- `PSM self-spin offset deg`：沿器械杆轴自旋，范围 `[-180°, 180°]`，默认 `-27°`。
- `PSM jaw opening offset deg (+open)`：额外开合，范围 `[-30°, 30°]`，左右各转一半。
- `Manual image X/Y/Z mm`：沿相机方向做临时整体平移。

这些量都叠加在 selected driver 上，不会覆盖或改写 `psm_part_corrected_pose_driver.npz`。按 Reset 后，自旋回到命令行默认值 `-27°`，jaw 回到 `0°`。

## 15. 用投影图确认“GUI 实际会看到什么”

下图使用最终 corrected driver、完整 URDF 远端链和 GUI 默认 `-27°` 自旋，覆盖 frame `0,480,560,800,1120,1280,1440`：

![最终 corrected GUI 等价投影](data/super/psm_tracking/gui_projection_corrected_urdf_chain_roll_minus27/contact_sheet_lnd_dvrk_zoom.png)

看图方法：

- 青色点：当前 GUI 的 tip surface Gaussian 投影，也是最应该关注的显示结果。
- 黄色长线和彩色点/线：原始 LND skeleton 与特征，用来说明编码器运动学先验在哪里。
- 背景：同一时间戳的真实左目视频。

它由下面的离线命令生成，使用与 GUI 相同的手动 offset 实现：

```bash
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/overlay_super_lnd_on_png.py \
  --pose-driver data/super/psm_tracking/psm_part_corrected_pose_driver.npz \
  --frames 0 480 560 800 1120 1280 1440 \
  --tip-only \
  --roll-offset-deg -27 \
  --surface-label "FINAL: CORRECTED ANCHOR + URDF DISTAL CHAIN + ROLL -27deg" \
  --out-dir \
  data/super/psm_tracking/gui_projection_corrected_urdf_chain_roll_minus27
```

## 16. 从现有原始资产完整重跑的命令顺序

大多数时候只需运行 GUI，不要重复下面这些耗时步骤。需要从已经提取好的 SUPER 视频/q7 重新生成 corrected driver 时，顺序如下：

```bash
cd /Media_HDD/jwshan/wad/Embodied_gaussians_fixed_super_best

# 1. LND 稳定运动学中间结果
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/build_super_psm_lnd_intermediates.py

# 2. strict LND GUI 对照 driver
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/calibrate_psm_lnd_pose_driver.py

# 3. 纯论文 1441 帧结果
MPLCONFIGDIR=/tmp/matplotlib-online-dvrk \
XDG_CACHE_HOME=/tmp/online-dvrk-cache \
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/track_super_psm_paper_exact.py \
  --checkpoint-every 25

# 4. 四个三部分人工关键帧
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/experiment_super_psm_part_segmentation.py \
  --frames 141,142,160,320

# 5. 四个关键帧的小范围位姿验证
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/refine_super_psm_part_keyframes.py

# 6. body / jaw_left / jaw_right 传播到 1441 帧
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/propagate_super_psm_part_masks.py

# 7. 完整有界矫正，并生成最终两个 GUI driver
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/track_super_psm_part_corrected.py \
  --output-dir \
  data/super/psm_tracking/part_pose_correction_expanded_3mm_8deg_12deg_25deg

# 8. 夹爪和完整远端 URDF 链的数值验证
MPLCONFIGDIR=/tmp/matplotlib-online-dvrk \
XDG_CACHE_HOME=/tmp/online-dvrk-cache \
/Media_HDD/jwshan/conda_envs/online_dvrk/bin/python \
  scripts/validate_super_psm_gui_jaws.py \
  --states \
  data/super/psm_tracking/part_pose_correction_expanded_3mm_8deg_12deg_25deg/tracking_states_part_corrected.npz \
  --registration \
  data/super/psm_tracking/part_pose_correction_expanded_3mm_8deg_12deg_25deg/paper_to_gui_registration.npz \
  --fixed-poses \
  data/super/psm_tracking/part_pose_correction_expanded_3mm_8deg_12deg_25deg/visual_poses_part_corrected.npz \
  --output-dir \
  data/super/psm_tracking/urdf_distal_chain_validation \
  --frames 0,480,560,800,1120,1280,1440
```

然后使用 GUI 环境播放：

```bash
bash scripts/run_demo_browser_12.sh \
  --psm-pose-driver corrected \
  --psm-visual-mode tip \
  --psm-roll-offset-deg -27
```

## 17. 常见问题

### 17.1 `robots.json` 已经有每帧状态，为什么还要图像修正

`robots.json` 记录的是编码器读数，不是相机中的绝对三维真值。它不知道 hand-eye 残差、工具安装零点、线缆回差、真实 CAD 型号差异和图像采集延迟。它非常适合做稳定主干，但不能单独消除所有投影偏差。

### 17.2 夹爪尖端识别错了会不会影响位姿

会。因此当前方法做了三层保护：低置信度降权、修正幅度有上限、最终残差做时间平滑。错误观测可能留下局部偏差，但不应把整段器械拉飞。frame `800` 和 `1120` 仍应视为需要补人工 anchor 的已知弱点。

### 17.3 为什么不用纯论文结果

纯论文方法主要依赖图像 mask。一旦 union mask 漏掉一片夹爪或把器械杆并入夹爪，优化器就会拟合错误目标。它适合做论文复现基线，不适合直接作为这段视频的物理接触真值。

### 17.4 为什么 URDF 不能直接解决所有对齐问题

URDF 能保证“怎样动才符合机械结构”，例如共同铰点、平行铰轴和对称开合；但它不能自动知道 hand-eye 是否偏了 2 mm、工具自旋零点是否偏了 27°，也不能保证当前 STL 与真实器械型号完全相同。

### 17.5 `corrected` 是不是把 LND、论文和 GUI 三种驱动混在一起播放

不是。它们只在离线生成阶段各司其职：LND 提供主干、图像提供残差、URDF提供机械约束。最终 GUI 只加载一份 `corrected` driver。

### 17.6 为什么投影图很好，GUI 以前却有偏差

旧 GUI 曾有两类适配问题：paper CAD 与 GUI CAD 的固定坐标差没有统一处理，以及 GUI 手动 offset/q7 路径可能重新计算 strict FK 覆盖 selected driver。现在 GUI 直接从 selected driver 开始叠加 offset，并在每个 physics step 后重新写回，因此两条路径已经统一。

## 18. 当前仍未解决的事情

- frame `800`、`1120` 的 jaw mask 仍不可靠，需要补人工 anchor，而不是继续无限放宽修正范围。
- 默认 `-27°` 自旋来自当前序列的目视校正，还不是独立标定板得到的工具零点。
- 当前两片 jaw visual mesh 使用同一个 STL，通过旋转镜像得到；若实物左右齿形或工具型号不同，姿态正确时轮廓仍可能不同。
- 每片夹爪目前只有约 64 个 surface Gaussian，显示会略粗、略圆。
- 自动门禁和抽样投影已经通过，但完整 1441 帧仍需要人工从头到尾验收。

## 19. 最后用一句话记忆

```text
q7/LND 保证“别乱跑”
图像分割保证“尽量贴视频”
URDF 保证“关节动作像真的”
pose driver 负责“把最终答案交给 GUI 播放”
```
