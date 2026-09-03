# 当前 SUPER 组织—器械—双目视觉闭环框架

本文更新于 2026-08-14，只描述当前代码和 GUI 默认入口实际使用的方案。历史实验、已回滚
参数和中间结论见 `PROGRESS.md`，不再混入这里。本文中的“视觉”均指左右目 **2D RGB
图像**；物理节点仍在三维空间运动，但没有把双目图像先重建成在线 3D 点云，也没有使用
3D Chamfer residual。

本文供外部审阅当前实时更新实现。为避免把不同变量混为一谈，下文明确区分两条链路：
双目 residual 更新的是物理节点位置状态 `particle_q`；在线材料更新只修改逐节点的
distance/shape 刚度。两者不是同一个优化变量，也不是一次多帧可微 XPBD 参数辨识。
视觉 residual 与在线刚度是受 Liang et al.
[arXiv:2309.11656](https://arxiv.org/abs/2309.11656) 启发的初步适配，代码已经接入 GUI，
但尚无完整轨迹的实时闭环验证结果。

## 1. 当前资产、分层与数据流

默认组织模式为 `paper_pbd`，资产是：

```text
data/super/grasp5_native/tissue_multiview_v1/
paper_pbd_tissue_v15_denser_mild_paper_constraints_centroid_ellipsoids/
tissue_paper_pbd_centroid_gaussians.npz
```

当前三层分辨率彼此独立：

| 层 | 当前规模 | 作用 |
|---|---:|---|
| 四面体物理节点 | 1,465 | 质量、速度、外力和位置残差自由度 |
| 四面体 | 5,502 | 距离、体积、形状匹配与有限体积传播 |
| 闭合碰撞外皮 | 2,116 三角面、1,060 节点 | 夹爪—组织接触 |
| 高分辨率视觉网格 | 13,583 顶点、26,754 面 | 保留双目/多帧表面细节 |
| 组织 Gaussian | 26,754 | 每个视觉面一个椭球 |
| 器械 Gaussian | 76,798 | 四个器械 link 的外观渲染 |

因此“高分辨率统一接触表面”不是把 26,754 个视觉面都变成碰撞单元。当前接触仍在
2,116 面的物理边界上进行，高分辨率视觉面只负责渲染和图像梯度。视觉面数量与接触
几何分辨率没有直接联动；后者由物理表面节点、碰撞面和四面体拓扑决定。

总体数据流为：

```text
离线器械 SE(3)+q7 ──> 运动学夹爪 ──> 三角外皮接触 ──> 四面体 XPBD
                                              │
                                              └──> 双侧 u_t 抓取与局部抬升

物理节点 ──> 嵌入的高分辨率视觉面 ──> 面中心 Gaussian ──> 左/右目 2D 渲染
   ▲                                                        │
   ├──── accepted 位置 residual r_i ────────────────────────┘
   │                  │
   │                  └──> 候选 k_dist / k_shape ──> 影子 rollout 验证 ──> 正式 XPBD
   └──── 视觉力（与 residual 互斥的回退模式） ───────────────┘
```

## 2. 组织几何、质量与视觉嵌入

### 2.1 静止形状与体网格

当前 rest shape 是视频中已受支撑和重力影响的观测平衡形状，而不是恢复出的无载荷形状。
为避免把重力重复施加到已下垂的几何上，运行时取：

```text
gravity                         0 m/s²
density                         1000 kg/m³
velocity damping               10 s⁻¹
total tissue mass              about 41.57 g
fixed support particles        131
```

四面体质量按静止体积分到节点：

\[
m_i=\sum_{e\ni i}\frac{\rho V_e^0}{4}.
\]

固定节点逆质量为零。GUI 中约 `0.45 mm` 的粒子球只是显示标记，物理粒子碰撞半径为零；
真正的器械接触几何是闭合三角外皮。

v15 对常用抓取区作温和且连续的体网格加密：

```text
center, r <= 20 mm             3.25 mm target spacing
transition, 20 < r <= 45 mm    3.25 -> 3.75 mm
outer surface                  3.75 mm
background                     4.75 mm
deep volume                    5.50 mm
minimum rest tet volume        0.1168 mm³
maximum rest-matrix condition  117.29
```

拓扑在运行时固定，不自动重网格化。物理网格密度决定夹爪足迹和局部弯曲的空间分辨率，
接触位移幅度、接触方向和体积约束尺度则由其他独立模块决定。

### 2.2 高分辨率视觉面与 Gaussian

每个视觉顶点嵌入最近物理表面三角形 \((a,b,c)\)。设重心权重为
\(\beta_1,\beta_2,\beta_3\)，静止偏移为 \(d^0\)，物理面的静止/当前局部架为
\(R_f^0,R_f(t)\)，则

\[
v(t)=\sum_{k=1}^{3}\beta_kx_{i_k}(t)+R_f(t)(R_f^0)^Td^0,
\qquad \sum_k\beta_k=1.
\]

视觉三角面 \(g=(v_1,v_2,v_3)\) 的 Gaussian 中心为

\[
\mu_g(t)=\frac{v_1(t)+v_2(t)+v_3(t)}{3}.
\]

Gaussian 的 RGB、opacity、静止四元数和三轴尺度来自重建外观；静止尺度约
`0.175–0.700 mm`，各向异性中位数约 `1.91`。运行时不再把尺度固定：视觉三角面
发生切向拉伸或剪切后，其完整静止协方差随三角面形变传输，并重新分解为当前
quaternion 和三轴 scale。厚度方向使用单位法向，避免把面面积变化错误地变成厚度伸缩。

## 3. 当前 `paper_pbd` 组织模型

### 3.1 三类约束

物理网格含 8,024 条唯一四面体边。边 \((i,j)\) 的距离约束是

\[
C_{ij}^{dist}=\|x_i-x_j\|-L_{ij}^{0}.
\]

当前源码执行的有符号体积约束是

\[
V_e(x)=\frac{((x_1-x_0)\times(x_2-x_0))\cdot(x_3-x_0)}{6},
\qquad C_e^{vol}=V_e(x)-V_e^0.
\]

Liang et al. 的正文明确给出 XPBD 更新、distance/volumetric/shape-matching 三类约束、
参与粒子刚度平均，以及固定 \(k_{vol}=10^{10}\)，但没有进一步给出 volumetric
constraint 的归一化公式。本项目当前没有实现此前设想的
\(V_e/V_e^0-1\) 无量纲主约束，运行代码就是上述绝对体积形式。

形状匹配对每个四面体单独计算

\[
F=D_sD_m^{-1},\qquad R=\operatorname{polar}(F),
\]

其中反射分量会被移除。令 \(c=\tfrac14\sum_i x_i\)、
\(q_i=x_i^0-c^0\)，标量约束为

\[
C_e^{shape}=\sqrt{\sum_{i=0}^{3}\|x_i-(c+Rq_i)\|^2}.
\]

静止态另有相对 `1e-4` 的位置死区，避免 float32 SVD 噪声累计。

三类约束都采用 XPBD 更新：

\[
\Delta\lambda=
\frac{-C-\widetilde\alpha\lambda}
{\sum_i w_i\|\nabla_iC\|^2+\widetilde\alpha},
\qquad
\widetilde\alpha=\frac{s_{comp}}{k_c\Delta t_{sub}^2},
\qquad
\Delta x_i=w_i\Delta\lambda\nabla_iC.
\]

距离刚度取边两端节点值的平均，形状刚度取四个节点值的平均；体积刚度仍是全局标量。

### 3.2 当前参数

```text
k_dist                         0.20
k_volume                       1e10
k_shape                        0.004
material iterations            6 per substep
material relaxation            1.0
material compliance scale      1.0
material minimum volume ratio  0.01
generic XPBD iterations        3
material/contact velocity gain 0.14 / 0.35
```

`0.20/1e10/0.004` 是针对“夹取位移传播到工具后方 10 mm 以上”切换的明显软档：材料传播
由每子步 8 次降为 6 次；volume 保持固定，避免用局部塌缩伪装柔软。它仍是 Reset 后的
GUI 观察初值，不是已经由真实材料标定得到的参数。在线 distance/shape 下限现为
`0.10/0.003`，因此 Reset 初值分别保留 50%/25% 的向下修正空间；distance 上限扩到 `2.00`，
允许不同区域最终形成最多 20 倍的有界数值跨度。
`paper_soft` 的 `E=50 Pa, nu=0.35` 只保留作旧 Neo-Hookean 回归，不是当前主模型。

物理状态和材料参数在运行时的存储/更新关系如下：

| 量 | 运行时表示 | Reset 后初值 | 双目 residual 模式是否更新 |
|---|---|---:|---|
| 节点位置 | `particle_q[1465,3]` | 首帧物理状态 | 是；动态节点加 accepted residual，固定节点增量为零 |
| 节点速度 | `particle_qd[1465,3]` | 首帧物理状态 | 否 |
| distance 刚度 | `k_dist[i]`，逐节点数组 | 全部 `0.20` | 是；有界 log-space 更新 |
| volume 刚度 | 单个全局标量 | `1e10` | 否 |
| shape 刚度 | `k_shape[i]`，逐节点数组 | 全部 `0.004` | 是；有界 log-space 更新 |
| 密度、质量、damping | 固定标量/数组 | 第 2 节所列数值 | 否 |
| rest 节点、边长、静止体积 | 固定数组 | 资产值 | 否 |
| 接触/抓取参数 | 全局配置 | 第 4 节所列数值 | 否 |
| Gaussian RGB/opacity | 固定外观数组 | 重建资产值 | 否 |
| Gaussian quaternion/scale | 视觉三角面形变协方差 | 重建资产值 | 是；随绑定面变化，不是独立优化变量 |
| 左右目内外参 | 每帧观测相机数组 | 离线数据值 | 否 |
| 器械 SE(3)、q7 | 时间戳对齐离线轨迹 | 当前视频时间戳 | 否；不接收 RGB residual |

下一次 XPBD 距离约束实际读取
\(k_{ij}^{dist}=(k_i^{dist}+k_j^{dist})/2\)，shape matching 读取
\(k_e^{shape}=\sum_{i\in e}k_i^{shape}/4\)。因此在线更新保存的是节点参数，约束自身没有
单独的可学习参数副本。

候选材料运动低于 `J=V/V0=0.01` 时，求解器先局部保留单元平均平移、移除危险相对
形变，再回退受影响的一环节点；最后用 `J>=1e-8` 作绝对不翻转兜底。这个局部修复不等于
材料模型已经能从任意深度穿透中恢复高质量单元。

## 4. 当前器械接触与抓取

### 4.1 接触表示和下压

器械由离线 SE(3) 轨迹和原始 q7 运动学驱动，组织反力不会反向改变器械轨迹。两个夹爪
表面以 `0.65 mm` 间距生成 7,579 个采样点；shaft 不参与组织接触。每个采样点在
`10 mm` 内查询最近组织三角面，普通接触间隙为 `0.40 mm`，切向摩擦系数为 `1.5`。

最近面重心权重为 \(b_k\)、节点逆质量为 \(w_k\) 时，单个接触修正按

\[
D=\sum_{k=1}^{3}w_kb_k^2,\qquad
\Delta x_k=\frac{w_kb_k}{D}\Delta x_{contact}
\]

分给命中面的三个物理节点，并经过局部不翻转 line search。

当前普通 jaw 接触只保留世界 `xy` 平面内的法向/切向分量，竖直分量严格为零。原因是夹爪
进入重建顶面以后，闭合外皮的最近面查询可能命中组织底面；若继续沿底面外法向投影，
下行器械会把底面反向向上顶起。现在竖直职责被明确分开：抓取前只由 oriented top
barrier 产生向下位移，捕获后只由四点 `u_t` 和共同腕部 support 产生三维抬升。

当前“压下不过狠、尖端允许少量进入”的配置为：

```text
ordinary jaw correction cap          0.030 mm/pass
oriented top-barrier cap             0.008 mm/pass
contact solve                         1 pass every physics substep
contact spread                        0 tet layers
post-contact material iterations      0
final extra barrier                   OFF
jaw distal contact/grip band          5.0 mm
tip entry allowance                   3.0 mm
top-barrier active distal band        3.00 .. 3.01 mm
top-barrier lateral tolerance         0.30 mm
direct downward particle core         2.50 mm
top-barrier clearance                 0.25 mm
contact minimum volume ratio          0.03
```

尖端前 `3.0 mm` 不受 oriented top barrier 阻挡；其后的 `0.01 mm` 保留最窄的弱法向屏障；
后部 `1.99 mm` 仍可提供 q7 切向接触和抓取候选，但不主动把尚未夹住的邻域压下。这里的
`3.01 mm` 隐藏 barrier 末端是为了满足 `tip < barrier <= 5 mm jaw band` 的启动校验；人工 support column、
邻域下压和 pressure shoulder 都为零。direct downward core 现在围绕每个实际命中点只选择
切平面距离不超过约 `2.5 mm` 的物理面节点；如果粗三角形没有节点落入 core，只保留最近一个，
不会恢复整张三角形的向下位移。双侧接触中心可用后，同一 `2.5 mm` 半径限制完整 direct jaw
patch。范围外仍允许四面体 XPBD 的自然响应，但不再收到显式 contact delta。材料投影速度
回灌当前按用户调参设为 `0.14`，几何接触速度仍为 `0.35`。相比此前 `0.05`，材料恢复运动
进入下一子步更多、跟随更连续，但抑制乱飘和长距离惯性尾巴的能力会相应减弱。

这项改动只属于接触局部性和速度重建：`k_dist/k_volume/k_shape`、材料 6 passes、在线 bounds、
EMA、candidate/verified 与开放环验收均未改变。

### 4.2 双侧 `u_t` 抓取

非粘附接触只能推，不能在夹爪离开表面后带起组织。当前 persistent grip 因此把双侧
接触转换成四个显式位置控制：每个夹爪恰好选择两个健康的真实接触表面节点。捕获时将
节点保存到各自夹爪局部坐标，之后目标为

\[
u_a(t)=T_{jaw(a)}(t)\,T_{jaw(a)}(t_c)^{-1}x_a(t_c).
\]

所以继续闭合 q7 时左右两组 \(u_t\) 可以相对运动，而不是跟随一个虚拟中点刚体。

捕获门限为：

```text
q7 closed threshold                    <= 0.13 rad
minimum contact samples                8 per jaw
maximum jaw-patch separation           6 mm
required sustained solves              3
direct anchors                         2 per jaw, 4 total
maximum capture penetration            2.8 mm including 0.4 mm margin
minimum incident-tet ratio at capture  0.20
```

attachment 使用零历史 compliant XPBD：

```text
compliance                             0.05 m/N
relaxation                             1.0
maximum correction                     0.10 mm/substep
finite-volume transfer                 1 incident-tet layer
runtime grip volume floor              0.03
```

每个夹爪的两个直接目标先形成各自的位移源场，再通过一层 incident tetra 将主要运动传给
局部体积节点；直接锚点保持为各自夹爪的满权控制，候选位移按该夹爪单独做 `J>=0.03`
line search。这里的有限体积传播是让局部四面体共同移动，减少“四点拉走、周围完全不动”
的尖峰，不是把整个抓取区改成刚体。

夹爪外侧还允许一圈小范围表面节点随整体抬起：只沿 collision-skin 共享边扩展一代，
边长不超过 `4.0 mm`，权重按距离 Gaussian 衰减。仍只扩展一代，具体 support 数量由
四个 direct anchor 所在位置的局部表面拓扑决定。为避免 q7 闭合把外圈向两侧拉开，
这些 support 不属于任一
per-jaw 源场，而只绑定到两夹爪共同的 jaw-hinge parent/wrist frame；它们在两侧直接控制
之后应用一次，并再次经过 `J>=0.03` line search。q7 相对捕获角打开 `0.08 rad`，或进入
张开/宽开状态时，direct 和 support 绑定全部清除。

## 5. 左右目 2D RGB 反馈

### 5.1 共同的 2D 图像损失

左右目分别渲染并计算 masked Smooth-L1 RGB 残差，`beta=0.05`。每个相机在自己的有效
像素权重内归一化，再对有效相机等权平均：

\[
D_{vis}=\frac{1}{|\mathcal C|}\sum_{c\in\mathcal C}
\frac{\sum_pW_c(p)\,\rho(I_c^{sim}(p)-I_c^{obs}(p))}
{\sum_pW_c(p)}.
\]

\(W_c\) 来自组织 mask、器械遮挡 mask、图像边界和 **2D 夹爪距离场**。当前主要范围是
夹爪附近 `120 px` 满权、`360 px` 外为零；组织边缘 `24/64 px`、图像边缘
`48/96 px` 平滑过渡，器械遮挡带 `6 px`。图像上方后侧在 `140–280 px` 继续衰减。
组织 mask 先做 `7 px` erosion，高亮像素权重为 `0.10`。这些都是左右目各自的 2D
像素门控，不是 3D 距离场。输入视频帧在损失前由 BGR 转为 RGB；residual 模式把图像、
权重和相机内参共同缩放到原分辨率的 `0.25`，左右相机仍各自归一化后等权平均。

### 5.2 视觉力：已实现的回退模式

使用 `--visual-feedback-mode force` 时，每次图像更新只临时优化组织 Gaussian 中心一轮：

```text
mean learning rate                    2e-4
optimizer iterations                  1
k_p                                   0.05
single Gaussian force cap             3e-5 N
single physical particle force cap    3e-5 N
particle acceleration cap             1 m/s²
sum of particle-force norms cap       0.02 N
artificial force spread               0 layers
```

设临时优化目标为 \(\mu_g^*\)，则

\[
\Delta\mu_g=\mu_g^*-\mu_g,\qquad
f_g=\operatorname{clip}(\alpha_gk_p\Delta\mu_g,3\times10^{-5}\,\mathrm N).
\]

每个 Gaussian 绑定到一个物理表面三角面，保存重心权重 \(\lambda_{gi}\)。为了不让力随
Gaussian 密度线性增大，先对每个物理节点计算支持和

\[
S_i=\sum_{h:i\in T_h}\lambda_{hi},
\]

再散射

\[
f_i^{raw}=\sum_{g:i\in T_g}\frac{\lambda_{gi}}{S_i}f_g.
\]

固定节点不受力；动态节点再经过单节点力、质量相关加速度和全局总预算限幅，作为
`particle_f` 进入下一物理积分。`spread_layers=0` 表示视觉力只直接作用于绑定面的节点，
体内和周围组织由距离/体积/形状约束传播。新图像力每 3 个物理帧求一次，中间帧缓存并
重新施加同一组已限幅粒子力。这个模式改变速度和后续动力学，但力散射只是最近物理面的
近似，不是完整视觉嵌入对所有节点的精确转置 Jacobian。

### 5.3 视觉 residual：当前初步尝试

GUI CLI 默认选择 `--visual-feedback-mode residual`。它不把图像误差转换成力，而直接优化
物理节点的有界增量 \(r\in\mathbb R^{1465\times3}\)：

\[
L=D_{vis}+0.4L_{dist}+1.0L_{vol}+0.008L_{shape}
+0.05L_{spatial}+0.10L_{temporal}+0.01L_{mag}.
\]

这里 residual 内部使用归一化边长误差、\((V/V_0-1)^2\) 和
\((F^TF-I)^2\) 作正则；这与主 XPBD 当前仍执行的绝对体积约束不同。每个 Gaussian 的
平移增量由其视觉三角形三个顶点、每个顶点三个物理支持的线性重心权重得到，最多涉及
九个物理节点；静止 offset-frame 旋转项暂未进入线性中心 Jacobian，但优化器会从九个支持
节点精确重建三个视觉顶点，并把当前视觉面形变作用到静止 Gaussian 协方差。因此 residual
渲染中的 Gaussian mean、quaternion 和 scale 都随候选节点残差变化；RGB 和 opacity 保持
固定。器械区域仍由 2D mask 抑制。

当前强档为：

```text
left/right observations                 2D RGB, equal camera weight
render scale                            0.25
Smooth-L1 beta                          0.05
Adam iterations                         8 per update
node learning rate                      4e-5 m (0.040 mm)
maximum node residual                   0.60 mm/update
loss weights, data/dist/vol/shape       1.0 / 0.4 / 1.0 / 0.008
loss weights, spatial/temporal/magnitude 0.05 / 0.10 / 0.01
update interval                         3 physics frames
healthy-tet residual floor              J >= 0.30
already-bad tet allowed relative loss   <= 2% per update
local freeze passes / global backtracks 4 / 10
```

每次从零增量开始，上一轮 accepted residual 只作为 temporal reference，不再次作为本轮
初值叠加。候选先局部清零危险 tetra 的相关节点，必要时最多做 10 次二分回溯。若某个
tet 在修正前已有 \(J<0.30\)，本轮只允许它相对当前值再下降最多 `2%`，而不是强制恢复到
`0.30`。最终接受条件包括：位置和 residual 数值有限、没有新增翻转、精确运行时重蒙皮
后的双目视觉损失严格小于初始损失，且任一有效相机的 loss 不允许明显退化；总损失不作
为接受门限。优化器中的平移 Jacobian 虽然忽略了 offset-frame 旋转，但接受前会把候选
节点写入临时状态，用正式视觉顶点嵌入和 Gaussian 蒙皮重新计算 mean/quaternion/scale，再渲染
一次。如果精确结果不再改善，就恢复原始 `particle_q` 并重新蒙皮，候选不会残留。

接受后只写 `particle_q` 并重新蒙皮 Gaussian，不写 `particle_qd`，所以这是一项位置状态
修正，不是人工冲量。拒绝时物理位置、速度和刚度均不改变，同时清空 temporal reference；
下一次 residual 仍从零增量开始。

抓取激活后，visual residual 现在使用运行时动态排除掩码。种子只包含当前实际绑定的 direct
anchors 和 support nodes；掩码再包含与任一种子直接共享一个四面体的节点。扩张只执行一次，
新邻居不继续作为种子，因此是严格的一层 incident-tet 邻域，不会沿四面体图扩成第二圈。
掩码内 residual、优化梯度和 temporal reference 均强制为零；释放后掩码自动清空。这个小范围
隔离用于消除“图像把抓取节点拉走、下一物理子步 `u_t` 又拉回”的周期性跳点，同时保留外围
组织的双目修正。`Visual residual diagnostics` 显示动态排除的非固定粒子数量。

视觉数据项还会单独反传一次节点梯度，只用来判断节点是否真正受到当前 masked RGB 的
监督；每目同时记录 loss、权重和有效像素/mask 覆盖率。这条链路已经接到 GUI 并有单帧/
合成门禁，但仍是初步尝试：尚未完成 Reset 后连续
压下—闭合—抬起—放下—松开的双目闭环验收，也尚未证明 residual 的方向在左右目、
遮挡和接触区始终一致。它不能代替接触非穿透和四面体质量修复。

### 5.4 在线刚度修正：当前初步尝试

在 `paper_pbd + residual` 模式下，CLI 默认同时开启 `--online-stiffness-update`。更新器持有
XPBD projector 内部的 `k_dist[i]` 和 `k_shape[i]` 张量，但先在独立张量中生成候选值，
不会立刻覆盖当前已验证参数。volume 刚度始终是固定全局标量 `1e10`。

设物理预测相对 rest 的形变为 \(d_i=x_i^{pred}-x_i^0\)，accepted residual 为 \(r_i\)。
节点先经过以下原始信号门控：

```text
non-fixed node                         required
residual norm                          >= 0.02 mm
deformation norm                       >= 0.10 mm
all incident tets before residual      J >= 0.03
all incident tets after residual       J >= 0.03
masked RGB node gradient               non-negligible
direct/support u_t and one-ring        excluded
capture/release/contact-patch change   pause current update only
rapid q7                               candidate allowed; no history snapshot
penetration > 2.8 mm                   paused
residual/deformation full scale        0.20 / 0.75 mm
```

当前全局过深穿透暂停门为 `2.8 mm`，与 persistent-grip capture 上限对齐；transition cooldown
从 3 次缩为 1 次，全局 rapid-q7 暂停关闭。它们只决定是否允许产生/验证刚度 candidate，
不改变接触投影或抓取捕获本身。快速 q7 仍不会进入准静态 history snapshot。

提交门现在使用 5 帧真实闭环影子，并要求第 1..5 帧 residual 写回前 prediction loss 的
平均值至少下降 `1e-5`。每个候选方向还会测试 `0.5/1.0/1.5/2.0` 四个 log-step 倍率，
选择双目累计 loss 最低且通过局部体积、翻转、穿透、锚点和历史门的倍率。

满足门控的节点产生有界方向信号

\[
s_i=\operatorname{clip}\!\left(
-\frac{d_i\cdot r_i}{\|d_i\|\|r_i\|}+0.15,-1,1\right)
\operatorname{clip}\!\left(\frac{\|r_i\|}{0.20\,\mathrm{mm}},0,1\right)
\operatorname{clip}\!\left(\frac{\|d_i\|}{0.75\,\mathrm{mm}},0,1\right).
\]

其中正值表示硬化，负值表示软化。原始信号在 8,024 条四面体边组成的节点图上做一次
邻居平均混合，`blend=0.35`。固定、低质量、无有效视觉监督以及 `u_t` 控制区既不能作为
更新目标，也不能通过邻域平滑把信号泄漏回控制区。之后生成候选逐节点 EMA：

\[
e_i^{candidate}=0.70e_i^{verified}+0.30\bar{s}_i^t.
\]

质量门控失败、固定或进入 `u_t` 控制区的节点会把候选 EMA 清零。其他节点即使本轮原始
residual 低于阈值，也可能因邻居平滑或尚未衰减完的 EMA 继续发生小幅更新。最终在
log-space 生成候选：

\[
\ell_i=\operatorname{clip}(0.18e_i^t,-0.18,0.18),
\]

\[
k_i^{dist,candidate}=\operatorname{clip}(k_i^{dist,verified}e^{\ell_i},0.10,2.00),
\qquad
k_i^{shape,candidate}=\operatorname{clip}(k_i^{shape,verified}e^{1.5\ell_i},0.003,0.020).
\]

一次 accepted 更新中，distance 的对数步长最多为 `±0.18`，shape 的对数步长最多为
`±0.27`；随后再受各自绝对上下界限制。EMA 从零开始且收到满幅证据时，第一轮实际响应为：
软化信号下 distance `0.20 -> 0.19103`、shape `0.004 -> 0.0037339`；硬化信号下分别为
`0.20 -> 0.21110`、`0.004 -> 0.0043375`。连续且每次都通过验收的满幅证据中，软化会在
第 7/3 次提交把 distance/shape 推到 `0.10/0.003` 下限；硬化会在第 16/9 次提交到达
`2.00/0.020` 上限。候选先在最多 4 个准静态历史快照上做短松弛，
比较当前 verified 与 candidate 的质量加权节点 RMS。通过历史门后保存完整 rollout 快照，
等待下一张真实图像。

之后连续收集 5 个视频帧的真实 `SE(3)+q7` 命令，从同一快照分别运行 verified/candidate
影子。影子使用真实 capture/release 状态机，并在中间帧重放与 live 相同的视觉 residual；
候选帧本身已经写入快照，不重复回灌。验收目标为

\[
L_{H=5}=\frac{1}{5}\sum_{h=1}^{5}D_{vis}^{pre\text{-}residual}(t+h),
\]

所以候选不能只让第 5 帧好看、却把前 4 帧弄差。通过所有门后同时提交材料场和这条已经
验证的固定滞后状态；状态接回要求全粒子 RMS 不超过 `0.25 mm` 且任一粒子不超过 `1.0 mm`，
超限时只提交材料、不接回状态。否则保留 verified、衰减 EMA 并记录拒绝原因。影子路径最多
保存 120 个物理步、最多跨 10 个视频帧；Pause 或视频结束时取消未完成候选。

当前 residual 优化器本身仍不读取在线 `k_dist/k_shape`，也不计算
\(\partial D_{vis}/\partial k\)。因此这仍是 residual 驱动的局部启发式提案，加上不可微
影子 rollout 验收，而不是端到端可微材料辨识。

### 5.5 实际运行时序、持久状态与 Reset

GUI 的一次物理循环顺序固定为：

```text
1. environment.step：使用 verified k_dist/k_volume/k_shape 完成 12 个 substeps
2. 把器械重新锚定到当前离线 SE(3)+q7
3. 若正在 Play、到达反馈间隔且出现新图像：求解双目 residual
4. 用正式视觉嵌入精确重蒙皮并验收；accepted 才写 particle_q
5. accepted 且在线刚度开启：应用质量/视觉/u_t/接触门，生成 candidate
6. 用准静态历史快照比较 verified/candidate 的短松弛漂移
7. 保存完整状态和之后每一步器械命令；真实参数仍保持 verified
8. 累积到 H=5：从同一快照运行 verified 与四个 candidate step 的真实 residual/抓取闭环
9. 五帧平均 prediction gap 和物理门都通过才 commit；否则 reject 并衰减 EMA
10. commit 时接回已验证分支状态；RMS/单粒子超过 0.25/1.0 mm 时只提交材料
```

视频帧由另一个标称 `30 FPS` 的协程推进；物理循环的配置步长是 `1/60 s`。Pause 时视频
与视觉反馈停止，但物理循环仍继续执行，器械保持当前时间戳姿态。`force`、`residual` 和
`off` 三种模式互斥；在线刚度更新器只在 `paper_pbd + residual` 下构造。

影子路径的快照不是只有 `particle_q`：它同时保存物理 state/control、Gaussian state、
`sim_time`、运动学插值起点/目标、全部 persistent-grip 状态与数组，以及 distance/shape
刚度。影子运行冻结新的 capture/release，但保留已经存在的 direct/support 绑定。

`Reset` 会恢复首帧上述完整 rollout 状态，把 `k_dist` 和 `k_shape` 恢复到更新器创建时保存
的 `0.20/0.004`，清零 EMA、候选、历史缓冲、上一轮 residual、视觉力缓存、验证指标和
累计 candidate/commit/reject count。accepted residual 和在线刚度都只在当前运行期间
保留，不写回资产或配置文件。

### 5.6 GUI 当前暴露的调参与观测量

默认面板已精简为每类 1--4 行摘要：材料 Reset 值和最大位移、distance/shape
min/median/max、active/harden/soften、commit/reject、抓取状态与左右接触数、gap/penetration，
以及 residual 是否接受、loss 降幅和最大修正。逐相机 loss、质量/视觉/`u_t` 屏蔽数、EMA、
参数边图粗糙度、预测 horizon、contact safe scale 等仍保留，但分别折叠在 `Stiffness
diagnostics`、`Contact diagnostics` 和 `Visual residual diagnostics`。PSM 手动对齐和静态
求解器配置也默认折叠，不再占满实时观察窗口。

`Online stiffness tuning (pause to apply)` 现在可运行时调节全部更新器公式参数：distance/
shape 上下界、log learning rate、EMA 新证据权重、hardening bias、shape gain、maximum log
step、residual/deformation 的最小值与 full scale、空间平滑次数/blend，以及 candidate 被拒绝
后的 EMA 保留比例。滑条先修改草稿；必须 Pause 且没有 pending candidate 时点击 Apply。
为允许做出肉眼明显的 A/B，材料 bound GUI 当前使用宽实验范围：distance lower/upper 分别为
`0.01..1.00 / 0.10..10.00`，shape lower/upper 分别为
`0.0001..0.030 / 0.001..0.100`。启动 bound 仍是 `0.10..2.00 / 0.003..0.020`，只有 Apply
后才改变 verified 场；lower 不得高于对应 upper。shape 超过 `0.020` 明确属于偏离论文搜索
范围的实验档，不应直接当生产默认。
Apply 保留当前 verified 材料场并按新界限裁剪，清空 EMA、历史快照和 temporal residual，
再进入 3 次更新冷却；它不重置粒子位置或固定 `k_volume`。`Restore startup values` 只恢复
草稿，仍需 Apply。Reset 恢复 `0.20/0.004` 材料初值，但保留本次进程中已应用的调参规则。

完整公式、每个旋钮变大/变小的影响和推荐顺序见
`在线刚度参数小白说明与GUI调参指南.md`。当前仍没有逐节点参数热力图；轨迹指标已可以持久化，
并已生成 frame `350..560` 的三组初步 CUDA 对照，但尚不是 Reset 后跑到 release 的完整循环。

实时更新的当前控制入口为：

| CLI | 默认值 | 直接控制的量 |
|---|---:|---|
| `--visual-feedback-mode` | `residual` | `residual / force / off` 三选一 |
| `--visual-residual-iterations` | `8` | 每次 residual 的 Adam 步数 |
| `--visual-residual-learning-rate-m` | `4e-5` | 节点 residual 学习率，单位 m |
| `--visual-feedback-update-interval` | `3` | 两次图像更新之间的 physics frame 数 |
| `--online-stiffness-update` | `true` | accepted residual 后是否运行刚度更新 |
| `--stiffness-log-learning-rate` | `0.18` | \(\ell_i\) 的 log-space 系数 |

`maximum_residual=0.60 mm` 和视觉 loss 权重仍由 residual mapper 的启动配置控制；tet 质量、
prediction gap、历史状态和接触过渡等安全门仍集中在入口顶部 `STIFFNESS_*` 常量中，不允许
GUI 热修改。材料更新器自身的界限、增益、EMA、证据尺度与空间平滑已经可以按上述流程
安全热调。

### 5.7 接触与持续抓取 GUI 热调

`Tip entry and grip (pause to apply)` 默认显示三个参数，并在跳点稳定折叠区显示四个参数：

```text
Tip entry allowance       启动 3.00 mm，自动扩展barrier，GUI上限 4.99 mm
Capture max penetration   启动 2.80 mm，GUI 范围 0.10..10.00 mm
Grip support radius       启动 4.00 mm，GUI 范围 0.00..30.00 mm

Particle jump stabilization
  Grip correction cap     启动 0.10 mm/substep，GUI 0.005..2.000
  Grip compliance         启动 0.05 m/N，GUI 0.00..2.00
  Material/grip velocity  启动 0.14，GUI 0.00..1.00
  Particle damping        启动 10/s，GUI 0..100/s
```

contact margin/correction/relaxation、friction、采样/query/CCD/spread、solver passes、minimum
`J`、support generations、transfer layers 和捕获/释放角等仍保留在启动配置与
内部 schema 中，但不再构建 GUI 控件。这样实时窗口不会出现一整套容易混调的接触参数。

滑条仍采用 draft -> Pause -> Apply。接触/抓取 Apply 会：

- 先完整校验 `query>=margin`、`tip<barrier<=jaw distal`、
  `closed<release<=wide-open`、体积阈值、层数和柔顺度；
- 用当前工具 shape 和未暴露的 pressure-shoulder/top-support 配置重建
  `TriangleSkinContactProjector`，使采样 mask、support 邻域和 transfer tet 与新参数一致；
- 保留当前组织粒子位置和 verified distance/shape 材料场，但主动释放旧的 persistent-grip
  anchor，并清掉旧接触阶段的刚度 EMA、历史/validation/residual，进入 3 次更新冷却；
- 只在当前进程生效；`Restore startup entry/grip` 恢复上述七个草稿值，仍需 Apply。

`Simulator.configure_triangle_skin_contacts()` 同步补齐 release angle、release delta、wide-open
angle 和 q7 motion epsilon 接口，避免 GUI 只改变捕获阈值却暗中恢复另一套释放状态机。
`query distance=10 mm` 仍只是工具采样到组织三角形的搜索半径，不是位移传播半径；远端运动
主要由材料图约束、contact spread、grip support radius/generations 和 transfer layers 决定。

headless fake-imgui 门禁明确检查只有上述七个参数控件存在，并检查原 contact margin、friction、
query、spread、angle 等控件均未生成；暂停应用、运行中拒绝、projector 替换和旧
anchor/evidence 清理仍有覆盖。完整 CPU `paper_soft` 场景回归也通过全部 gate，默认
`0.13/0.15 rad` 捕获/释放行为未改变。

## 6. FPS 与当前实时性边界

必须区分配置频率和墙钟性能：

| 项目 | 配置/实测 |
|---|---:|
| 视频请求播放 | 30 FPS |
| 物理时间步 | 60 Hz，`dt=1/60 s` |
| 每物理帧子步 | 12，即 720 substeps/s 的仿真时间 |
| 双目反馈更新间隔 | 每 3 个物理帧，标称 20 updates/s |
| force 缓存重放 | 标称每个物理帧，即 60 Hz |
| 8-iter residual 单帧 warm 测试 | 92.4 ms，约 10.8 solves/s，未含完整循环 |
| 软化前 CPU 两步接触诊断 | 638.2 ms/physics frame 中位数，约 1.57 FPS |

CPU 数字来自软化前 `paper_pbd v15`、7,579 个器械采样和 state `1795..1796` 的短跳帧
诊断，只用于确认量级，不是当前软档的 GUI 基准。此前较接近的 CUDA 接触段测得约
`60.7 ms/physics frame`（约 `16.5 FPS`），但它早于最后的 barrier/共同腕部 support
修改，也没有同时运行当前 8-iter residual，因此不能作为最终当前 FPS。

当前 GUI 的物理协程在计算后还会 `sleep(1/60)`，视觉 residual 又是同步插入的，所以
`60 Hz` 和 `20 updates/s` 都只是仿真设定上限，不代表已达到实时。当前尚无“完整双目
residual + 在线刚度 + 接触/抓取”的冻结 CUDA 端到端 FPS；30 FPS 实时闭环仍未实现。

## 7. 当前验证覆盖与未验证边界

已经确认：

- v15 的节点、tet、碰撞面、视觉面和 Gaussian 数量能按当前入口正确加载；
- 软化前短 CPU 诊断加载 `k=0.35/1e10/0.006`、`0.008 mm` top barrier、7,579 个采样，
  两个跳帧状态没有新翻转；
- 当前明显软档的独立材料 CPU 门禁加载 `k=0.20/1e10/0.004` 和 6 次材料迭代；静止态
  最大漂移 `2.77e-10 m`、最小 `J=0.999993`，中心节点扰动 `0.75 mm` 后恢复路径无翻转、
  最小 `J=0.7858`，并同时降低 edge/volume constraint RMS；
- 合成测试覆盖每侧两个 `u_t`、一层有限体积传播、共同腕部 support、q7 打开后清空；
- visual-force 三节点归一化散射、节点/加速度/总力限幅已有独立测试；
- visual residual 的单帧优化、正体积回溯、逐相机/有效像素指标和节点视觉监督梯度已有测试；
- 在线刚度的 candidate/verified 隔离、上下界、EMA、局部质量门、`u_t` 硬屏蔽和拒绝回滚
  已有合成测试；当前 `0.20/0.004` 初值明确验证为可双向更新，连续满幅软化提交在第 7/3
  次分别达到 `0.10/0.003` 下限，连续满幅硬化在第 16/9 次达到 `2.00/0.020` 上限，两端
  继续同向更新都会饱和且可以反向离开；默认一环平滑开启、相邻链两端持续收到相反证据
  时，最终 distance 为 `0.10/0.23896/2.00`，验证 20 倍区域跨度不会被空间平滑抹掉；
- `0.10/1e10/0.003` 下限和 `2.00/1e10/0.020` 上限、6 次材料迭代的独立 CPU 门禁均通过；
  两端静止态无翻转、最小 `J=0.999993`，`0.75 mm` 扰动后的 80 子步路径也无翻转，最低
  `J` 分别为 `0.87864/0.94422`；
- `paper_soft` 完整 CPU 场景已验证 rollout 快照可同时恢复粒子/Gaussian、插值、持握和材料
  状态；20 步静态门禁无翻转，最小 `J=0.9999986`。

当前没有以下验证结果：

- Reset 后连续 CUDA 压下—闭合—抬起—放下—松开全轨迹结果；
- 同一轨迹 `off / force / residual` 的冻结对照数据；
- 已有 frame `350..560` 的实际 CUDA 三组输出，但尚缺 Reset 后到 release 的完整循环；
  关键点误差和端到端 FPS 仍不在本轮刚度评测 schema 内；
- H=5 累计闭环验收与固定滞后状态接回已经接线；最终完整 CUDA 复核产生 11 次 commit，
  9 次状态接回，2 次因单粒子变化超过 `1 mm` 只提交材料；
- `H=1/3/5/10` 多帧开放环已完成：11 次 started、10 次完整走完 H=10、1 次在轨迹尾端
  不完整。end-to-end H=1/3/5 正改善分别为 `8/11、9/11、10/11`，H=10 为 `7/10`；
- 211 帧总体 prediction loss：Fixed `0.02412187`、Residual-only `0.02111430`、最终 Online
  `0.02105285`。Online 相对 Residual-only 低 `0.2911%`，逐帧 `169:41`，配对 t 检验
  `p=0.00257`；当前短轨迹已经证明净收益，但尚不能外推到完整 release 循环；
- 在线 `k_dist/k_shape` 已自动按动作阶段分组，尚无参数收敛或跨次运行重复性数据；
- residual loss 下降来自真实组织轮廓对齐、而非 mask/遮挡区或局部体积补偿的独立判别。

## 8. 关键代码

| 功能 | 文件 |
|---|---|
| 场景、材料和接触参数 | `examples/embodied_environments/super_embodied/super_embodied.py` |
| GUI、双目反馈时钟和默认 CLI | `examples/example_embodied_super_offline.py` |
| 距离/体积/形状 XPBD | `src/embodied_gaussians/physics_simulator/integrator.py` |
| 三角外皮接触、`u_t` 与共同腕部 support | `src/embodied_gaussians/physics_simulator/triangle_skin_contact.py` |
| Gaussian 蒙皮和视觉力散射 | `src/embodied_gaussians/embodied_simulator/warp.py` |
| 视觉力限幅、residual 接受与状态回写 | `src/embodied_gaussians/embodied_simulator/simulator.py` |
| 双目 residual 目标与求解器 | `src/embodied_gaussians/physics_simulator/visual_tissue_residual_mapping.py` |
| residual 驱动的在线刚度 | `src/embodied_gaussians/physics_simulator/online_tissue_stiffness.py` |
| 多 horizon 评测、动作阶段和 JSONL/summary | `src/embodied_gaussians/physics_simulator/stiffness_evaluation.py` |
| 三组 headless 轨迹入口 | `scripts/run_super_three_way_stiffness_comparison.sh` |
| 三组共同帧损失聚合 | `scripts/summarize_super_three_way_comparison.py` |
| 详细历史和已回滚实验 | `PROGRESS.md` |

当前框架最准确的概括是：**物理接触和抓取负责主要运动，左右目 2D RGB 提供小幅、有界、
可拒绝的状态/参数修正；在线刚度用 H=5 闭环证据和固定滞后状态接回兑现收益。** frame
350..560 短轨迹已获得统计显著改善，但尚未达到实时，也尚未完成 Reset 到 release 全循环复核。

## 9. 2026-08-13 刚度优化完成度审计

对照 `刚度优化方案.md`、实际 runtime 和现有门禁后的结论是：**在线安全闭环主体已实现，
整套方案尚未完成最终评测验收。** 已实现部分包括 residual 前后状态分离、`u_t`/一环硬
排除、局部质量与视觉监督门、空间平滑、EMA、log-space candidate、candidate/verified
双缓冲、完整状态回滚、最多 4 个历史准静态状态的 1-frame 初版松弛门、下一可用新图像的
verified/candidate 双影子比较，以及图像/体积/穿透/锚点/历史联合提交门。

第 5、6 节的 `H=1/3/5/10` 开放环、全轨迹持久化和动作阶段分组已于 2026-08-14 补齐代码；
2026-08-17 的同一 CUDA 短轨迹 `fixed-k / residual-only / residual+stiffness` 对照已验证材料
更新与固定滞后状态接回的额外收益；跨次独立重复、完整 release 循环和参数真实性仍缺。
因此当前可以称为“方案功能实现完成，并在单条短轨迹上证明闭环收益”，仍不能称为已完成
跨轨迹材料辨识验收。

面向小白的完整公式、变量含义、启动条件、物理/视觉双时钟、一次 candidate 从生成到
commit/reject 的逐步流程，以及完成方案仍需哪些实验，见
`当前刚度优化方法与运行流程_小白公式版.md`。

## 10. 2026-08-14 恢复版方案复核与公式文档重写

用户恢复后的 `刚度优化方案.md` 共 402 行，正文包含第 5 节 `H=1/3/5/10` 多帧开放环和
第 6 节完整指标/动作阶段分组。本节记录的是补齐前的复核结果；两项缺口已在下一节所述实现中
补齐。当前 GUI 可使用在线刚度闭环，显式启用评测输出后还会运行完整多 horizon 评测。

恢复版正文中的 `0.20..1.60` distance bounds、`0.004..0.020` shape bounds、`0.20/0.80`
EMA 和旧投影型方向项已经被后续明确调参覆盖。当前代码实际使用 `0.10..2.00`、
`0.003..0.020`、`0.30/0.70`，并把方向写成归一化夹角，再独立乘 residual/deformation
幅值因子；这属于较新的运行实现，不应为了逐字匹配恢复版而回滚。

`当前刚度优化方法与运行流程_小白公式版.md` 已重写为逐公式教程：从向量长度/点积开始，
为 distance、volume、shape XPBD、Smooth-L1、双目等权损失、七项 residual 目标、软硬信号、
邻域平滑、EMA、指数候选、历史质量加权 RMS 和下一图像 prediction gap 分别提供符号解释、
直观比喻、带数值算例及参数增减效果；同时给出从启动、普通物理帧、accepted residual、
pending 到双影子 commit/reject 的完整状态机。

## 11. 2026-08-14 第 5、6 节评测实现

方案正文要求的功能代码现已全部接通。`SuperPlaybackControls` 对每个 committed candidate 保存
提交前快照、old/new 刚度和真实器械命令，在精确目标视频帧 `H=1/3/5/10` 上分别运行：

1. `material_isolation`：冻结快照中的 direct/support 与 grip 状态机；
2. `end_to_end`：允许接触和抓取状态机在同一器械命令下自行演化。

每种协议都比较 baseline/candidate 的双目 Gap、逐相机 Gap、distance/volume/shape loss、最小
`J`、翻转数、穿透、锚点误差和粒子 RMS，并额外运行一次不写回的 visual residual，记录
开放环状态重新对齐图像所需的 residual RMS/max。目标图像按 `start_frame+H` 精确读取；评测
结束恢复 live 仿真、控制器和当前图像。

新增 `stiffness_evaluation.py`，提供 `press/close/capture/lift/place/release/idle` 分类器和安全
持久化 recorder。显式传入 `--stiffness-evaluation-output` 后输出 `metadata.json`、append-only
`events.jsonl` 和持续原子更新的 `summary.json`；summary 按阶段、图像/物理/材料/预测类别为
所有数值生成 count/mean/min/max，并对 accepted/status/protocol/拒绝原因等分类值计数。已有
结果文件不会被覆盖。普通 GUI 默认不启用四 horizon
评测，因此不承担额外影子和 residual solve 成本。

便捷入口为 `scripts/run_super_stiffness_evaluation.sh`。另新增三组 headless 入口和聚合器，已在 CUDA 上
生成 frame `350..560` 共 211 帧的初步实测。完整动作循环、真正刚度 commit 后的
`H=1/3/5/10` 和重复性仍是下一步实验任务。

## 12. 2026-08-17 H=5 累计闭环与固定滞后状态接回

单帧 H=1 近全开放版本虽有 33 次 commit，但总体比 residual-only 差 `0.1233%`。修正分三步：

1. candidate 等待 5 个真实视频帧，影子重放中间帧的视觉 residual 和实际抓取状态机；
2. admission 直接最小化第 1..5 帧平均 prediction loss，并对 log-step 做
   `0.5/1.0/1.5/2.0` line search；证据门取 `1e-5`；
3. commit 后接回已验证分支的固定滞后状态，RMS/单粒子硬上限为 `0.25/1.0 mm`。

为检查“是不是门禁压住了收益”，还跑了一次只在独立进程内生效的极松试验：局部 J 学习门
`0.03→0.005`、穿透门 `2.8→10 mm`、cooldown `1→0`，并大幅放开图像/相机/体积/穿透/
锚点/历史容差。它产生 41 次 commit，但总体比 residual-only 差 `0.0612%`，逐帧 `89:121`；
说明继续放门禁会引入长期漂移。试验后临时开关和极松常量已删除，默认恢复正式门禁。

最终 H=1/3/5/10 完整 CUDA 复核：frame `350..560`，211 个共同视频帧，每帧 3 个物理步。

| 模式 | 平均 prediction loss | 相对 Fixed | 相对 Residual-only |
|---|---:|---:|---:|
| Fixed PBD | `0.02412187` | baseline | - |
| Residual-only | `0.02111430` | `-12.468%` | baseline |
| Residual + online + fixed-lag | `0.02105285` | `-12.723%` | `-0.2911%` |

在线组逐帧胜负为 `169:41`，另 1 帧相同；配对 t 检验 `p=0.00257`，Wilcoxon 单侧
`p=6.62e-18`。按 50 帧窗口，350--399、400--449、450--499、500--549 分别改善
`0.404/0.503/0.079/0.362%`；最后 550--560 回退 `0.193%`。安全指标为 0 翻转、最小
`J=0.010010`、最大穿透 `2.353 mm`。

11 次 commit 中 9 次状态接回；接回 RMS 最大 `0.211 mm`，未超过 `0.25 mm`。两次候选的
单粒子最大变化超过 `1 mm`，因此只提交材料而不接回状态。end-to-end 多 horizon 正改善数：

| H | 样本 | 正改善 | 平均 gap 改善 | 中位数 |
|---:|---:|---:|---:|---:|
| 1 | 11 | 8 | `+1.641e-6` | `+3.859e-6` |
| 3 | 11 | 9 | `+31.289e-6` | `+17.591e-6` |
| 5 | 11 | 10 | `+29.215e-6` | `+20.342e-6` |
| 10 | 10 | 7 | `-15.352e-6` | `+13.845e-6` |

H=10 均值仍被 3 个负样本拉低，所以当前结论限定为这条 211 帧短轨迹上的统计显著收益；
下一步仍是独立重复运行、完整 Reset 到 release 循环、关键点误差和 CUDA 端到端性能。

正式结果：

```text
outputs/stiffness_online_h13510_fixed_lag_final_350_560_20260817/
outputs/stiffness_three_way_h13510_fixed_lag_final_350_560_20260817/comparison.json
outputs/stiffness_three_way_h13510_fixed_lag_final_350_560_20260817/comparison.md
```
