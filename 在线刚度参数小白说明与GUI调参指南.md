# 组织刚度、接触与抓取参数小白说明与 GUI 调参指南

本文解释当前 `paper_pbd` 组织为什么会变形、在线刚度为什么会变大或变小，以及每个可调
参数的直观影响。它面向第一次接触 XPBD、EMA 和视觉 residual 的读者。

当前实现不是直接从图像求一个“真实杨氏模量”。它做的是：物理先预测组织位置，双目图像
指出预测应该往哪里修，再根据修正方向提出局部 distance/shape 刚度候选；候选只有通过历史
状态和下一张真实图像的影子回放，才写入正式物理参数。

## 1. 先记住三个层次

### 第一层：材料本身有多硬

当前每个物理节点有两种局部刚度，另外还有一个全局体积刚度：

| 参数 | Reset 值 | 一句话理解 |
|---|---:|---|
| distance stiffness (k_{dist}) | `0.20` | 相邻节点之间的“弹簧长度”有多不愿改变 |
| volume stiffness (k_{vol}) | `1e10` | 四面体有多不愿压扁或膨胀；当前全局固定 |
| shape stiffness (k_{shape}) | `0.004` | 一个四面体有多想保持原来的局部形状 |

这三个数不能直接比较大小。它们进入的约束公式和量纲缩放不同，不能因为 `1e10` 比 `0.20`
大，就说 volume 比 distance 硬五百亿倍。

直观影响：

- (k_{dist}) 变大：拉伸恢复更强、形变更容易沿连接传播，局部更像绷紧的皮；过大时夹爪
  附近会和远处一起动。
- (k_{dist}) 变小：更容易局部拉长，远处跟随通常减少；过小时会显得松、局部位移和速度
  可能增大。
- (k_{shape}) 变大：局部四面体更不愿剪切、弯折或歪斜；过大时组织更像一整块。
- (k_{shape}) 变小：更容易弯折和局部改变形状；过小时容易出现软塌感。
- (k_{vol}) 变大：更接近不可压缩，压下去的体积会从周围鼓出。
- (k_{vol}) 变小：更容易直接压瘪，虽然看起来“软”，但可能是在用体积塌缩伪装柔软。

因此当前只在线更新 distance 和 shape，volume 固定为 `1e10`。

### 第二层：图像证据说应该变硬还是变软

物理预测相对静止形状的变形为

\[
d_i=x_i^{pred}-x_i^0.
\]

双目 residual 接受后，图像要求节点追加的修正为

\[
r_i=x_i^{accepted}-x_i^{pred}.
\]

把它想成两支箭头：

- (d_i) 是“物理已经把节点推到哪里”；
- (r_i) 是“图像说还应该往哪里补”。

方向信号为

\[
s_i=\operatorname{clip}\left(
-\frac{d_i\cdot r_i}{\|d_i\|\|r_i\|}+b,-1,1\right)
\operatorname{clip}\left(\frac{\|r_i\|}{R_{full}},0,1\right)
\operatorname{clip}\left(\frac{\|d_i\|}{D_{full}},0,1\right).
\]

- (d_i) 和 (r_i) 反向：图像把节点往静止位置拉，说明物理变形过大，(s_i>0)，倾向
  硬化。
- 两者同向：图像还要节点继续变形，说明物理变形不足，(s_i<0)，倾向软化。
- residual 或 deformation 很小：证据弱，信号接近零。

当前偏置 (b=0.15)，所以模棱两可的证据轻微偏向硬化。这是安全偏置，不是材料常数。

### 第三层：即使有证据，也不能一步跳到极端值

原始信号先在四面体边图上做邻域混合：

\[
\bar{s}_i=(1-\beta)s_i+\beta\operatorname{mean}_{j\in\mathcal N(i)}s_j,
\qquad \beta=0.35.
\]

再做时间 EMA：

\[
e_{i,t}=0.30\bar{s}_{i,t}+0.70e_{i,t-1}.
\]

最后在 log-space 更新：

\[
\ell_i=\operatorname{clip}(\eta e_{i,t},-L,L),
\]

\[
k_{dist,i}^{candidate}=\operatorname{clip}
(k_{dist,i}^{verified}e^{\ell_i},0.10,2.00),
\]

\[
k_{shape,i}^{candidate}=\operatorname{clip}
(k_{shape,i}^{verified}e^{g\ell_i},0.003,0.020).
\]

当前 (eta=0.18)、(L=0.18)、(g=1.5)。使用乘法而不是加法的好处是：`0.1` 和 `1.0`
处在不同数量级时，同一个证据仍表示相近的百分比变化。

理论单次上限是：

- distance：乘以 (e^{0.18}=1.197)，最多约 `+19.7%`；反向最多乘以 `0.835`；
- shape：乘以 (e^{0.27}=1.310)，最多约 `+31.0%`；反向最多乘以 `0.763`。

从零 EMA 开始，实际第一轮没有这么大。当前满幅合成证据测得：

| 方向 | distance | shape |
|---|---:|---:|
| 第一次软化 | `0.20000 -> 0.19103`，`-4.49%` | `0.004 -> 0.0037339`，`-6.65%` |
| 第一次硬化 | `0.20000 -> 0.21110`，`+5.55%` | `0.004 -> 0.0043375`，`+8.44%` |

## 2. GUI 基础调节区

展开 `Online stiffness tuning (pause to apply)` 可看到以下参数。

| GUI 参数 | 当前值 | 调大后 | 调小后 | 常见风险 |
|---|---:|---|---|---|
| Distance lower bound | `0.10`（GUI `0.01..1.00`） | 禁止特别软的区域 | 能形成更软、更局部的拉伸区 | 太低会松散、位移变大 |
| Distance upper bound | `2.00`（GUI `0.10..10.00`） | 允许形成更硬的区域 | 硬区更早饱和 | 太高可能和夹爪对抗、传播变远 |
| Shape lower bound | `0.003`（GUI `0.0001..0.030`） | 保留更多局部形状 | 更容易弯折和剪切 | 太低会软塌；高于 upper 会被拒绝 |
| Shape upper bound | `0.020`（GUI `0.001..0.100`） | 允许更强的形状保持 | 硬化更早饱和 | 超过 `0.020` 属于明显偏离论文范围的实验档 |
| Learning rate | `0.18` | 每个已验收图像改得更快 | 收敛慢但稳 | 太大易在软硬之间来回跳 |
| EMA new-evidence weight | `0.30` | 更相信当前图像、响应快 | 更相信历史、响应慢 | 太大追噪声，太小几乎不动 |
| Hardening bias | `0.15` | 更容易硬化 | 更容易软化 | 太大整体越学越硬；太小易软塌 |
| Shape update gain | `1.50` | shape 比 distance 改得更快 | 两者变化更接近 | 太大局部很快变成硬块 |

注意：上下限决定“长期最多能走多远”，learning rate、EMA 和最大 log step 决定“每次走多快”。
把范围扩大不会让下一帧突然跳到边界。

## 3. GUI 高级证据和空间平滑区

展开 `Advanced evidence and smoothing`：

| GUI 参数 | 当前值 | 变大 | 变小 |
|---|---:|---|---|
| Maximum log step | `0.18` | 放宽单次最坏变化 | 更保守，但到目标更慢 |
| Minimum residual | `0.020 mm` | 忽略更多小图像误差 | 更多节点参与，也更容易学噪声 |
| Residual full scale | `0.200 mm` | 同样 residual 的信号变弱 | 更快达到满幅信号 |
| Minimum deformation | `0.100 mm` | 只在明显变形区学习 | 靠近静止状态也会学习，方向更不稳定 |
| Deformation full scale | `0.750 mm` | 同样形变的信号变弱 | 小形变也能产生强更新 |
| Spatial smoothing passes | `1` | 材料区更平滑、边界更宽 | 更局部；`0` 时易出现椒盐参数 |
| Neighbor smoothing blend | `0.35` | 更听邻居，区域差异更平滑 | 更听自己，边界更尖锐 |
| Rejected EMA keep ratio | `0.50` | 拒绝后仍保留更多旧趋势 | 拒绝后更快忘记旧证据 |

`Minimum residual` 不应大于 `Residual full scale`；`Minimum deformation` 也不应大于对应 full
scale。Apply 时会检查，不合理组合会显示 `Settings rejected`。

## 4. 为什么有时参数完全不更新

满足公式还不够。以下情况会屏蔽节点或整轮暂停：

- fixed node；
- residual 前后任一 incident tetra 的 (J=V/V_0<0.03)；
- 双目 mask/renderer 对该节点没有有效监督；
- 节点属于夹爪 direct/support `u_t` 或其物理一环；
- capture/release 当帧、穿透过大或接触片突然切换；快速 q7 不再全局暂停候选，
  但仍不会写入准静态历史快照；
- residual 本身被拒绝；
- candidate 的历史松弛或下一图像影子回放没有比 verified 更好。

这样设计是为了不把“夹爪强制带着走”“遮挡错误”或“坏四面体”误认为材料软硬。

常用安全门当前为：

| 门 | 当前值 | 含义 |
|---|---:|---|
| 学习 tet 最小 (J) | `0.03` | 低质量单元不提供材料证据 |
| 过渡冷却 | `1` 次更新 | 只暂停接触状态切换的当前更新 |
| 全局快速 q7 暂停 | 关闭 | 交给下一帧影子验证判断；历史快照仍避开快速运动 |
| 最大允许穿透 | `2.8 mm` | 与 capture 上限对齐，超过后暂停刚度学习 |
| 下一帧最小降损 | `1e-6` 绝对值 | 取消旧 `0.1%` 相对门，但仍要求候选真的更好 |
| 历史快照 | 最多 `4` 个 | 防止新参数破坏旧的准静态状态 |
| 下一图像等待 | 最多 `120` 物理步/`10` 视频帧 | 超时就拒绝 candidate |

这是接近全开放的候选入口，但不是无条件改刚度：candidate 仍须通过双目图像、体积、翻转、
穿透和锚点检查后才能写入 verified。上述门目前没有放进常用 GUI，以免观察材料时误关保护。它们集中在
`examples/example_embodied_super_offline.py` 文件顶部的 `STIFFNESS_*` 常量中，修改后需要
重启程序。正式校准时应一次只改一类门，并保留 A/B 报告。

## 5. XPBD 求解器参数

单个约束的核心更新是

\[
\Delta\lambda=
\frac{-C-\widetilde\alpha\lambda}
{\sum_i w_i\|\nabla_iC\|^2+\widetilde\alpha},
\qquad
\widetilde\alpha=\frac{s_{comp}}{k\Delta t_{sub}^2}.
\]

不用记推导，只看分母：(k) 变大时 (widetilde\alpha) 变小，约束更敢修正位置，所以更硬；
子步时间和迭代次数也会影响最终效果。

| 求解器参数 | 当前值 | 增大影响 | 减小影响 |
|---|---:|---|---|
| material iterations/substep | `6` | 约束收敛更强，传播也可能更远，耗时增加 | 更软、更局部，但可能欠收敛 |
| material relaxation | `1.0` | 当前已是完整修正 | 小于 1 会削弱每次材料投影 |
| substep dt | 约 `1/720 s` | 大步更难稳定 | 小步更稳但更慢 |
| velocity damping | `10/s` | 更快消除摆动，也更黏 | 更有弹性，也更易振荡 |
| material minimum (J) | `0.01` | 更早局部修复塌缩单元 | 容许更扁的单元，翻转风险增加 |

这些参数定义 Reset 基线和求解时序，不适合在 candidate 验收中途热切换。GUI 的
`Material and contact setup details` 只读显示它们；要改默认值，在
`examples/embodied_environments/super_embodied/super_embodied.py` 的
`PAPER_CONSTRAINT_*`、`PAPER_SOFT_TISSUE_*` 集中配置区修改并重启。

## 6. 接触参数：夹爪什么时候碰到、每次推出多少

材料软硬和接触强弱不是一回事。对一个工具采样点，求解器先算它到组织表面的有符号距离
`d`。若 `d` 小于 contact margin `m`，就提出法向修正。省略质量和四面体安全缩放后，可近似
理解为：

\[
\Delta x_n\approx
\operatorname{clip}\left(\rho(m-d)_+,0,\Delta x_{max}\right),
\]

其中 `rho` 是 contact relaxation，`Delta x_max` 是 correction cap。真实代码还会把可能造成
四面体低于 minimum `J=V/V0` 的修正缩小或拒绝。

摩擦满足类似库仑上限：

\[
\|\Delta x_t\|\leq \mu\|\Delta x_n\|.
\]

`mu` 越大，夹爪沿组织表面越不容易滑；但摩擦只在接触存在时生效，不会隔空吸住组织。

接触/抓取 GUI 默认显示三个几何/范围旋钮，并在 `Particle jump stabilization` 中提供四个
专门抑制跳点的旋钮：

| GUI 参数 | 启动值 | 变大 | 变小/风险 |
|---|---:|---|---|
| Tip entry allowance | `3.0 mm` | 允许夹爪尖端进入上表面下方更深 | 必须小于 top-barrier distal length；启动时隐藏末端为 `3.01 mm` |
| Capture max penetration | `2.8 mm`（GUI `0.1..10 mm`） | 更深的双侧接触仍允许建立持续抓取 | 太大会把错误穿透状态锁住 |
| Grip support radius | `4.0 mm`（GUI `0..30 mm`） | 夹爪带动更宽的表面邻域 | 更局部；太小可能更容易滑脱 |
| Grip correction cap | `0.10 mm/substep`（GUI `0.005..2.0`） | 锚点追夹爪更快 | 调低是抑制孤立粒子跳动的第一步 |
| Grip compliance | `0.05 m/N`（GUI `0..2`） | 绑定更软、闭合冲击更小 | `0` 近似硬绑定，最容易突然猛拉 |
| Material/grip velocity transfer | `0.12`（GUI `0..1`） | 更多位置修正成为下一步速度 | 调低能直接削弱跳点的速度正反馈 |
| Particle velocity damping | `10 /s`（GUI `0..100`） | 全局更快消除速度振荡 | 太高会显得黏、恢复变慢 |

GUI 中 `Tip entry allowance` 会在后台同步扩展隐藏的 top-barrier distal length，但不会超过
`5.0 mm` jaw contact distal length；因此当前可调到约 `4.99 mm`，同时始终保持
`tip < barrier <= jaw distal`，不会通过取消几何校验换取范围。

下面这些仍是接触算法的重要启动参数，但不暴露为 GUI 滑条：

| 启动参数 | 当前值 | 变大 | 变小/风险 |
|---|---:|---|---|
| Contact margin | `0.40 mm` | 更早开始分离，表面间隙更大 | 太小更容易可见穿透 |
| Surface correction cap | `0.030 mm/pass` | 每次消除穿透更快 | 太大会踢动、鼓包或摆动 |
| Top barrier correction cap | `0.008 mm/pass` | 更强阻止从上表面穿下去 | `0` 表示继承 surface cap，不是关闭 |
| Contact relaxation | `1.0` | 当前为完整有界修正 | 小一些更柔和，但更容易残留穿透 |
| Jaw friction coefficient | `1.5` | 更不易沿夹爪滑动 | 太大可能拖拽过强，太小夹不稳 |

未暴露参数中最容易混淆的是 `Contact query distance=10 mm`。它只控制宽相搜索：“最多去
多远寻找可能相交的三角形”；最终是否产生修正仍由 `d<m` 决定。因此 query distance 为
`10 mm` **不等于会推动 10 mm 外的组织**。若 1 cm 外也明显移动，应依次检查材料 distance/
shape、material iterations、contact spread layers、grip support radius/generations，而不是先
把 query distance 降低。

| 高级接触参数 | 启动值 | 变大 | 变小 |
|---|---:|---|---|
| Tool sample spacing | `0.65 mm` | 采样更稀，速度快但可能漏碰撞 | 更密、更平滑但 GPU 更慢 |
| Contact query distance | `10 mm` | 只扩大搜索并增加开销 | 太小会漏掉高速/跨步接触 |
| CCD velocity scale | `1.0` | 沿工具运动方向预看更远 | 太小更易高速穿透 |
| Contact spread layers | `0` | 把瞬时接触分给更远图邻域 | `0` 最局部；过大正是远端位移来源之一 |
| Contact iterations | `1` | 接触更硬、收敛更强、耗时增加 | 更柔和，也可能穿透 |
| Post-contact material passes | `0` | 接触后更快恢复材料形状，也会扩散位移 | `0` 最不容易在同一子步向远处传播 |
| Contact substep stride | `1` | 数值表示每隔几子步求一次；值大更省时 | `1` 是每个子步都求接触 |
| Contact minimum J | `0.03` | 更早阻止局部压扁 | 太低会允许坏四面体；`0` 仅供备用模式关闭此门 |
| Contact velocity transfer | `0.35` | 更多位置修正变成速度，推动更明显 | 更不易弹跳，但可能显得黏滞 |
| Jaw/top distal length | `5/3 mm` | 更多夹爪长度参与接触/barrier | 太短可能只剩尖端工作 |

## 7. 持续抓取参数：什么时候锁住、带动多大范围

普通接触只负责“不能穿过去”。持续抓取是在检测到可靠双侧夹持后，为每侧选两个表面点，
一共四个 direct anchor，把它们的局部坐标记在夹爪坐标系中。抓取只有同时满足以下近似条件
并持续 `activation steps` 才建立：

\[
n_L,n_R\geq N,\quad s_{LR}\leq s_{max},\quad
p\leq p_{max},\quad J\geq J_{capture},\quad
q_7\leq q_{closed}.
\]

也就是：左右夹爪都要有足够采样点、两侧接触片不能相距太远、穿透不能过深、局部体积
不能已经坏掉，而且夹爪确实处于闭合阶段。这不是“只要 q7 闭合就吸住”。

anchor 使用带柔顺度的 XPBD 位置约束：

\[
\widetilde\alpha_{grip}=\frac{c_{grip}}{\Delta t_{sub}^2},
\qquad
\Delta x_{grip}\propto
\frac{-C-\widetilde\alpha_{grip}\lambda}
{w+\widetilde\alpha_{grip}},
\]

并再次被 `grip correction cap` 和 minimum `J` 限制。因此 grip compliance `c_grip` 越大，
分母越大，绑定越软；`0` 才接近刚性跟随。

四个 direct anchor 周围的 support 不是硬固定。半径 `R` 内节点按近似高斯权重接收衰减后的
目标：

\[
w(r)=\exp\left(-\frac{r^2}{2\sigma^2}\right),\qquad r\leq R.
\]

所以 support radius 和 generations 才是“夹取时周围粒子带动范围”的主要旋钮。

| 抓取参数 | 启动值 | 变大 | 变小/风险 |
|---|---:|---|---|
| Enable persistent grip | 开 | 建立四点持续绑定 | 关闭后只剩普通接触/摩擦 |
| Grip support radius | `4.0 mm` | 带动范围更宽、较稳 | 更局部，也更易滑脱 |
| Grip support generations | `1` | 继续向邻域迭代传播 | `1` 保持单层；过大易变成宽硬块 |
| Grip compliance | `0.05 m/N` | 绑定更软、减少闭合瞬间抽动 | 越小越硬；`0` 最容易猛拉 |
| Grip correction cap | `0.10 mm/substep` | 更快追随夹爪 | 太大产生尖峰，太小会明显滞后 |
| Maximum patch separation | `6.0 mm` | 更容易接受宽的双侧接触 | 太小经常捕获失败 |
| Capture max penetration（GUI） | `2.8 mm` | 深接触也可捕获 | 太大可能把错误穿透锁住 |
| Minimum samples per jaw | `8` | 捕获更严格、抗孤立噪声 | 太小容易误抓 |
| Activation contact steps | `3` | 要求接触稳定更久 | 太小会一碰就锁 |
| Grip closed angle | `0.13 rad` | 更早允许闭合捕获 | 必须小于 release angle |
| Grip release angle | `0.15 rad` | 需要张开更多才开始释放 | 太接近 closed 会在阈值附近抖动 |
| Grip capture minimum J | `0.20` | 更严格拒绝压坏状态 | 太低会在坏单元上建锚 |
| Grip minimum J | `0.03` | 持握运动更保体积 | 太高可能经常限幅，太低可能压塌 |
| Grip tetra transfer layers | `1` | 向内部更多四面体传力 | 过大使抓取区又深又硬 |

`release angle delta`、`wide-open angle` 和 `motion epsilon` 组成释放迟滞/方向判断。正常调参先
保持启动值；若只是“夹不住”，不要第一步改释放状态机，先看 GUI 的左右 contact count、
patch separation、capture allowed 和 maximum penetration。

判断问题来源时：

- 组织已接触但拉得太远：先看 distance/shape、material iterations、spread 和 support 范围；
- 夹爪从表面穿过：先看 margin/barrier/correction，不要只加材料刚度；
- 已夹住但抬不动：看 attached、safe scale、support radius、compliance 和 correction cap；
- 闭合时摆一下再夹住：先减 grip correction cap 或增加 compliance，再检查 contact velocity；
- 后侧鼓包：当前 pressure shoulder 已关闭，应看 volume、distance 传播、spread/support 范围。

## 8. GUI 正确操作顺序

1. 点 `Pause`。
2. 展开 `Online stiffness tuning (pause to apply)`。
3. 一次只改一类参数，例如先改范围，不要同时改 learning rate 和 EMA。
4. 点 `Apply stiffness settings`。
5. 看到 `Applied: verified k preserved/clipped; EMA and history cleared.`。
6. 再点 `Play`，观察至少一段完整压下—闭合—抬起过程。

Apply 的行为：

- 不重置粒子位置；
- 保留已经 verified 的区域刚度，但把超出新上下界的值裁回范围；
- 清空旧 EMA、历史快照和上一轮 residual，避免新旧公式混用；
- 加入 3 次更新冷却；
- 不修改固定 `k_volume=1e10`。

`Restore startup values` 只把启动值装入草稿，还需要再点 Apply。程序内点 `Reset` 会把材料场
回到 `0.20/0.004` 并清除学习结果，但保留本次运行已经 Apply 的调参规则。退出并重新启动后，
GUI 参数回到源码/CLI 的启动默认值。

接触/抓取使用同样的草稿流程：Pause 后展开 `Tip entry and grip (pause to apply)`；跳点相关
参数位于其中的 `Particle jump stabilization`。修改后点 `Apply entry/grip settings`。这次
Apply 会保留当前组织粒子位置和已验证材料参数，但会
重建工具采样/接触缓存并主动释放旧的持续抓取 anchor；否则旧 anchor 仍按旧半径和旧阈值
工作。随后清空旧接触阶段的刚度 EMA/历史并冷却 3 次更新。若当时已经夹住，需要继续播放并
重新满足双侧接触条件后才会再次显示 `GRASPED`。

`Restore startup entry/grip` 会恢复这七个草稿值，仍需 Apply。GUI 修改只在当前进程有效；重启后
回到 `super_embodied.py` 的启动默认值。

## 9. 精简后应该看哪几行

默认 GUI 只保留：

- `Distance/Shape min / median / max`：有没有形成区域差异、是否大量贴边；
- `Evidence active / harden / soften`：本轮有多少有效材料证据；
- `commit / reject`：candidate 是否真的被下一图像接受；
- `Grip`、左右接触数、gap、penetration：当前到底有没有抓住；
- residual 是否接受、loss 降幅和最大位置修正。

逐相机 loss、tet 屏蔽数、EMA、edge roughness、safe scale 等移入三个折叠区：
`Stiffness diagnostics`、`Contact diagnostics`、`Visual residual diagnostics`。平时不需要盯着
它们；只有出现“不更新、抓不住、突然变硬或局部异常”时再展开排查。

## 10. 推荐的保守调参顺序

如果区域差异仍不够：

1. 先观察是否贴上下限；只有贴边才扩大 bound。
2. 没贴边但变化太慢，再把 EMA new weight 从 `0.30` 调到 `0.35`。
3. 仍太慢，再把 learning rate 从 `0.18` 调到 `0.20`，不要直接跳很大。
4. 参数出现椒盐斑点，把 smoothing blend 从 `0.35` 调到 `0.45`；不要先加 passes。
5. 整体越来越硬，先把 hardening bias 从 `0.15` 降到 `0.10`。
6. commit 很少时先看 rejection reason；放宽学习率解决不了安全门拒绝。

每次只改一个旋钮，Reset 后用同一段轨迹 A/B，才能知道是哪一个参数产生了效果。
