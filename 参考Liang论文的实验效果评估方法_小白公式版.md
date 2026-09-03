# 参考 Liang 等人论文的实验效果评估方法：小白公式版与本项目迁移指南

本文面向第一次接触“软组织仿真评估”的读者，详细解释论文
[Real-to-Sim Deformable Object Manipulation: Optimizing Physics Models with Residual Mappings for Robotic Surgery](https://arxiv.org/abs/2309.11656)
如何判断 residual mapping 和在线刚度优化是否有效，以及怎样把它的**评估思路**迁移到当前
`Embodied_gaussians_fixed_super_best` 项目。

本文参考的是论文 2024-05-29 的 arXiv v2：

- [论文摘要页](https://arxiv.org/abs/2309.11656)
- [论文 HTML 全文](https://arxiv.org/html/2309.11656v2)
- [论文 PDF](https://arxiv.org/pdf/2309.11656)

> [!IMPORTANT]
> 本项目不是要严格复现论文。论文使用 3D 点云 Chamfer distance 和可微 PBD；本项目使用
> 双目 2D RGB residual、Gaussian 渲染、启发式局部刚度信号、candidate/verified 双缓冲和
> 影子 rollout 安全验收。我们借鉴的是论文“怎样证明方法有效”的实验逻辑，而不是照抄它的
> 损失尺度、学习率、网格规模或材料参数。

---

## 1. 先记住最重要的一句话

一张模拟画面和真实图像很像，不等于材料刚度已经学对。

可以把它想成一个学生做数学题：

- residual mapping 像老师在当前答案上直接帮他改错；
- stiffness optimization 像学生真正理解了规律，以后遇到新题也能自己做对。

所以实验必须分别回答两个问题：

1. **当前状态问题：**老师改完以后，这一帧是否更接近真实观测？
2. **未来预测问题：**不再让老师改，仿真自己往前运行时是否仍然更准确？

论文的关键贡献不是只把当前帧贴准，而是检查更新后的材料参数能否改善未来开放环预测。
本项目也应把“未来预测”作为刚度有效性的主要证据。

---

## 2. 看懂公式前只需要四个数学概念

### 2.1 一个三维位置

一个粒子的位置可以写成：

\[
x_i=(x_i^x,x_i^y,x_i^z).
\]

例如：

\[
x_i=(10,20,5)\ \mathrm{mm}
\]

表示它在三个方向上的坐标分别是 `10 mm`、`20 mm` 和 `5 mm`。

### 2.2 两个位置之间的距离

预测位置为 \(a\)，真实位置为 \(b\)，它们的欧氏距离是：

\[
\|a-b\|_2
=
\sqrt{(a_x-b_x)^2+(a_y-b_y)^2+(a_z-b_z)^2}.
\]

假设：

\[
a=(0,0,0)\ \mathrm{mm},\qquad b=(3,4,0)\ \mathrm{mm},
\]

那么：

\[
\|a-b\|_2=\sqrt{3^2+4^2}=5\ \mathrm{mm}.
\]

### 2.3 平均值

如果未来三帧的误差分别为 `1 mm`、`2 mm`、`3 mm`，平均误差是：

\[
\frac{1+2+3}{3}=2\ \mathrm{mm}.
\]

平均值回答的是“整体上通常有多差”，但可能被极端值影响，所以本项目还应该同时报告中位数、
四分位数和最大值。

### 2.4 开放环预测

“开放环”不是把系统关掉，而是：

1. 在时刻 \(t\) 保存当前仿真状态；
2. 继续输入已经记录好的未来器械轨迹；
3. 暂时关闭视觉 residual 和在线参数更新；
4. 让物理模型独立预测未来；
5. 到未来真实帧再比较预测和观测。

这就像先遮住答案，让模型独立答题，最后才揭晓真实结果。

---

## 3. 论文先做了什么实验

论文使用 da Vinci Research Kit 操作真实鸡皮和鸡胸肌：

- 鸡皮代表薄壳组织，厚度约 `3 mm`；
- 鸡胸肌代表体组织，厚度约 `2 cm`；
- 一侧用金属针固定；
- 使用 720p 双目内窥镜；
- RAFT-Stereo 估计深度；
- Segment Anything 分割组织；
- 将双目深度反投影为 3D 点云；
- 点云降采样到约 9,000 个点；
- 仿真网格重建到约 600 个粒子；
- 共采集四条真实操作轨迹：两条薄壳、两条体组织。

这些数字描述论文自己的实验平台，不是本项目必须复制的设置。

论文把方法分成几组：

| 论文组别 | residual mapping | 在线刚度 | 这组回答什么问题 |
|---|---:|---:|---|
| PBD | 关闭 | 关闭 | 原始物理基线有多差 |
| PBD-RM | 开启 | 关闭 | 只修正当前状态是否有帮助 |
| PBD-RM-ON | 开启 | 开启 | 在线刚度是否进一步改善未来预测 |
| PBD-DiffCloud | 离线参数优化 | 离线 | 薄壳实验中的已有方法对照 |

最关键的比较不是 `PBD-RM-ON` 对 `PBD`，而是：

\[
\boxed{\text{PBD-RM-ON 对比 PBD-RM}}
\]

因为这两组都有 residual mapping，主要差别才是“有没有在线更新刚度”。

---

## 4. 论文怎样评价当前帧是否对齐

### 4.1 为什么不能直接逐点相减

真实点云和仿真表面通常没有一一对应关系：

- 仿真中的第 10 个点，不一定对应真实点云中的第 10 个点；
- 两边点数也可能不同；
- 相机只能看到表面，看不到体组织内部粒子。

论文因此使用 Chamfer distance。

### 4.2 Chamfer distance 公式

仿真表面点集合记为 \(X\)，真实观测点集合记为 \(Z\)。论文写出的双向 Chamfer distance 是：

\[
\mathcal D(X,Z)
=
\sum_{x\in X}\min_{z\in Z}\|x-z\|_2^2
+
\sum_{z\in Z}\min_{x\in X}\|z-x\|_2^2.
\]

它分成两个方向。

第一个方向：

\[
\sum_{x\in X}\min_{z\in Z}\|x-z\|_2^2
\]

意思是：每一个仿真点都去寻找离自己最近的真实点。

第二个方向：

\[
\sum_{z\in Z}\min_{x\in X}\|z-x\|_2^2
\]

意思是：每一个真实点也去寻找离自己最近的仿真点。

为什么必须双向？假设仿真只生成真实组织的一小部分：仿真点都能找到附近真实点，第一个方向
可能很小；但很多真实点没有仿真点覆盖，第二个方向就会暴露缺失区域。

### 4.3 一个极简数字例子

为了方便，只看一维：

```text
仿真点 X = {0, 4}
真实点 Z = {1, 3}
```

仿真到真实：

```text
0 最近的是 1，平方距离 = 1
4 最近的是 3，平方距离 = 1
合计 = 2
```

真实到仿真：

```text
1 最近的是 0，平方距离 = 1
3 最近的是 4，平方距离 = 1
合计 = 2
```

所以未归一化双向 Chamfer 值为：

\[
\mathcal D=2+2=4.
\]

### 4.4 论文用它评价 residual mapping

论文比较：

```text
PBD：只使用物理预测
PBD-RM：物理预测后，再用 residual mapping 修正
```

论文报告四条轨迹上的平均对齐误差从：

```text
薄壳 1：1.84 -> 0.001 mm
薄壳 2：2.55 -> 1.31  mm
体组织 1：4.24 -> 1.78 mm
体组织 2：5.79 -> 1.93 mm
```

这里证明的是：residual mapping 可以让**当前状态**更贴近观测。

> [!CAUTION]
> 论文公式写的是平方距离求和，但结果表以 mm 表示；实际实现通常还会进行平均、开根号或其他
> 归一化。正文没有把最终报告单位的全部细节写完整。因此迁移到本项目时，必须固定并公开我们
> 自己的 loss 定义，不能把本项目的 RGB loss 数字直接与论文的 mm 数字比较。

---

## 5. residual mapping 为什么还能成为刚度线索

论文先由物理模型预测：

\[
x_t=\mathcal{PBD}(x_{t-1},u_t,\Gamma,C,k_c).
\]

符号含义：

- \(x_{t-1}\)：上一时刻组织状态；
- \(u_t\)：当前器械控制；
- \(\Gamma\)：固定边界；
- \(C\)：距离、体积、形状等约束；
- \(k_c\)：约束刚度；
- \(x_t\)：物理模型预测的当前状态。

然后 residual mapping 求一个修正 \(\Delta_t\)：

\[
\Delta_t
=
\arg\min_\Delta
\left[
\mathcal D(x_t+\Delta,z_t)
+
\mathcal E(x_t+\Delta)
\right].
\]

其中：

- \(\mathcal D\) 让表面靠近真实点云；
- \(\mathcal E\) 使用物理约束，防止网格为了贴图而乱折、乱缩；
- \(z_t\) 是真实观测。

修正完成后：

\[
x_t\leftarrow x_t+\Delta_t.
\]

如果物理模型已经很准，所需修正 \(\Delta_t\) 应该比较小。如果每帧都需要很大的 \(\Delta_t\)，
说明物理模型或参数与真实组织不一致。因此论文定义：

\[
\mathcal L_{gap}=\|g(x_t,z_t)\|=\|\Delta_t\|.
\]

这里的直观意思是：

```text
需要老师改很多   -> 模型自己预测得不好
需要老师改很少   -> 模型自己预测得比较好
```

论文进一步通过可微 PBD 把这个误差反向传播给刚度参数：

\[
\frac{\partial \mathcal L_{gap}}{\partial k}
=
\frac{\partial\|g\|}{\partial x_t}
\frac{\partial\mathcal{PBD}}{\partial k_c}
\frac{\partial k_c}{\partial k}.
\]

这条链表示：

1. 刚度 \(k\) 改变约束刚度 \(k_c\)；
2. 约束刚度改变 PBD 预测 \(x_t\)；
3. PBD 预测改变所需 residual；
4. residual 变大还是变小，反过来告诉优化器刚度该怎么改。

本项目没有照搬这条端到端梯度，而是用 accepted residual 的方向生成有界 candidate，再通过
真实下一帧影子 rollout 检验 candidate 是否有预测价值。评估时仍然可以借用论文的核心原则：
**好的材料参数应该让未来所需 residual 更小。**

---

## 6. 论文为什么还要检查历史状态

只看当前一帧容易追随噪声。某一帧深度偏了 `1 mm`，优化器可能错误地把局部组织判成更软或
更硬。

论文从最近 20 帧中均匀取 4 个历史快照，定义：

\[
\mathcal L_{hist}
=
\sum_{h\in H_t}
\left\|
x_h+\Delta_h
-
\mathcal{PBD}(x_h+\Delta_h,0,C,k_c)
\right\|.
\]

理解这条公式可以分四步：

1. \(x_h+\Delta_h\) 是历史上已经和观测对齐的状态；
2. 把新的器械控制设成 0；
3. 使用当前候选刚度再运行一次 PBD；
4. 看它是否会从已经对齐的位置大幅漂走。

如果原状态是 \(q_h\)，松弛后变成 \(q_h^{relaxed}\)，漂移量可以简单理解为：

\[
d_h=\|q_h-q_h^{relaxed}\|.
\]

四个历史状态的平均漂移越小，说明候选刚度越能保持过去已经成立的状态。

### 数字例子

假设旧刚度在四个历史快照上的漂移是：

```text
0.10, 0.12, 0.11, 0.09 mm
```

平均值为：

\[
L_{hist}^{old}=0.105\ \mathrm{mm}.
\]

候选刚度的漂移是：

```text
0.11, 0.13, 0.12, 0.10 mm
```

平均值为：

\[
L_{hist}^{candidate}=0.115\ \mathrm{mm}.
\]

候选略差，但是否拒绝还要由预先规定的容差决定。本项目当前使用自己的质量加权 RMS 和
`baseline + 10% + 0.01 mm` 容差，不需要照抄论文未明确给出的阈值。

---

## 7. 论文为什么平滑空间刚度

组织相邻区域的材料通常不会毫无原因地从“非常软”瞬间跳成“非常硬”。论文因此惩罚相邻
粒子的刚度差异，概念上可写成：

\[
\mathcal L_{smooth}
=
\frac{1}{|E|}
\sum_{(i,j)\in E}(k_i-k_j)^2.
\]

其中 \(E\) 是相邻粒子边集合。

假设三个相邻节点刚度为：

```text
0.2, 0.21, 1.8
```

第三个节点与前两个差距极大，平滑损失会很大。优化器会倾向把它变成更连续的空间分布。

> [!NOTE]
> 论文 HTML 中公式（9）的绝对值或平方形式存在排版含糊，但正文意图明确：惩罚相邻节点刚度
> 差异。本项目使用物理边图上的邻域混合和 edge roughness 指标，不必照抄这条排版含糊的式子。

---

## 8. 论文怎样评价未来预测

### 8.1 这是整篇论文最值得迁移的评估思想

在时刻 \(t\)，我们知道当前状态和接下来真实执行的器械控制。先把未来观测藏起来，让仿真独立
预测：

\[
\hat x_{t+1},\hat x_{t+2},\ldots,\hat x_{t+T}.
\]

然后再取出未来真实观测 \(z_{t+1},\ldots,z_{t+T}\)，计算每一步预测还需要多大的 residual 才能
重新对齐。

论文定义平均未来 gap：

\[
e_t
=
\frac{1}{T}
\sum_{s=1}^{T}
\|g(\hat x_{t+s},z_{t+s})\|.
\]

论文使用：

\[
T=10.
\]

注意：这里不是在 rollout 过程中使用未来观测修正状态。未来观测只在预测完成后用于打分。

### 8.2 数字例子

为了简化，假设只预测三步：

```text
第 1 步重新对齐需要 residual RMS = 0.4 mm
第 2 步重新对齐需要 residual RMS = 0.7 mm
第 3 步重新对齐需要 residual RMS = 1.0 mm
```

那么：

\[
e_t=\frac{0.4+0.7+1.0}{3}=0.7\ \mathrm{mm}.
\]

另一个刚度模型的三步 residual 是：

```text
0.3, 0.4, 0.5 mm
```

它的未来 gap 为：

\[
e_t^{new}=0.4\ \mathrm{mm}.
\]

未来 gap 改善量为：

\[
\Delta e_t=e_t^{old}-e_t^{new}=0.7-0.4=0.3\ \mathrm{mm}.
\]

因为 \(\Delta e_t>0\)，新刚度预测更好。

### 8.3 为什么不只看当前 residual

一个非常软的模型可能先预测错，再被 residual 强行拉回正确位置。当前帧看起来很好，但下一帧
它又会自己跑错。未来开放环会暴露这种问题，因为 rollout 时没有 residual 帮忙。

因此：

\[
\boxed{\text{当前 residual 小，不自动等于未来预测好}}
\]

---

## 9. 论文怎样使用关键点做独立检查

Chamfer/residual 与优化目标关系很近。为了避免只用同一种误差“自己证明自己”，论文还标注
组织关键点。

论文每条轨迹选择 15 个形变区域中的可辨认关键点，每 10 帧标注一次。设：

- \(p_0\)：关键点初始位置；
- \(p_{t+T}\)：未来真实关键点位置；
- \(x_0^{nn}\)：初始时离关键点最近的仿真粒子；
- \(x_{t+T}^{nn}\)：该仿真粒子未来的位置。

论文用最近粒子的位移近似关键点位移：

\[
\Delta p
\approx
x_{t+T}^{nn}-x_0^{nn}.
\]

预测关键点为：

\[
\hat p_{t+T}=p_0+\Delta p.
\]

关键点误差为：

\[
f_t
=
\sum_p
\|p_{t+T}-\hat p_{t+T}\|_2.
\]

### 数字例子

初始关键点：

\[
p_0=(10,10)\ \mathrm{px}.
\]

它绑定的仿真点预测向右移动 `3 px`、向上移动 `2 px`：

\[
\Delta p=(3,2)\ \mathrm{px}.
\]

预测关键点：

\[
\hat p=(13,12)\ \mathrm{px}.
\]

真实关键点是：

\[
p^{obs}=(14,10)\ \mathrm{px}.
\]

误差为：

\[
\|p^{obs}-\hat p\|_2
=
\sqrt{(14-13)^2+(10-12)^2}
=
\sqrt5
\approx2.24\ \mathrm{px}.
\]

关键点误差不要求整张图颜色完全一致，更直接地测量有意义组织位置是否预测正确。

---

## 10. 论文怎样检查初始值敏感性

优化算法可能只在某个幸运初值上成功。论文为此使用三组均匀初始刚度，并对每组都比较不同
方法。

它想回答：

```text
如果一开始猜得很硬，方法还能修正吗？
如果一开始猜得很软，方法还能修正吗？
换一个初值以后，未来预测改善是否仍然存在？
```

论文固定 volume stiffness 为：

\[
k_{vol}=10^{10},
\]

优化 distance 和 shape stiffness。

> [!CAUTION]
> 论文正文同时写了 shape 范围 `[0, 0.02]` 和超出该范围的 `0.1/0.15` 初值。这在文字上互相
> 矛盾，可能是排版、参数域或版本问题。迁移时应借鉴“多初值测试”，不要照搬这些冲突数字。

---

## 11. 论文结果能够证明什么，不能证明什么

### 能够支持的结论

- residual mapping 明显降低当前观测对齐误差；
- 加入在线刚度以后，多数轨迹和初值上的未来 gap、关键点误差进一步下降；
- 在线更新比只在轨迹开头离线拟合更不容易只适应开头一小段；
- 空间刚度可以随位置和时间变化。

### 不能直接支持的结论

- 优化出来的每个局部刚度就是真实材料杨氏模量；
- 方法已经实时；
- 对所有组织和所有操作都泛化；
- 结果具有充分的跨次统计显著性。

论文自己也说明：薄壳和体组织每步分别约需 `0.9 s` 和 `2.6 s`，感知每张图还需约 `0.4 s`；
正文也没有报告系统性的跨次重复均值、方差或置信区间。

---

## 12. 当前项目与论文的关键差异

| 环节 | Liang 等人的论文 | 当前项目 | 对评估的影响 |
|---|---|---|---|
| 真实观测 | 双目重建 3D 点云 | 左右目 2D RGB + mask | 不能直接复用论文 Chamfer 数值 |
| 当前对齐损失 | 3D Chamfer | masked Smooth-L1 RGB | 本项目 loss 无直接 mm 含义 |
| residual 变量 | 物理粒子 \(\Delta_t\) | 物理节点 accepted residual \(r_i\) | 思想对应，优化器不同 |
| 刚度更新 | 可微 PBD 反向传播 | `d/r` 方向信号 + EMA + log-space candidate | 不是严格复现 |
| volume | 固定 \(10^{10}\) | 固定 \(10^{10}\) | 可以保持相同实验控制原则 |
| 参数写入 | 在线梯度更新 | candidate 通过验证才 commit | 本项目安全性更强 |
| 历史 | 最近 20 帧均匀抽 4 帧 | 最多 4 个准静态完整快照 | 本项目更适合抓取状态 |
| 未来窗口 | 固定 \(T=10\) | \(H=1,3,5,10\) | 本项目能看误差随时间增长 |
| 接触/抓取 | 点式位置控制 | direct/support、双侧抓取状态机 | 需要材料隔离协议 |
| 关键点 | 15 个人工点 | 已有 tracklet 资产，未接入当前 evaluator | 这是当前主要缺口之一 |

因此本项目的方法应描述为：

> 受 Liang 等人启发的 residual-driven 在线刚度修正，并通过历史门、candidate/verified 双缓冲、
> 下一观测影子 rollout 和物理安全门决定是否提交。

---

## 13. 本项目当前的图像误差是什么

本项目没有把双目图像在线重建成 3D Chamfer 点云，而是在左右目分别计算 masked Smooth-L1
RGB loss。

单个像素颜色误差记为 \(e\)，Smooth-L1 为：

\[
\rho_\beta(e)
=
\begin{cases}
\dfrac{e^2}{2\beta}, & |e|<\beta,\\[6pt]
|e|-\dfrac{\beta}{2}, & |e|\geq\beta.
\end{cases}
\]

当前 \(\beta=0.05\)。小误差使用平方形式，优化更平滑；大误差近似线性，减少异常高亮或遮挡
像素的破坏。

相机 \(c\) 的 loss 为：

\[
D_c
=
\frac{
\sum_p W_c(p)\rho_\beta(I_c^{sim}(p)-I_c^{obs}(p))
}{
\sum_p W_c(p)+\varepsilon
}.
\]

双目平均：

\[
D_{vis}=\frac{1}{|\mathcal C|}\sum_{c\in\mathcal C}D_c.
\]

其中 \(W_c(p)\) 包含组织 mask、器械遮挡、图像边缘和夹爪附近权重。

两个重要后果：

1. 左右目先各自归一化，再等权平均，不能让像素更多的一目完全支配结果；
2. \(D_{vis}\) 是颜色误差，不是毫米误差，不能和论文的 Chamfer mm 横向比较。

---

## 14. 把论文三组对照迁移到本项目

### 14.1 主实验三组

| 本项目组别 | `visual-feedback-mode` | 在线刚度 | 用途 |
|---|---|---:|---|
| A：Fixed PBD | `off` | 关闭 | 辅助物理基线 |
| B：Residual-only | `residual` | 关闭 | 最重要的刚度对照 |
| C：Full improved | `residual` | 开启 | 当前完整改进方法 |

判断 residual 是否有效：

\[
B\quad\text{对比}\quad A.
\]

判断在线刚度是否有效：

\[
\boxed{C\quad\text{对比}\quad B.}
\]

如果只比较 C 和 A，即使 C 更好，也分不清改善来自 residual 还是刚度。

### 14.2 可选安全机制消融

为了评价本项目相对论文增加的安全机制，可在**冻结离线快照**上增加：

| 消融 | 想回答的问题 |
|---|---|
| 去掉历史稳定门 | 历史门是否减少长期漂移和错误提交 |
| 去掉下一观测 candidate 验证 | 影子 rollout 是否过滤错误刚度 |
| 去掉 direct/support/一环排除 | 抓取隔离是否减少跳点和视觉拉扯 |
| 去掉空间平滑 | 平滑是否减少材料噪点，又没有抹掉区域差异 |
| 完整方法 | 所有改进共同运行的效果 |

这些消融可能降低安全性，应该在保存的 shadow state 上离线运行，不建议直接作为正常 GUI
抓取配置。

---

## 15. 本项目需要两层未来评估

### 15.1 第一层：单次 candidate 的因果比较

在同一个 accepted 状态 \(S_t\) 上运行两条影子：

\[
\hat q_{t+H}^{old}
=
\operatorname{Rollout}(S_t,k_{verified},u_{t:t+H}),
\]

\[
\hat q_{t+H}^{new}
=
\operatorname{Rollout}(S_t,k_{candidate},u_{t:t+H}).
\]

除了刚度不同，其他都必须相同：

- 初始 particle positions 和 velocities；
- 器械 `SE(3)+q7` 命令；
- 相机和图像；
- contact 参数；
- direct/support 绑定；
- residual、visual force 和新在线更新全部关闭。

这回答：

> 这一次 candidate 本身是否比旧刚度更会预测未来？

当前项目已经实现这条 old/new 同状态比较。

### 15.2 第二层：完整轨迹方法比较

分别从 Reset 运行：

```text
B：Residual-only，全程固定 k
C：Full improved，允许在线 candidate/commit/reject
```

这回答：

> 把所有更新积累起来以后，完整系统是否真的更好？

第一层控制变量最干净，但只评价单次局部更新；第二层包含在线方法积累、接触状态和误差传播，
更接近最终使用效果。正式结论需要两层同时成立。

---

## 16. 本项目的未来 Gap 公式

对于方法 \(m\)、起点 \(t\) 和 horizon \(H\)：

\[
Gap_m(t,H)
=
D_{vis}
\left(
\operatorname{Render}(\hat q^m_{t+H}),
I^{obs}_{t+H}
\right).
\]

当前使用：

\[
H\in\{1,3,5,10\}\ \text{个视频帧}.
\]

绝对改善量：

\[
\Delta Gap(t,H)
=
Gap_{baseline}(t,H)-Gap_{improved}(t,H).
\]

解释：

- \(\Delta Gap>0\)：改进方法更好；
- \(\Delta Gap=0\)：没有区别；
- \(\Delta Gap<0\)：改进方法更差。

相对改善率：

\[
R_{Gap}(t,H)
=
\frac{Gap_{baseline}(t,H)-Gap_{improved}(t,H)}
{Gap_{baseline}(t,H)+\varepsilon}
\times100\%.
\]

### 数字例子

如果 Residual-only 的 `H=10` loss 是 `0.040`，Full improved 是 `0.034`：

\[
\Delta Gap=0.040-0.034=0.006,
\]

\[
R_{Gap}=\frac{0.006}{0.040}\times100\%=15\%.
\]

可以说：在这个起点上，完整方法的十帧未来 RGB Gap 降低了 `15%`。

---

## 17. 两种 rollout 协议为什么都要保留

### 17.1 `material_isolation`

它保留快照中已有的 direct/support 绑定，继续播放相同器械轨迹，但冻结新的抓取状态变化。

目的：尽量只让刚度成为唯一变量。

它应该是评价在线刚度的**主要协议**。

### 17.2 `end_to_end`

它允许 contact、capture、support 和 release 状态机正常演化。

目的：检验最终系统运行时，刚度变化会不会和接触/抓取相互作用，导致完全不同的状态分支。

它应该是评价系统实际效果的**辅助协议**。

如果 `material_isolation` 改善、`end_to_end` 恶化，通常说明材料方向可能是对的，但接触或抓取
状态机对小位移过于敏感。此时不应简单地把问题归咎于刚度。

---

## 18. 把论文关键点误差迁移到本项目

本项目已有：

```text
scripts/build_super_tissue_keypoint_tracks.py
```

它使用 Shi-Tomasi 特征和前后向 LK 光流，在夹爪附近跟踪组织点，排除器械遮挡，并把关键点
关联到保存的表面 3D 点。

迁移时还需要完成三步：

1. 把关键点的初始表面 3D 点绑定到物理节点或视觉三角面重心坐标；
2. 开放环结束后，由预测物理节点重建关键点 3D 位置；
3. 投影到左右目，与真实 tracklet 比较。

左目或右目的 2D 关键点误差：

\[
E_{kp,c}(t,H)
=
\frac{1}{N_{valid}}
\sum_{j\in valid}
\left\|
\pi_c(\hat X_{j,t+H})-u^{obs}_{j,c,t+H}
\right\|_2.
\]

其中：

- \(\hat X_j\)：仿真预测的关键点 3D 位置；
- \(\pi_c\)：相机 \(c\) 的投影函数；
- \(u^{obs}_{j,c}\)：图像中真实跟踪点；
- `valid`：没有被遮挡、没有跟踪失败的点。

如果表面观测的 3D 关联可靠，还可以报告：

\[
E_{kp,3D}(t,H)
=
\frac{1}{N_{valid}}
\sum_j
\|\hat X_{j,t+H}-X^{obs}_{j,t+H}\|_2.
\]

建议把 2D pixel error 作为稳定主指标，把 3D mm error 作为深度质量通过门后的补充指标。

关键点必须只用于打分，不能再反过来修改 residual 或在线刚度。如果现有 Stage-D 材料标定曾经
使用过同一批关键点，还应划分互不重叠的训练帧和评测帧，或重新建立 held-out tracklets，避免
“用参与调参的数据评价调参效果”。

---

## 19. 物理安全指标及公式

视觉更像不能以组织翻转或穿透为代价。

### 19.1 四面体体积比

\[
J_e(t)=\frac{V_e(t)}{V_e^0}.
\]

解释：

```text
J = 1       体积与静止状态相同
0 < J < 1   四面体被压缩
J 很小      接近压扁，数值危险
J = 0       完全塌平
J < 0       四面体翻转
```

全局最小值：

\[
J_{min}(t)=\min_e J_e(t).
\]

翻转数：

\[
N_{inv}(t)=\sum_e\mathbf 1[J_e(t)\leq0].
\]

完整方法至少不能比 baseline 新增翻转。

### 19.2 接触穿透

如果采样点的有符号距离为 \(d_s\)，规定间隙为 \(m\)，穿透可写成：

\[
p_s=\max(0,m-d_s).
\]

报告：

- 最大穿透 \(p_{max}\)；
- 95% 分位穿透 \(p_{95}\)；
- 穿透采样数量。

### 19.3 抓取锚点误差

锚点实际位置为 \(x_a\)，夹爪目标位置为 \(u_a\)：

\[
E_{anchor,RMS}
=
\sqrt{
\frac{1}{N_a}
\sum_a\|x_a-u_a\|_2^2
}.
\]

它过大说明组织没有可靠跟随夹爪，也可能表示材料和抓取约束在互相对抗。

### 19.4 粒子跳动指标

粒子离散加速度近似为：

\[
a_{i,t}
\approx
\frac{x_{i,t+1}-2x_{i,t}+x_{i,t-1}}{\Delta t^2}.
\]

抓取邻域的跳动 RMS：

\[
E_{jitter}(t)
=
\sqrt{
\frac{1}{|G|}
\sum_{i\in G}\|a_{i,t}\|_2^2
}.
\]

其中 \(G\) 可以取 direct/support 外侧的一小圈动态节点。还应同时报告最大粒子速度和离群粒子
数量，避免平均值掩盖少数疯狂跳点。

---

## 20. 材料更新自身应该怎样评价

### 20.1 提交率

\[
R_{commit}
=
\frac{N_{commit}}{N_{candidate}}.
\]

提交率不是越高越好：

- 太高可能说明验收门过松；
- 太低可能说明信号噪声太大或门过于保守；
- 必须结合未来预测结果解释。

### 20.2 committed candidate 的未来胜率

\[
R_{win}(H)
=
\frac{
\#\{candidate:\Delta Gap(t,H)>0\}
}{N_{evaluated}}.
\]

如果很多 candidate 通过下一帧门，但到了 `H=10` 经常变差，说明当前提交门只会优化短期，
没有可靠筛选长期材料效果。

### 20.3 拒绝门是否过于保守

对于 rejected candidate，也可以在不写回 live 状态的情况下离线继续 rollout：

\[
R_{false\ reject}(H)
=
\frac{
\#\{rejected:\Delta Gap(t,H)>0\ \text{且物理安全}\}
}{N_{rejected}}.
\]

比例很高说明很多好 candidate 被拒绝；比例很低说明拒绝门确实过滤了坏更新。

### 20.4 刚度饱和比例

distance 下限饱和比例：

\[
R_{dist,min}
=
\frac{\#\{i:k_i^{dist}=k_{min}^{dist}\}}{N}.
\]

同理记录 distance 上限、shape 下限和 shape 上限。如果大面积长期贴着边界，通常说明：

- 搜索范围太小；或
- 信号存在系统偏置；或
- 当前模型只能靠极限参数补偿未建模因素。

### 20.5 空间粗糙度

\[
R_{edge}(k)
=
\frac{1}{|E|}
\sum_{(i,j)\in E}|k_i-k_j|.
\]

粗糙度很大表示材料场像“椒盐噪声”；粗糙度过小也不一定好，因为真实区域差异可能被完全抹掉。

---

## 21. 多初值实验怎样在本项目中设置

不要复制论文存在矛盾的 shape 数字。建议使用本项目当前安全范围中的三档：

| 本项目初始档 | distance | shape | volume | 用途 |
|---|---:|---:|---:|---|
| soft | 0.10 | 0.003 | \(10^{10}\) | 检查能否从偏软状态恢复 |
| default | 0.20 | 0.004 | \(10^{10}\) | 当前生产初值 |
| stiff | 1.00 | 0.012 | \(10^{10}\) | 检查能否从明显偏硬状态恢复 |

`2.00/0.020` 建议作为单独极限压力测试，而不是正常主表的硬档。

三档实验必须保持以下配置完全相同：

- 组织资产和初始位置；
- 相机参数和图像 mask；
- 器械完整轨迹；
- contact margin、friction、tip entry；
- capture penetration、support radius；
- material/contact velocity transfer；
- residual 更新间隔和学习率；
- 固定 volume stiffness；
- 评测帧和随机种子。

否则最后无法判断差异究竟来自刚度还是接触调参。

> [!NOTE]
> 当前项目还没有一个专门用于批量实验的“初始 distance/shape CLI 配置矩阵”。正式跑批时应
> 新增可记录到 metadata 的显式初值参数，不应靠操作者在 GUI 中手动拖滑条。

---

## 22. 为什么要预先固定评测起点

当前多 horizon evaluator 在 candidate 已经 commit 后才建立评测任务。这对评价“已提交更新的
后续表现”很有用，但会产生选择偏差：只有已经通过下一帧门的 candidate 才被纳入。

正式方法 A/B 还应预先规定统一起点，例如：

```text
在有效操作区间内每 10 个视频帧取一个起点
跳过序列末尾不足 H=10 的起点
保留起点所属 press/close/capture/lift/place/release 阶段
所有方法使用完全相同的起点集合
```

这就像考试前先公布所有考题位置，不能只挑自己已经做对的题统计。

建议同时保留两类结果：

1. `commit-triggered`：评价提交门选中的 candidate；
2. `scheduled-checkpoint`：公平比较完整方法 B/C。

---

## 23. 为什么要按动作阶段统计

同一个平均数可能掩盖完全不同的问题：

- press：主要考验局部压缩和 top barrier；
- close：主要考验 q7 横向接触；
- capture：主要考验状态切换和排除掩码；
- lift：主要考验 direct/support 与材料传播；
- place：主要考验卸载和回弹；
- release：主要考验锚点清理和恢复。

例如总体未来 Gap 改善 `5%`，可能实际上是：

```text
press    改善 20%
close    改善 10%
capture  恶化 30%
lift     改善 15%
place    持平
release  恶化 10%
```

如果不分阶段，就会错过 capture 阶段的严重问题。

当前 recorder 已有：

```text
press / close / capture / lift / place / release / idle
```

因此迁移时应直接复用阶段标签。

---

## 24. 重复性和统计应该怎样做

### 24.1 同一轨迹重复运行

同一配置至少重复 3 次，用来检查：

- CUDA 数值是否稳定；
- 视频/物理异步是否改变更新时序；
- 接触候选和抓取状态机是否发生分叉；
- commit/reject 序列是否可重复。

同一轨迹重复只能证明**重复性**，不能证明对其他组织或轨迹的泛化。

### 24.2 不同轨迹或区域

不同抓取点、不同运动方向或不同组织序列才用于评估泛化。如果当前只有一条完整轨迹，应诚实地
把结果称为“单轨迹 case study”，不能把每一帧当成一条独立实验来夸大样本量。

### 24.3 配对统计

同一个起点上的 B/C 使用相同状态和控制，所以应计算配对差值：

\[
d_j(H)=Gap_B(t_j,H)-Gap_C(t_j,H).
\]

报告：

- 平均值；
- 中位数；
- 25%/75% 分位数；
- 胜率 \(d_j>0\)；
- 以轨迹或连续时间块为单位的 95% bootstrap 置信区间。

连续视频帧高度相关，不应把上千帧当成上千个独立样本做普通独立样本显著性检验。

---

## 25. 性能指标怎样记录

论文公开报告了较慢的单步时间，因此本项目也应明确报告性能，而不是只写配置频率。

分开记录：

```text
physics frame wall time
12-substep XPBD wall time
双目渲染和 RGB loss 时间
8-iteration residual solve 时间
history baseline/candidate 时间
下一观测双影子验证时间
H=1/3/5/10 额外评测时间
GPU 峰值显存
完整轨迹总墙钟时间
```

实时因子定义为：

\[
RTF
=
\frac{\text{推进的仿真时间}}{\text{实际墙钟时间}}.
\]

例如仿真推进 `1 s`，实际花 `4 s`：

\[
RTF=\frac14=0.25.
\]

这表示比实时慢 4 倍。

要分别报告：

1. 普通 GUI，关闭多 horizon 评测；
2. 完整实验 evaluator 开启。

评测模式本来就需要大量影子路径，不能把它的 FPS 当作正常 GUI FPS。

---

## 26. 推荐的完整实验矩阵

设：

- 方法数：3，A/B/C；
- 初值数：3，soft/default/stiff；
- 轨迹数：\(N_{traj}\)；
- 重复次数：\(R=3\)。

完整轨迹运行数量为：

\[
N_{runs}=3\times3\times N_{traj}\times3.
\]

如果只有一条轨迹：

\[
N_{runs}=27.
\]

可以分阶段执行：

### 第一阶段：快速可行性

```text
1 条轨迹
default 初值
A/B/C 各 1 次
```

先确认输出、关键点、阶段和开放环都没有错误。

### 第二阶段：初值敏感性

```text
1 条轨迹
soft/default/stiff
B/C 各 1 次
```

先把主要比较集中在 Residual-only 与 Full improved。

### 第三阶段：正式实验

```text
所有轨迹
三种初值
A/B/C
每组至少 3 次
```

---

## 27. 最终表格应该长什么样

### 27.1 未来预测主表

| 方法 | 初值 | protocol | H=1 Gap | H=3 Gap | H=5 Gap | H=10 Gap | KP error |
|---|---|---|---:|---:|---:|---:|---:|
| Residual-only | soft | material isolation |  |  |  |  |  |
| Full improved | soft | material isolation |  |  |  |  |  |
| Residual-only | default | material isolation |  |  |  |  |  |
| Full improved | default | material isolation |  |  |  |  |  |
| Residual-only | stiff | material isolation |  |  |  |  |  |
| Full improved | stiff | material isolation |  |  |  |  |  |

每个格子建议写：

```text
mean ± std
median [p25, p75]
```

### 27.2 物理安全表

| 方法 | min J | inversion count | penetration p95/max | anchor RMS/max | jitter RMS | outlier particles |
|---|---:|---:|---:|---:|---:|---:|
| Residual-only |  |  |  |  |  |  |
| Full improved |  |  |  |  |  |  |

### 27.3 材料更新表

| 初值 | candidates | commits | rejects | H=10 win rate | dist saturation | shape saturation | edge roughness |
|---|---:|---:|---:|---:|---:|---:|---:|
| soft |  |  |  |  |  |  |  |
| default |  |  |  |  |  |  |  |
| stiff |  |  |  |  |  |  |  |

### 27.4 阶段表

| 阶段 | H=10 Gap improvement | KP improvement | min J change | penetration change | jitter change |
|---|---:|---:|---:|---:|---:|
| press |  |  |  |  |  |
| close |  |  |  |  |  |
| capture |  |  |  |  |  |
| lift |  |  |  |  |  |
| place |  |  |  |  |  |
| release |  |  |  |  |  |

---

## 28. 什么条件下可以说“当前改进有效”

最终阈值应在正式实验前固定，不能看完结果再挑有利规则。至少应满足：

1. Full improved 相对 Residual-only 在 `material_isolation` 的 `H=5/10` 上稳定降低 Gap；
2. 独立关键点误差同步改善，或至少没有显著退化；
3. soft/default/stiff 三种初值下改善方向基本一致；
4. 不新增翻转；
5. 最小 \(J\)、穿透、锚点误差不越过现有安全门；
6. 抓取邻域 jitter 和离群粒子没有增加；
7. commit 不是只在单一动作阶段有效；
8. 重复运行的结论方向一致；
9. 额外计算开销被完整报告。

如果只有当前 RGB Gap 改善，而关键点、物理质量或未来预测恶化，只能说“画面对齐目标下降”，
不能说“刚度优化有效”。

---

## 29. 当前项目哪些部分已经具备

### 已经实现

- accepted residual 前后的 `q_pred/q_acc/r` 分离；
- per-node distance/shape candidate；
- fixed volume stiffness；
- candidate/verified 双缓冲；
- 最多 4 个准静态历史快照的 baseline/candidate 松弛；
- 下一真实观测 old/new 双影子提交门；
- `H=1/3/5/10` 多 horizon；
- `material_isolation` 和 `end_to_end` 两种协议；
- 左右目 Gap、最小 \(J\)、翻转、穿透、锚点误差和 residual RMS；
- press/close/capture/lift/place/release 阶段分类；
- `metadata.json`、append-only `events.jsonl` 和 `summary.json`。

主要代码：

| 功能 | 文件 |
|---|---|
| 运行时和影子 rollout | `examples/example_embodied_super_offline.py` |
| 在线刚度 candidate | `src/embodied_gaussians/physics_simulator/online_tissue_stiffness.py` |
| 阶段和 JSON recorder | `src/embodied_gaussians/physics_simulator/stiffness_evaluation.py` |
| 关键点 tracklet | `scripts/build_super_tissue_keypoint_tracks.py` |
| 当前评测入口 | `scripts/run_super_stiffness_evaluation.sh` |

### 仍需补齐

1. A/B/C 三组的批量实验 runner；
2. 可复现、写入 metadata 的初始 distance/shape 参数；
3. 与 commit 无关的 scheduled checkpoint evaluator；
4. 将关键点 tracklet 接入每个 horizon；
5. 记录 jitter、离群粒子和分模块 CUDA wall time；
6. 按 `方法 × 初值 × 轨迹 × 重复 × protocol × horizon × 阶段` 聚合 JSONL；
7. 自动生成表格、曲线和置信区间；
8. 在有 CUDA 和显示服务的机器上跑完整轨迹。

当前仓库尚未发现真实完整刚度评测产生的 `outputs/stiffness_evaluation_*/events.jsonl` 和
`summary.json`。因此目前是“评测功能已有主体”，不是“实验结果已经产生”。

---

## 30. 当前入口怎么运行

当前完整方法可以运行：

```bash
STIFFNESS_EVALUATION_OUTPUT=outputs/stiffness_eval_full_default_r01 \
bash scripts/run_super_stiffness_evaluation.sh
```

它默认开启：

```text
tissue mode                 paper_pbd
visual feedback             residual
online stiffness            ON
visual update interval      3 physics frames
horizons                    1,3,5,10
```

输出：

```text
metadata.json    本次配置和来源信息
events.jsonl     每个事件一行，程序中断后仍可读取
summary.json     按动作阶段累计的 count/mean/min/max
```

> [!IMPORTANT]
> 这个入口只直接运行 Full improved，并且多 horizon 任务由 committed candidate 触发。它还不是
> 完整 A/B/C 批量实验器。Residual-only 和 Fixed PBD 的公平 scheduled-checkpoint 评测仍需补充。

---

## 31. 常见的错误结论

### 错误 1：当前帧 loss 降了，所以刚度学对了

不成立。residual 可以直接移动节点，当前帧变好可能完全不依赖刚度。

### 错误 2：Full 比完全关闭视觉的 PBD 好，所以在线刚度有效

不充分。改善也可能来自 residual。刚度的主要对照必须是 Residual-only。

### 错误 3：只统计 committed candidate

会产生选择偏差。还需要预先固定相同评测起点。

### 错误 4：把本项目 RGB loss 写成毫米

错误。RGB Smooth-L1 没有直接空间长度单位。毫米指标应来自可靠 3D 关键点或 residual 节点位移。

### 错误 5：把优化后的刚度热力图叫作真实组织刚度图

没有材料 ground truth 时，它只能被称为“有效仿真参数场”或“预测性刚度场”。

### 错误 6：A/B 两组顺手调了不同接触参数

这样无法归因。tip entry、support radius、velocity transfer、penetration limit 等必须冻结。

### 错误 7：把每一帧当作独立样本

连续视频高度相关。统计置信区间应按轨迹或连续时间块计算。

### 错误 8：只报告平均值

少量严重翻转、穿透或跳点可能被平均值掩盖。必须同时报告最大值、分位数和失败计数。

---

## 32. 给小白的最终总结

可以把完整实验想成三场考试。

### 第一场：老师改完以后像不像

比较 PBD 与 residual 后的当前视觉误差。

回答：

> 当前帧能不能对齐？

### 第二场：不让老师帮忙还能不能做对

从同一状态关闭 residual，预测未来 `H=1/3/5/10`，比较 Residual-only 和 Full improved。

回答：

> 在线刚度有没有真正提高物理预测能力？

### 第三场：做对的同时有没有把纸撕烂

检查最小 \(J\)、翻转、穿透、锚点误差、跳动、离群粒子和运行时间。

回答：

> 改善是否安全、稳定、可重复、值得计算开销？

只有三场都通过，才可以说：

> 当前改进后的在线刚度方案，不仅让画面更贴近观测，而且在关闭视觉帮助后仍能更准确地预测
> 未来组织形变，同时没有以物理不稳定和严重性能下降为代价。
