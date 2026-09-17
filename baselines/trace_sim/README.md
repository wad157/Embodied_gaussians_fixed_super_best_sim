# TRACE SIM 基线

本适配器将官方 [vLAR-group/TRACE](https://github.com/vLAR-group/TRACE) 固定到
提交 `a4597585bc0e56c56922abe75be9198eb119c95a`，并接入现有 SIM
`joint_reconstruction_7to1_future_80to20` 协议。

## 基线边界

- 训练只读取前 80% 中合法的非留出双目 RGB 和标定相机元数据。
- 不读取 PSM 位姿或控制、夹持边界、深度、mask、GT 轨迹、测评查询点、留出 RGB 或未来 RGB。
- 官方 TRACE checkout 保持干净。适配层根据完整的 `fx`、`fy`、`cx`、`cy` 构造非对称投影矩阵，
  只补齐非中心主点相机支持，不修改 TRACE 算法。
- 关闭 TRACE 的可选 `--freegave` 扩展。
- Future 使用 TRACE 原生平移—旋转动力学，在
  `max_time=(future_start-1)/(frames-1)` 之外进行外推，不提供未来观测或控制。
- 冻结 checkpoint 后才读取 30 个测评查询点。锚点深度来自 TRACE 自身渲染；持久 Gaussian
  位移解码不使用 GT 深度、尺度拟合、刚体对齐或后验纠正。
- TRACE 表示并渲染完整场景，导出的 alpha 是完整不透明度而不是由 GT 得到的组织 mask。
  现有组织层图像指标为保持协议连续性而原样报告；解释结果时必须保留这一差异。

本机复用了已有官方兼容环境
`/Media_HDD/jwshan/conda_envs/freegave`（Python 3.7.16、PyTorch 1.13.1、CUDA 11.6），
避免重复安装约 13 GB 的依赖。

## 运行方法

运行 SIM-03 seed 0：

```bash
SIM_GPU_ID=0 bash scripts/run_trace_sim_baseline_once.sh \
  sim03 repeat_01 0 outputs/trace_sim03_v1/repeat_01
```

运行三数据集 × 三种子的双 GPU 调度：

```bash
bash scripts/run_trace_sim_all_repeats_scheduler.sh outputs/trace_sim_all_v1
```

调度器在每张 GPU 上使用串行队列，保留失败现场并最多重试三次；全部 9 次实验通过各自的
协议审计和评测后，生成 `summary/summary.{json,md}`。

## 正式结果

9 次运行全部完成。数值为算术均值 ± 总体标准差，没有挑选或丢弃运行。

| 数据集 | 分区 | 3D 均值 (mm) ↓ | 2D 均值 (px) ↓ | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---:|---:|---:|---:|---:|
| SIM-01 | Reconstruction 7:1 | 8.034 ± 1.303 | 3.471 ± 0.377 | 9.685 ± 0.014 | 0.3209 ± 0.0002 | 0.3822 ± 0.0007 |
| SIM-01 | Future 80:20 | 8.924 ± 1.488 | 23.330 ± 2.861 | 9.333 ± 0.042 | 0.2236 ± 0.0008 | 0.4730 ± 0.0034 |
| SIM-02 | Reconstruction 7:1 | 12.574 ± 4.751 | 4.626 ± 1.396 | 9.514 ± 0.030 | 0.3002 ± 0.0003 | 0.3871 ± 0.0015 |
| SIM-02 | Future 80:20 | 12.282 ± 3.922 | 23.322 ± 1.440 | 9.260 ± 0.031 | 0.2087 ± 0.0018 | 0.4793 ± 0.0040 |
| SIM-03 | Reconstruction 7:1 | 12.924 ± 2.547 | 9.664 ± 1.936 | 13.754 ± 0.009 | 0.3444 ± 0.0009 | 0.3359 ± 0.0013 |
| SIM-03 | Future 80:20 | 14.108 ± 2.436 | 27.290 ± 3.407 | 12.392 ± 0.513 | 0.2370 ± 0.0125 | 0.4513 ± 0.0204 |

发布的汇总见 [可读结果](../../outputs/trace_sim_all_v1/summary/summary.md) 和
[机器可读结果](../../outputs/trace_sim_all_v1/summary/summary.json)。JSON 还保留完整场景
PSNR/SSIM 及每项指标的三次原始数值。
