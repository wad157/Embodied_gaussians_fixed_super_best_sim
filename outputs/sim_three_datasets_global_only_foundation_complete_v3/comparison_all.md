# SIM-01/02/03 独立测评总表

三套数据分别生成RGB深度、CoTracker、夹持边界和评估清单；只在本表阶段汇总指标。

| 数据集 | 能力 | 方法 | 3D Mean mm↓ | 2D Mean px↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---:|---:|---:|---:|---:|
| SIM-01 Planar-X-Pull | 7:1重建 | A：纯 PBD | 1.5585 | 15.543 | 19.056 | 0.8381 | 0.4217 |
| SIM-01 Planar-X-Pull | 7:1重建 | B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度 | 0.4985 | 4.852 | 19.970 | 0.8469 | 0.4076 |
| SIM-01 Planar-X-Pull | 7:1重建 | C：B + 全局distance刚度与全局阻尼更新（无区域/逐粒子刚度） | 0.5007 | 4.907 | 19.970 | 0.8470 | 0.4078 |
| SIM-01 Planar-X-Pull | 80/20预测 | A：纯 PBD | 4.2814 | 42.833 | 16.456 | 0.8160 | 0.4512 |
| SIM-01 Planar-X-Pull | 80/20预测 | B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度 | 1.6501 | 16.091 | 18.412 | 0.8349 | 0.4210 |
| SIM-01 Planar-X-Pull | 80/20预测 | C：B + 全局distance刚度与全局阻尼更新（无区域/逐粒子刚度） | 1.4865 | 14.481 | 18.719 | 0.8379 | 0.4186 |
| SIM-02 Planar-Y-Pull | 7:1重建 | A：纯 PBD | 1.4335 | 13.424 | 16.425 | 0.8171 | 0.4295 |
| SIM-02 Planar-Y-Pull | 7:1重建 | B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度 | 0.3779 | 3.700 | 17.001 | 0.8267 | 0.4176 |
| SIM-02 Planar-Y-Pull | 7:1重建 | C：B + 全局distance刚度与全局阻尼更新（无区域/逐粒子刚度） | 0.3783 | 3.709 | 17.013 | 0.8266 | 0.4179 |
| SIM-02 Planar-Y-Pull | 80/20预测 | A：纯 PBD | 4.1152 | 37.595 | 15.476 | 0.7996 | 0.4537 |
| SIM-02 Planar-Y-Pull | 80/20预测 | B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度 | 1.3338 | 13.726 | 16.569 | 0.8212 | 0.4266 |
| SIM-02 Planar-Y-Pull | 80/20预测 | C：B + 全局distance刚度与全局阻尼更新（无区域/逐粒子刚度） | 1.3287 | 13.530 | 16.672 | 0.8219 | 0.4260 |
| SIM-03 Edge-Z-Lift | 7:1重建 | A：纯 PBD | 2.2596 | 19.890 | 19.079 | 0.7767 | 0.5488 |
| SIM-03 Edge-Z-Lift | 7:1重建 | B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度 | 0.4778 | 4.628 | 21.895 | 0.7933 | 0.5370 |
| SIM-03 Edge-Z-Lift | 7:1重建 | C：B + 全局distance刚度与全局阻尼更新（无区域/逐粒子刚度） | 0.4859 | 4.432 | 21.774 | 0.7948 | 0.5297 |
| SIM-03 Edge-Z-Lift | 80/20预测 | A：纯 PBD | 5.9556 | 53.957 | 14.343 | 0.7462 | 0.5653 |
| SIM-03 Edge-Z-Lift | 80/20预测 | B：PBD + CoTracker轨迹校正 + FoundationStereo RGB深度 | 1.0134 | 9.525 | 20.878 | 0.7972 | 0.5212 |
| SIM-03 Edge-Z-Lift | 80/20预测 | C：B + 全局distance刚度与全局阻尼更新（无区域/逐粒子刚度） | 0.9884 | 8.025 | 20.544 | 0.7925 | 0.5304 |
