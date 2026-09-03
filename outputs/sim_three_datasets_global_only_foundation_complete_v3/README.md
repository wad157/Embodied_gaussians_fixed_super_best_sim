# 三数据集全局刚度完整测评

本目录记录 SIM-01、SIM-02、SIM-03 的正式完整测评：

- A：纯 PBD；
- B：PBD + CoTracker 轨迹校正 + 各数据集独立 FoundationStereo RGB 深度；
- C：B + 单一全局 distance stiffness 与单一全局阻尼在线更新。

空间刚度表示是全局标量，不包含分区域或逐粒子刚度。重建使用 7:1 因果留出，未来预测使用前 80% 在线更新、后 20% 开环预测。最终轨迹指标只使用 30 个固定非夹持点。

## 仓库中保留的结果

- `comparison_all.md`：三数据集总表；
- 每个数据集的 `comparison_complete.md/json`；
- 每种方法的逐帧及汇总 `trajectory_metrics`、`render_metrics`；
- 预测轨迹、材料诊断、刚度更新信号、运行元数据和日志。

逐帧渲染 PNG 约占 900 MB，属于可由预测轨迹重新生成的缓存，因此不纳入 Git。PSNR、SSIM、LPIPS 的逐帧 CSV 和最终 JSON 均完整保留。原始数据集位于本地 `data/sim/`，遵循项目既有 `.gitignore`，不随代码仓库分发。

总表见 [comparison_all.md](comparison_all.md)。
