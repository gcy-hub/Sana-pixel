# SANA-Pixel 第一阶段工程验证记录

验证日期：2026-09-22；节点：`node01`；GPU：NVIDIA A40 46GB。

## 数据审计

- WebDataset：10 shards，每片 2,000 个图文样本。
- 20,000 个唯一 key，10,000 个唯一 `source_id`，每个 source 有两个 variants。
- seed 0–19,999，无重复；全部图像为 1024×1024。
- 类别记录数：Nature 11,400 / Design 5,400 / People 2,600 / Synthetic 600。
- 官方训练入口实际报告 dataset length 20,000、token grid 32×32。

## 权重转换

- 源 checkpoint：6,430,666,040 bytes。
- 源 SHA256：`5e09953e952956c14570820e39abef8ac7483b88ff3abb406a8612f1dac3a2b4`。
- 源张量 398：精确复制 393，按 allowlist 丢弃 latent I/O 5，新建 pixel I/O 36。
- 转换后严格加载：missing 0 / unexpected 0。
- 完整模型参数：1,630,080,003；训练 850,784,003；冻结 779,296,000。

转换报告：

```text
/home/ganchangyi/huggingface_ckpts/SANA1.5_1.6B_1024px_pixel_init/checkpoints/
SANA1.5_1.6B_1024px_pixel_init.pth.conversion.json
```

## node01 G2 验证

1024×1024、FP32 master weights、BF16 autocast、AdamW 单步：

- 输出形状 `[1, 3, 1024, 1024]`，全部 finite。
- loss `0.0045313546`（合成 smoke 输入）。
- 首 5 / 尾 5 blocks、RGB patch embedder、Detailer 第一层均有梯度。
- 中间冻结 block 无梯度，optimizer step 后冻结探针不变。
- 所有可训练梯度 finite；可训练 backbone 与 interface 探针均发生更新。
- 峰值 CUDA allocated：20,399,864,320 bytes（约 19.0 GiB）。

真实 WebDataset + 本地 Gemma + flow loss 的单步训练：

- loss `1.6261`，grad norm `0.8543`。
- 日志明确报告 pixel mode 禁用 VAE construction/encode/decode；VAE time `0.000`。
- 输出日志：`output/sana_pixel_e2e_smoke/train_log.log`。

单样本 300-update 过拟合（相同图文记录，随机 flow timestep/noise）：

- 前 50 步平均 loss：`1.558002`。
- 后 50 步平均 loss：`0.880122`，相对下降 `43.51%`。
- 300 步均为 finite；最小/最大 loss：`0.768500 / 1.634600`。
- 输出日志：`output/sana_pixel_overfit1_300/train_log.log`。

完整训练状态保存/恢复 smoke：

- Step 1 保存为 `epoch_1_step_1.pth`，文件大小 13,339,029,516 bytes。
- `resume_from=latest` 成功恢复 model、AdamW、LR scheduler 与 global step；下一次更新严格从 Step 2 开始。
- warmup LR 从 Step 1 的 `2.0e-7` 连续到 Step 2 的 `4.0e-7`，没有重置。
- 测试期间发现并修复了原训练循环 checkpoint 名称领先训练进度 1 step 的问题。
- 输出日志：`output/sana_pixel_save_resume_smoke_fixed/train_log.log`。

直接 RGB 推理以最小合法的 2-step flow DPM-Solver 完成，并保存 1024×1024 RGB JPEG：

```text
output/sana_pixel_inference_smoke/vis/
```

当前只证明工程闭环正确；随机初始化的 pixel interfaces 尚未训练，因此该图不用于质量判断。

## 尚未完成

- 32/128 样本条件诊断与 1K pilot。
- 100-step 稳定性 smoke（最小 save/resume 已通过）。
- 完整 20K 图像的一轮 5K-update 训练与正式评估。
