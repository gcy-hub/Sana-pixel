# SANA 1.5 latent-to-pixel：第一阶段

第一阶段只使用 10K prompt × 2 variants（20K 图像记录）的自生成数据：

```text
/home/ganchangyi/dataset/SANA-Pixel-Dataset/webdataset
```

4K 数据 `/home/ganchangyi/dataset/MultiAspect-4K-1M` 不在本阶段加载。

## 已实现的迁移

- RGB `3×1024×1024` 直接进入 ps32 patch embedder，不创建或调用 VAE。
- 保持 SANA 1.5 的 flow-matching velocity 目标、`flow_shift=3.0` 与
  logit-normal timestep sampling。
- 复用 transformer、时间和文本条件权重；重新初始化 RGB patch embedder
  与 Pixel Detailer Head。
- 训练 transformer 前 5 块和后 5 块，冻结中间 10 块；两个新接口始终训练。
- 四卡、每卡 batch 1、无梯度累积；一轮 20K 图像对应 5K optimizer updates。

## 权重转换

```bash
cd /fs1/private/user/ganchangyi/code/Sana-pixel/Sana
PYTHONPATH=$PWD /home/ganchangyi/software/miniconda3/envs/pixel/bin/python \
  tools/convert_sana_to_pixel.py --source-sha256
```

默认输出：

```text
/home/ganchangyi/huggingface_ckpts/SANA1.5_1.6B_1024px_pixel_init/checkpoints/SANA1.5_1.6B_1024px_pixel_init.pth
```

## 验证与训练

```bash
PYTHONPATH=$PWD DISABLE_XFORMERS=1 \
  /home/ganchangyi/software/miniconda3/envs/pixel/bin/python \
  -m unittest -v tests.test_sana_pixel

# node01 上的 1024px 严格加载、前反向和 AdamW 单步审计
PYTHONPATH=$PWD /home/ganchangyi/software/miniconda3/envs/pixel/bin/python \
  tools/smoke_test_sana_pixel.py --optimizer-step --image-size 1024

# 正式四卡训练
bash train_scripts/train_sana_pixel.sh
```

单卡诊断可使用：

```bash
NP=1 WORK_DIR=output/sana_pixel_smoke bash train_scripts/train_sana_pixel.sh
```

从最近一次完整训练状态（模型、optimizer、scheduler 与 step）恢复：

```bash
NP=1 WORK_DIR=output/sana_pixel_smoke bash train_scripts/train_sana_pixel.sh \
  --resume_from=latest
```

推理使用同一 YAML；pixel 模式会读取 `model.load_from`，直接保存 RGB 输出，
不会加载 VAE。训练得到新 checkpoint 后，把 YAML 中的 `model.load_from` 指向它。
