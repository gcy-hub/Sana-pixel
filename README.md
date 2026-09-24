# Sana-pixel

**把 SANA 1.5 从 latent 空间迁移到 pixel 空间做训练的代码**（在官方 [Sana](https://github.com/NVlabs/Sana) 代码库基础上改的个人版本）。

具体做的事：RGB `3×1024×1024` 图像**不经过 VAE**，直接进 ps32 patch embedder；保持 SANA 1.5 的 flow-matching velocity 目标（`flow_shift=3.0` + logit-normal 时间步）；复用 transformer / 时间 / 文本条件权重，只重新初始化 RGB patch embedder 与 Pixel Detailer Head；训练时只放开 transformer 的**前 5 块 + 后 5 块**，中间 10 块冻结。

> **English TL;DR** — A personal fork of the Sana codebase for **latent→pixel transfer training** of SANA-1.5-1.6B at 1024px. See [§1](#1-给接手的人一页速览) for the handover checklist (what to copy, what to download, which paths to edit), [§6](#6-启动训练) for launch and [§7](#7-续训--断点恢复) for resume.

---

## 目录

1. [给接手的人：一页速览](#1-给接手的人一页速览)
2. [硬件与验证过的环境](#2-硬件与验证过的环境)
3. [要准备哪些模型和数据](#3-要准备哪些模型和数据)
4. [必须改的路径](#4-必须改的路径)
5. [数据集格式](#5-数据集格式)
6. [启动训练](#6-启动训练)
7. [续训 / 断点恢复](#7-续训--断点恢复)
8. [显存、磁盘、时间预算](#8-显存磁盘时间预算)
9. [推理（可选）](#9-推理可选)
10. [常见坑](#10-常见坑)
11. [上游与许可](#11-上游与许可)

---

## 1. 给接手的人：一页速览

### 1.1 你会拿到什么

| # | 东西 | 我这边的大小 | 用途 |
|---|---|---|---|
| 1 | 本仓库代码 | ~57 MB | 训练/推理全部代码 |
| 2 | `webdataset/` 数据集 | **27 GB** | 20,000 条 1024×1024 图文样本 |
| 3 | `gemma-2-2b-it/` 文本编码器 | **9.8 GB** | ⚠️ **必需**，不在模型 checkpoint 里 |
| 4 | 训练好的模型 `epoch_XXX_step_XXX.pth` | **13.3 GB / 个** | 续训 / 微调的起点 |

### 1.2 你还需要自己做什么

| 要做的事 | 说明 |
|---|---|
| 装环境 | Python 3.12 + CUDA 12.4 对应的 torch 2.5.1，见 [§2](#2-硬件与验证过的环境)。**不要**直接跑 `environment_setup.sh`（它 pin 的是另一套版本） |
| 改 3 个路径 | `configs/sana_pixel/*.yaml` 里的数据集路径、`model.load_from`、gemma 路径，见 [§4](#4-必须改的路径) |
| 改启动脚本 | `train_scripts/sbatch_train_sana_pixel_8gpu_100epoch.sh` 里的 `REPO_ROOT`、conda 环境名、Slurm 账号/分区，见 [§4](#4-必须改的路径) |
| **不需要**下载 VAE | DC-AE VAE (`mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`) 在 pixel 模式下**完全不会被构造**，不用下 |
| **不需要**下载 pixel-init | 如果你是**续训**我给的 checkpoint，`model.load_from` 会被自动跳过（见 [§7](#7-续训--断点恢复)） |

### 1.3 最短路径（假设你要接着我的训练继续跑）

```bash
# 0. 假设代码在 /your/path/Sana，数据集在 /your/data/SANA-Pixel-Dataset/webdataset
cd /your/path/Sana
export PYTHONPATH=$PWD                      # 必需，这个仓库不是以包形式安装的

# 1. 把模型放到 work_dir 的 checkpoints/ 下
mkdir -p output/sana_pixel_8gpu_1000epoch/checkpoints
cp /path/to/epoch_210_step_65597.pth output/sana_pixel_8gpu_1000epoch/checkpoints/

# 2. 改 configs/sana_pixel/Sana_1600M_1024px_webdataset_bf16_lr2e5.yaml 里的 3 个路径

# 3. 续训
NP=8 WORK_DIR=output/sana_pixel_8gpu_1000epoch bash train_scripts/train_sana_pixel.sh \
  --name=sana_pixel_8gpu_1000epoch --report_to=tensorboard --resume_from=latest \
  --train.train_batch_size=4 --train.gradient_accumulation_steps=4 \
  --train.num_epochs=1000 --train.early_stop_hours=0 --train.visualize=true \
  --train.eval_sampling_epochs=15 --train.eval_sampling_steps=1000000000 \
  --train.save_model_epochs=15 --train.save_model_steps=1000000000
```

---

## 2. 硬件与验证过的环境

### 2.1 硬件

| 项 | 我这边验证过的 | 说明 |
|---|---|---|
| GPU | **NVIDIA A40 46 GB** | 生产跑的是 2 节点 × 4 卡 = 8 卡（Slurm） |
| 最小可跑 | 1 卡（`NP=1`，仅用于 smoke） | 正式训练建议 ≥ 4 卡 |
| 节点资源 | 每节点 72 CPU / 大内存 | `num_workers: 10`，数据是 1024px 原图 |
| 驱动 | 550.54.14 | CUDA 12.4 运行时 |
| 本地磁盘 | 训练目录要留 **≥ 400 GB** | checkpoint 不自动清理，见 [§8](#8-显存磁盘时间预算) |

### 2.2 验证过的版本（**照抄这一套**）

这是实际跑通训练的环境（conda env 名为 `pixel`，Python **3.12.14**）：

```text
python          3.12.14
torch           2.5.1+cu124
torchvision     0.20.1+cu124
triton          3.1.0
transformers    5.17.0
diffusers       0.40.0
accelerate      1.15.0
peft            0.21.0
flash-linear-attention  0.3.2      # attn_type: linear 依赖它
mmcv            1.7.2
webdataset      0.2.111
omegaconf       2.3.1
numpy           1.26.4
tensorboard     2.16.2
setuptools      83.0.0
bitsandbytes    0.50.2
```

> ⚠️ **`environment_setup.sh` 不能直接用来复现这套环境。** 它按 `pyproject.toml` 装的是 Python 3.11 / `torch 2.9.1+cu128` / `transformers 4.57.3` /
> `flash-attn` / `transformer_engine`，和我实际跑通的版本**不一样**。
> **优先方案是直接复制 conda 环境**（`conda env export` + `conda env create`，或 `pip freeze` 后重装）。
> 如果必须重装，按上面这张表来装，**不要**跑 `environment_setup.sh`。

### 2.3 不需要装的东西

| 包 | 为什么不需要 |
|---|---|
| `flash-attn` | 配置里 `use_flash_attn: false`，交叉注意力走 PyTorch SDPA；实际环境里也没装 |
| `xformers` | 实际环境里没有；跑测试时用 `DISABLE_XFORMERS=1` |
| `transformer_engine` / `Pi3` / `qwen-vl-utils` / `hpsv2` | 这些是 SANA-WM / SANA-Video / 评测工具链用的，pixel 训练用不到 |

### 2.4 仓库是「就地运行」，不是安装成包

实测 `pixel` 环境里 `import diffusion` 是**失败**的——这个仓库没有 `pip install -e .`。
所有脚本都靠 `PYTHONPATH=$REPO_ROOT`（或 `cd` 到仓库根）来导入 `diffusion.*` / `sana.*`。

**所以每个训练命令前都要有 `export PYTHONPATH=$PWD`，这是最容易踩的坑。**

---

## 3. 要准备哪些模型和数据

### 3.1 必需

| 文件 | 大小 | 从哪来 | 放在哪 / 怎么配 |
|---|---|---|---|
| **数据集** `webdataset/` | 27 GB | 我直接给你 | `configs/.../Sana_1600M_1024px_webdataset_bf16_lr2e5.yaml` → `data.data_dir` |
| **Gemma-2-2B-it** | 9.8 GB | 我直接给你整个目录（推荐）；或自己从 `google/gemma-2-2b-it` 下（HF 上的**门控模型**，要先在网页上接受 Google 的许可协议，再 `huggingface-cli login`） | 同上 YAML → `text_encoder.text_encoder_name`（填**本地目录路径**，不要填 HF repo id） |

Gemma 不能随便换：`caption_channels: 2304`、`model_max_length: 300` 是跟这个编码器绑死的。

### 3.2 看情况需要

| 文件 | 大小 | 什么时候需要 |
|---|---|---|
| 我的**训练 checkpoint** `epoch_210_step_65597.pth` | 13.3 GB | **续训 / 微调**时。这是**完整训练状态**（模型 + optimizer + scheduler + epoch/step），不是纯权重 |
| **pixel-init** `SANA1.5_1.6B_1024px_pixel_init.pth` | 6.5 GB | **从零开始训**（不续训）时作为权重初始化 |
| SANA1.5 1.6B 原始 `SANA1.5_1.6B_1024px.pth` | 6.0 GB | 只有要从头**重新生成** pixel-init 时才要（见 [§3.4](#34-可选自己生成-pixel-init)） |

### 3.3 明确不需要

| 文件 | 为什么不需要 |
|---|---|
| **DC-AE VAE** `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers` (1.3 GB) | 代码里 `train.py:776` 只在 `not pixel_space` 时才构造 VAE；pixel 模式下 `vae = None`，**`vae_pretrained` 这个字段根本不会被读**。YAML 里留着它只是为了 `SanaConfig` 的结构完整性。日志里会打印 `data space: pixel (VAE construction, encode, and decode are disabled)` |

### 3.4 （可选）自己生成 pixel-init

只有在你没有我给的 pixel-init、又需要从零训练时才做：

```bash
cd <repo root>
export PYTHONPATH=$PWD

# 1) 下原始 SANA1.5 1.6B checkpoint（只取那个 .pth，别把 diffusers 权重也拉下来）
hf download Efficient-Large-Model/SANA1.5_1.6B_1024px \
   checkpoints/SANA1.5_1.6B_1024px.pth --local-dir /your/ckpts/SANA1.5_1.6B_1024px

# 2) 转换成 pixel 初始化权重
python tools/convert_sana_to_pixel.py \
  --source /your/ckpts/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth \
  --output /your/ckpts/SANA1.5_1.6B_1024px_pixel_init/checkpoints/SANA1.5_1.6B_1024px_pixel_init.pth
```

脚本会把 transformer / 时间 / 文本条件权重**原样复制**，丢掉 latent 的 `x_embedder.*` 与 `final_layer.*`，并用固定 `--seed 1`（可改）**确定性**初始化 RGB patch embedder 和 Pixel Detailer Head。
`--dry-run` 只看报告不落盘，`--source-sha256` 会顺便校验源文件哈希。
转换完旁边会有一个 `.conversion.json` 记录明细（missing / unexpected 都应为 0）。

---

## 4. 必须改的路径

### 4.1 配置文件：`configs/sana_pixel/Sana_1600M_1024px_webdataset_bf16_lr2e5.yaml`

**只有这 3 行是机器相关的**，其余超参**不用改**（见 [§6.4](#64-训练超参就是现在这一套)）：

| 行 | 字段 | 我这边 | 你改成 |
|---|---|---|---|
| 8 | `data.data_dir` | `/home/ganchangyi/dataset/SANA-Pixel-Dataset/webdataset` | 你的数据集根目录（**注意是列表** `[...]`） |
| 33 | `model.load_from` | `/home/ganchangyi/huggingface_ckpts/SANA1.5_1.6B_1024px_pixel_init/checkpoints/SANA1.5_1.6B_1024px_pixel_init.pth` | 你的 pixel-init 路径。**只做续训时它不会被读**，但不要留一个不存在的路径（某些回退分支下会报错） |
| 60 | `text_encoder.text_encoder_name` | `/home/ganchangyi/huggingface_ckpts/gemma-2-2b-it` | 你本地的 Gemma 目录 |

另外建议确认（一般不用改）：`train.null_embed_root: output/pretrained_models/` 和 `train.valid_prompt_embed_root: output/tmp_embed/` 是**相对路径**，第一次跑会自动生成（需要能读到 Gemma）。

### 4.2 启动脚本：`train_scripts/sbatch_train_sana_pixel_8gpu_100epoch.sh`

| 行 | 内容 | 要改什么 |
|---|---|---|
| 3–11 | `#SBATCH --account=students --partition=gpujl --nodes=2 --gpus-per-node=4 ...` | 改成你集群的账号、分区、节点数 |
| 12–13 | `--output=/home/ganchangyi/.../output/slurm-%x-%j.out` | Slurm 日志绝对路径 |
| 17 | `REPO_ROOT="/fs1/private/user/ganchangyi/code/Sana-pixel/Sana"` | 你的仓库绝对路径 |
| 20 | `TORCHRUN=".../envs/pixel/bin/torchrun"` | 你环境里的 `torchrun`（或直接用 `torchrun`） |
| 23–24 | `conda.sh` 路径 / `conda activate pixel` | 你的 conda 和环境名 |
| 27–28 | `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` | 本地权重齐了可以保留（避免联网卡住）；**Gemma 没缓存好就删掉这两行**，否则会直接报找不到模型 |

⚠️ `--gpus-per-node` 必须和 torchrun 的 `--nproc_per_node` 相等（脚本里硬编码是 4）。

### 4.3 单机脚本：`train_scripts/train_sana_pixel.sh`

不写死绝对路径，靠环境变量，一般不用改：

```text
CONFIG=configs/sana_pixel/Sana_1600M_1024px_webdataset_bf16_lr2e5.yaml
NP=4
WORK_DIR=output/sana_pixel_1024_10k
MASTER_PORT=29510
TORCHRUN=torchrun
```

### 4.4 代码本身

`diffusion/` 和 `sana/` 里的 Python 代码**没有**写死的 `/home/ganchangyi` 路径，不用改。
写死路径只在上面这些配置/脚本，以及 `tools/convert_sana_to_pixel.py`（只在 [§3.4](#34-可选自己生成-pixel-init) 才用）和 `tools/dataset_gen/*`（只在你重新造数据集时才用）。

---

## 5. 数据集格式

### 5.1 目录结构

```text
webdataset/
├── wids-meta.json          # 必需：分片清单
└── shards/
    ├── shard-00000.tar     # 每片 2000 条样本
    ├── shard-00001.tar
    └── ...                 # 当前共 10 片 = 20,000 条
```

`wids-meta.json`：

```json
{
  "wids_version": 1,
  "shardlist": [
    {"url": "shards/shard-00000.tar", "nsamples": 2000, "filesize": 2908313600}
  ]
}
```

`url` 是**相对路径**（相对 `wids-meta.json` 自己所在目录）；`wids_version` 必须是 `1`；每个分片必须有 `url` + `nsamples`。

### 5.2 每个样本

tar 里是**成对**的 `<key>.png` + `<key>.json`（USTAR、纯文件名、无目录）：

```text
Nature-Food-0787.png
Nature-Food-0787.json
```

`<key>.json` 里的 **`prompt` / `height` / `width` 是加载器真正要用的**（其余字段如 `source_id` / `class` / `seed` 是造数据时留的元信息）：

```json
{
  "file_name": "Nature-Food-0787.png",
  "prompt": "A stylized cacao pod with bold ridges, ...",
  "height": 1024,
  "width": 1024,
  "global_index": 5117, "variant": 1, "prompt_index": 5117,
  "class": "Nature", "subclass": "Food", "topic": "cacao pod",
  "angle": "Graphic", "seed": 5117,
  "source_id": "Nature/Food/0078/cacao_pod/08"
}
```

caption 走 JSON 里的 `prompt` 字段（配置 `caption_proportion: {prompt: 1}` + `caption_selection_type: proportion`），**不是** `.txt` 文件。

### 5.3 想自己扩充数据集

仓库里有专门给这个数据集用的打包脚本：

```bash
python tools/dataset_gen/consolidate_dataset.py --dry-run   # 只校验
python tools/dataset_gen/consolidate_dataset.py             # 生成 metadata.json
python tools/dataset_gen/pack_webdataset.py --strict --samples-per-shard 2000
python tools/dataset_gen/verify_webdataset.py               # 真正用 SanaWebDatasetMS 加载一遍
```

**踩坑提醒（都会静默丢样本，不报错）：**

- `<key>.json` 里缺 `height`/`width` → 该样本被采样器**静默跳过**。
- 分片是**顺序读取**的（`DistributedRangedSampler` 不分片内乱序），所以**打包时必须 shuffle**，否则每个 epoch 拿到的样本顺序是固定的。
- 加载器用了硬编码的 `lru_size = 10`，分片数建议 ≤ 10，否则缓存会抖动。
- `.jpeg` / `.webp` 不被支持，只认 `.png` / `.jpg`。

---

## 6. 启动训练

### 6.0 先自检（强烈建议）

```bash
cd <repo root>
export PYTHONPATH=$PWD

# 单元测试：验证 pixel 模型、权重转换、严格加载
DISABLE_XFORMERS=1 python -m unittest -v tests.test_sana_pixel

# 单卡前向/反向 + AdamW 单步审计（1024px，确认显存和梯度都对）
python tools/smoke_test_sana_pixel.py --optimizer-step --image-size 1024
```

两个都过了再上多卡。

### 6.1 单机多卡（推荐先用这个）

```bash
cd <repo root>
export PYTHONPATH=$PWD

NP=8 bash train_scripts/train_sana_pixel.sh \
  --name=sana_pixel_8gpu_1000epoch \
  --report_to=tensorboard \
  --train.train_batch_size=4 \
  --train.gradient_accumulation_steps=4 \
  --train.num_epochs=1000 \
  --train.early_stop_hours=0 \
  --train.visualize=true \
  --train.eval_sampling_epochs=15 \
  --train.eval_sampling_steps=1000000000 \
  --train.save_model_epochs=15 \
  --train.save_model_steps=1000000000
```

`WORK_DIR` 默认是 `output/sana_pixel_1024_10k`，建议显式指定：

```bash
NP=8 WORK_DIR=output/my_run bash train_scripts/train_sana_pixel.sh ...
```

这个脚本本质就是（`train_scripts/train_sana_pixel.sh:12-20`）：

```bash
torchrun --nproc_per_node=$NP --master_port=$MASTER_PORT \
  train_scripts/train.py \
  --config_path=configs/sana_pixel/Sana_1600M_1024px_webdataset_bf16_lr2e5.yaml \
  --work_dir=$WORK_DIR --name=$NAME --report_to=tensorboard "$@"
```

**`train.py` 必须用 `torchrun` 启动**：它直接读 `os.environ["LOCAL_RANK"]` / `WORLD_SIZE` / `RANK`，没有默认值，`python train.py` 会直接 `KeyError`。单卡也要写 `NP=1`。

### 6.2 多机 Slurm

改好 [§4.2](#42-启动脚本train_scriptssbatch_train_sana_pixel_8gpu_100epochsh) 里的东西后：

```bash
sbatch train_scripts/sbatch_train_sana_pixel_8gpu_100epoch.sh
```

脚本的做法是**每节点一个 torchrun**，由 `srun` 拉起，用 `scontrol show hostnames` 取第一个节点当 `MASTER_ADDR`：

```bash
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -lc '
  exec "$TORCHRUN" --nnodes=$SLURM_NNODES --nproc_per_node=4 \
    --node_rank=$SLURM_PROCID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ...
'
```

注意：

- `MASTER_PORT` 默认 29520，多任务共用节点时要错开。
- 集群有 HTTP 代理时，脚本里的 `NO_PROXY=127.0.0.1,localhost` 是必要的，否则 NCCL 初始化会卡住。保留它。

### 6.3 TensorBoard

配置里是 `--report_to=tensorboard`（`report_to` 的默认值其实是 `wandb`，一定要显式覆盖）。事件文件在：

```text
<work_dir>/logs/<tracker_project_name>/     # tracker_project_name 默认 sana-video-baseline
```

### 6.4 训练超参：就是现在这一套

下表是**实际生产跑的那一套**（YAML + 上面 CLI 覆盖合并后的有效值），你照抄就行：

| 项 | 值 | 来源 |
|---|---|---|
| 模型 | `SanaMSPixel_1600M_P32_D20` | YAML |
| 分辨率 / 数据空间 | 1024 / `pixel`（**不过 VAE**） | YAML |
| 每卡 batch | 4 | **CLI** |
| 梯度累积 | 4 | **CLI** |
| 全局 batch | 4 × GPU 数 × 4 | 推导 |
| 优化器 | AdamW，`lr 5e-5`，`betas (0.9, 0.999)`，`weight_decay 0.01`，`eps 1e-8` | YAML |
| LR 调度 | `constant` + `num_warmup_steps: 1000` | YAML |
| 精度 | bf16 (`mixed_precision: bf16`)，`fp32_attention: true` | YAML |
| 梯度检查点 / 裁剪 | `grad_checkpointing: true` / `gradient_clip: 1.0` | YAML |
| 分布式 | **DDP**（`use_fsdp: false`），`ema_update: false` | YAML |
| 可训练范围 | 前 5 块 + 后 5 块 + pixel 接口，中间 10 块冻结 | YAML `l2p_first/last_trainable_blocks: 5` |
| attention / FFN | `attn_type: linear`（走 flash-linear-attention）/ `ffn_type: glumbconv` | YAML |
| flow 设定 | `linear_flow`，`flow_shift: 3.0`，`predict_flow_v: true`，`logit_normal`（mean 0, std 1） | YAML |
| epochs | **1000** | **CLI**（YAML 默认 100） |
| 存 checkpoint | **每 15 epoch** | **CLI**（YAML 默认每 1 epoch） |
| 可视化 / 验证 | 每 15 epoch，`visualize: true`，`local_save_vis: true` | **CLI** |
| 训练时长上限 | **关闭**（`early_stop_hours: 0`） | **CLI**（YAML 默认 100 小时） |
| `num_workers` | 10 | YAML |

> YAML 里的 `eval_sampling_steps: 500` / `save_model_steps: 10000` 这些**被 CLI 覆盖成 `1000000000`**，
> 也就是把「按 step 触发」关掉，只留「按 epoch 触发」。**如果你漏掉这些覆盖，会变成每 1 个 epoch 存一次 checkpoint**，磁盘很快就满。

### 6.5 常用 CLI 覆盖

`train.py` 用 [pyrallis](https://github.com/eladrich/pyrallis) 解析，任何 YAML 字段都能用 `--a.b=value` 覆盖：

| 参数 | 说明 |
|---|---|
| `--config_path` | 基础 YAML |
| `--work_dir` | 运行目录（checkpoint / 日志 / 可视化都在这下面） |
| `--name` | 运行名 |
| `--report_to` | `tensorboard` 或 `wandb`（默认 `wandb`，**必须显式改成 tensorboard**） |
| `--resume_from` | `latest` 或某个 `.pth` 路径，见 [§7](#7-续训--断点恢复) |
| `--load_from` | 覆盖 `model.load_from`（权重初始化，不恢复 optimizer） |
| `--train.num_epochs` | 总 epoch 数 |
| `--train.train_batch_size` | **每卡** batch |
| `--train.gradient_accumulation_steps` | 梯度累积 |
| `--train.save_model_epochs` / `save_model_steps` | 存 checkpoint 的触发条件 |
| `--train.early_stop_hours` | 训练时长上限（小时），`0` = 关闭 |
| `--train.visualize` / `eval_sampling_epochs` / `eval_sampling_steps` | 验证可视化 |
| `--train.use_fsdp` | 默认 `false`（DDP） |

每次启动会把自己的**合并后配置** dump 到 `<work_dir>/config.yaml`，出问题先看这个文件。

---

## 7. 续训 / 断点恢复

### 7.1 checkpoint 是什么

- 位置：`<work_dir>/checkpoints/`
- 命名：`epoch_{E}_step_{S}.pth`，例如 `epoch_210_step_65597.pth`
- 内容：**完整训练状态**——`state_dict`、`optimizer`、`scheduler`、`epoch`、`step`、`rng_state`（如果开了 EMA 还有 `state_dict_ema`）
- 大小：**约 13.3 GB**（因为含 AdamW 的 optimizer state，不是纯权重）
- 另有 `latest.pth`，是一个**绝对路径软链接**指向最新的那个文件

### 7.2 恢复的语义（很重要）

| 你传的 | 行为 |
|---|---|
| **不传 `--resume_from`** | 全新训练。会加载 `model.load_from` 作为**权重初始化**（pixel 模式下只允许缺 `pos_embed`，其它 missing/unexpected 直接报错） |
| **`--resume_from=latest`** | 在 `<work_dir>/checkpoints/` 里找 checkpoint：优先用 `latest.pth`；如果它不存在或是**断链**，就按文件名里的 **step 号排序取最大** 的那个。恢复模型 + optimizer + scheduler + epoch/step |
| **`--resume_from=/abs/path/xxx.pth`** | 用指定文件。注意：这种写法**不会**恢复 LR scheduler（因为 YAML 里 `resume_lr_scheduler: false`），所以**推荐用 `latest`** |

三个关键点：

1. **`--resume_from` 一旦生效，`model.load_from` 就不会被读**（`train.py:703-712`：`load_from = False`）。所以你**不需要**把 pixel-init 也拷过去。
2. **`latest.pth` 是绝对路径软链，跨机器拷贝一定会断。** 但没关系：代码检测到断链会自动回退到「按 step 号取最大」，所以**直接把 `epoch_*.pth` 真实文件拷过去就能用**。想干净点就自己重建软链。
3. **`<work_dir>/checkpoints/` 为空时，`--resume_from=latest` 不会报错**，而是**静默退化**成「用 `model.load_from` 加载权重、不恢复 optimizer」起步。所以看到 loss 从初始值开始、没有 `Skipped Steps` 日志，就说明它其实没接上。

### 7.3 怎么续训

```bash
cd <repo root>
export PYTHONPATH=$PWD

# 1) 把真实 checkpoint 放进 work_dir（用真实文件，不要拷软链）
mkdir -p output/sana_pixel_8gpu_1000epoch/checkpoints
cp /path/to/epoch_210_step_65597.pth output/sana_pixel_8gpu_1000epoch/checkpoints/

# 2) 加 --resume_from=latest，其余参数和我完全一致
NP=8 WORK_DIR=output/sana_pixel_8gpu_1000epoch bash train_scripts/train_sana_pixel.sh \
  --name=sana_pixel_8gpu_1000epoch --report_to=tensorboard --resume_from=latest \
  --train.train_batch_size=4 --train.gradient_accumulation_steps=4 \
  --train.num_epochs=1000 --train.early_stop_hours=0 --train.visualize=true \
  --train.eval_sampling_epochs=15 --train.eval_sampling_steps=1000000000 \
  --train.save_model_epochs=15 --train.save_model_steps=1000000000
```

Slurm 的话 `sbatch` 脚本里**已经有** `--resume_from=latest`（第 60 行），直接 `sbatch` 就行。

### 7.4 只要权重、不要 optimizer（微调场景）

如果你想拿我的模型当初始化、但**从干净的 optimizer 重新开始**：

```bash
... train_sana_pixel.sh --load_from=/path/to/epoch_210_step_65597.pth   # 不要传 --resume_from
```

`load_from` 兼容完整训练状态文件（`checkpoint.py:293` 会取 `state_dict` 字段），所以可以直接指向训练 checkpoint。

### 7.5 ⚠️ 改 `num_epochs` 的坑

epoch 循环是 `range(start_epoch + 1, num_epochs + 1)`。如果你从 epoch 210 的 checkpoint 恢复，又设了 `--train.num_epochs=100`，**循环一次都不进，看起来「启动后什么都没发生」**。
恢复时务必保证 `num_epochs` 大于 checkpoint 里的 epoch。

---

## 8. 显存、磁盘、时间预算

实测数据（8 × A40 46 GB，2 节点 × 4 卡，20,000 样本）：

| 项 | 实测值 |
|---|---|
| 一轮完整数据遍历 | **625 个 local step**（= 20000 /（每卡 batch 4 × 8 卡）） |
| 单步耗时 | **约 2.0 s**（`time all:1.99, model:1.83, data:0.003, lm:0.15, vae:0.006`） |
| 一个 checkpoint 间隔 | 约 **2.4 小时**（每 15 epoch） |
| 单个 checkpoint | **13.3 GB** |
| 实测 37 小时 | 跑到 epoch 221 / global step 69170 |
| 单卡 1024px 单步峰值显存 | 约 **19 GiB**（`tools/smoke_test_sana_pixel.py`） |
| 无 OOM / 无 CUDA error | ✅ |

### 磁盘警告

**checkpoint 不会自动清理**（代码里没有任何 `os.remove`）。按 `save_model_epochs=15` 跑满：

```text
1000 / 15 ≈ 67 个 checkpoint × 13.3 GB ≈ 890 GB
```

必须提前规划。可选做法：

- 调大 `--train.save_model_epochs`（比如 30/50）；
- 或者定期手动删旧的 `epoch_*.pth`（**保留 `latest.pth` 指向的那个**）；
- 或者写个清理脚本只保留最近 3 个。

### 时间估算的两个提醒

1. **日志里的 `Epoch` 计数和「数据遍历次数」不是 1:1**（这个仓库实测约 2 epoch 对应 1 次完整遍历），所以**估时间请用 local step / checkpoint 文件名里的 step 号**，别用 epoch。
2. **代码内置的 `total_eta` 字段偏大**（它按 `dataloader_len × num_epochs` 算，和实际的 epoch 计数不一致）。我这边日志显示 `total_eta: 12 days`，按实测速率推算的实际总时长约为它的一半。**别拿这个数字做决策。**

### 显存不够怎么办

按这个顺序调：

1. 调小 `--train.train_batch_size`（每卡 batch），同时**按比例调大** `--train.gradient_accumulation_steps` 保持全局 batch 不变（全局 batch = 每卡 batch × 卡数 × 累积步数）；
2. 确认 `grad_checkpointing: true`（YAML 里已经是）；
3. 卡不够就用更少卡 + 更多累积（注意一轮的 local step 数 = 20000 /（batch × 卡数）会变，checkpoint 间隔的**语义**也跟着变）。

---

## 9. 推理（可选）

推理**复用同一份 YAML**：pixel 模式会读 `model.load_from`，直接输出 RGB，**同样不加载 VAE**。

```bash
cd <repo root>
export PYTHONPATH=$PWD

# 用训练好的 checkpoint 推理：把 YAML 里的 model.load_from 指向它，
# 或者不传 --resume_from、直接用 --load_from 覆盖
python scripts/inference.py --config configs/sana_pixel/Sana_1600M_1024px_webdataset_bf16_lr2e5.yaml
```

`scripts/inference.py` 只有一个 `--config` 参数，其余都从 YAML 里读（包括 prompt 列表、采样步数、`vis_sampler: flow_dpm-solver`、`flow_shift: 3.0`）。

> 注意：**pixel 接口是随机初始化的新模块**，如果你用的是 pixel-init 而不是训练过的 checkpoint，出来的图是没有意义的。

---

## 10. 常见坑

| 现象 | 原因 / 解决 |
|---|---|
| `ModuleNotFoundError: No module named 'diffusion'` | 没设 `PYTHONPATH`。这个仓库不装成包，必须 `export PYTHONPATH=$PWD`（且 `cd` 到仓库根） |
| `KeyError: 'LOCAL_RANK'` | 用 `python train.py` 直接跑了。必须用 `torchrun --nproc_per_node=N` |
| 训练日志跑到 wandb 上去了 / 报 wandb 相关错 | `report_to` 默认是 `wandb`，必须显式 `--report_to=tensorboard` |
| 磁盘疯涨，一个 epoch 一个 checkpoint | 漏了 `--train.save_model_steps=1000000000` / `--train.eval_sampling_steps=1000000000` 覆盖 |
| `--resume_from=latest` 后 loss 从初始值开始 | `<work_dir>/checkpoints/` 是空的，静默退化成了 `load_from` 权重初始化。确认你把 `.pth` 放对了目录 |
| `--resume_from` 后什么都不跑 | `--train.num_epochs` ≤ checkpoint 里的 epoch，epoch 循环为空。调大 `num_epochs` |
| 拷贝到新机器后 `latest.pth` 报断链 | 绝对路径软链失效。**这是正常的**，代码会回退到按 step 排序取最新；也可以手动重建软链 |
| OOM | 见 [§8](#显存不够怎么办) |
| 集群上 NCCL 初始化卡死 | HTTP 代理问题，保留启动脚本里的 `NO_PROXY` / `no_proxy` |
| 找不到 Gemma | 要么 `text_encoder_name` 路径写错了，要么开了 `HF_HUB_OFFLINE=1` 但本地没有缓存。删掉那两个 OFFLINE 变量或把权重放对 |
| 一开始就报严格加载失败 `missing=[...] unexpected=[...]` | `model.load_from` 指向的 checkpoint 和 `SanaMSPixel_1600M_P32_D20` 结构不匹配。确认用的是 pixel-init 或训练 checkpoint，pixel 模式下只允许缺 `pos_embed` |
| 数据集长度是 0 / 样本被跳过 | `<key>.json` 缺 `height`/`width`（会**静默**丢样本），或者 `wids-meta.json` 的 `url` 路径不对 |

---

## 11. 上游与许可

本仓库基于 NVIDIA 的 **Sana** 代码库改（[NVlabs/Sana](https://github.com/NVlabs/Sana)，Apache-2.0）。上游的完整文档、模型库、其它任务（SANA-Video、SANA-WM、SANA-Sprint、Sol-RL、ControlNet…）都在 `docs/` 和 <https://nvlabs.github.io/Sana/docs/>。

本仓库相对上游的改动，只服务于 **1024px latent→pixel 迁移训练**这一件事，主要涉及：

- `diffusion/model/nets/sana_pixel.py` — pixel 模型与 Pixel Detailer Head
- `train_scripts/train.py` — pixel 模式的 VAE 旁路、L2P 冻结策略、checkpoint 命名修正
- `configs/sana_pixel/` — 本任务的配置
- `tools/convert_sana_to_pixel.py`、`tools/smoke_test_sana_pixel.py` — 权重转换与审计
- `tools/dataset_gen/` — 本数据集的生成与打包
- `tests/test_sana_pixel.py` — pixel 模式测试

工程验证记录见 `docs/sana_pixel_phase1.md` 和 `docs/sana_pixel_phase1_results.md`。

上游引用：

```bibtex
@article{xie2025sana,
  title={Sana 1.5: Efficient Scaling of Training-Time and Inference-Time Compute in Linear Diffusion Transformer},
  author={Xie, Enze and Chen, Junsong and Chen, Junyu and Cai, Han and Tang, Haotian and Lin, Yujun and Zhang, Zhekai and Li, Muyang and Zhu, Ligeng and Lu, Yao and Han, Song},
  journal={arXiv preprint arXiv:2501.18427},
  year={2025}
}
```

许可证：Apache-2.0，见 `LICENSE`。