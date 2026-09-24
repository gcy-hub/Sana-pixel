# tools/dataset_gen — SANA-1.5 批量 T2I 数据集生成

用官方 SANA-1.5 1.6B 1024px 权重（启用 CHI）把 prompt JSON 批量渲染成 1024×1024 PNG，
支持**多 GPU 并行、断点续传、带 ETA 的聚合进度条**。

- 主脚本：`generate_t2i_dataset.py`
- 包装脚本：`output/gen_sana_t2i_dataset.sh`（含本机必需的环境变量修正）

## 快速开始

```bash
cd /fs1/private/user/ganchangyi/code/Sana-pixel/Sana

# 1) 先校验：不碰 GPU，检查 17 个 JSON / 10000 条 prompt 的命名与分片
python tools/dataset_gen/generate_t2i_dataset.py --scan-only --gpu-ids 0,1,2,3

# 2) 小样冒烟（3 张）
bash output/gen_sana_t2i_dataset.sh --limit 3 --gpu-ids 3

# 3) 全量（4 卡）
bash output/gen_sana_t2i_dataset.sh --gpu-ids 0,1,2,3
```

中断后**原样重跑同一条命令**即可续传（已存在的 PNG 会被跳过）。

## Slurm 提交（4 卡 / 1 天）

```bash
sbatch tools/dataset_gen/sbatch_t2i_dataset.sh      # account=students partition=gpujl 4卡 24h
squeue -u $USER
tail -f /home/ganchangyi/dataset/SANA-Pixel-Dataset/logs/slurm/slurm-<jobid>.out
```

**为什么用 `sbatch` 而不是仓库自带的 `sana-run`**：`sana-run` 底层是阻塞式 `srun`，
启动它的会话一断，整个进程树会被杀掉（本项目已经因此丢过一轮 2511 张的进度：4 个 shard
同时停在 `state=running`、`failures=0`，日志停在半途的采样步数上）。`sbatch` 提交后立即返回，
作业独立存活；本脚本本身可断点续传，作业被超时/抢占后**重新 `sbatch` 一次**即可继续。

### 两个必须显式申请的 Slurm 资源（否则会静默失败）

| 指令 | 不写会怎样 |
| --- | --- |
| `--mem=0` | Slurm 默认只给 **32G**（实测 `ReqTRES=...mem=32G`）。4 个 worker 同时 `torch.load` 6.4 GB 权重 + 模型/文本编码器，峰值约 48–64 GB → **OOM killer 直接 SIGKILL worker，日志无任何 traceback**，只剩部分卡继续跑，父进程会一直等下去。`--mem=0` = 用整机内存（节点 486 GB）。 |
| `--cpus-per-task=72` | 会退化成 `CPUs/Task=1`，tokenizer / PIL 编码 / fp32 VAE 等 CPU 侧工作被 cgroup 限流。 |

父进程会检测「worker 意外死亡」（退出码非 0，或最终状态不是 `finished`），
打印 `WARNING: gpu N worker died unexpectedly` 并在进度行尾标注 `DEAD [...]`，
继续让健康 worker 跑完后以非零码退出——不会像之前那样无限期挂住。

### 实测（node06，4×A40，475G 内存，4 个 worker 全活）

```
[04:20] 2734/10000 | eta 2:20:15 | 0.86 img/s | left 7266 | fail 0 | gpu0:770/2500 gpu1:635/2500 gpu2:684/2500 gpu3:645/2500
```

聚合 **0.86 img/s**（每卡约 4.6 s/张），剩余约 7300 张 → **2.3 小时**。
只有 2 张卡在跑时会掉到 0.42 img/s —— 所以务必确认 4 个 worker 都活着。

## 固定超参数（全部来自官方，不可改）

参数不是硬编码在脚本里，而是通过
`pyrallis.parse(config_class=SanaInference, config_path=<config>)` 读取
`scripts/inference.py` 的官方默认值 —— 即与官方推理脚本**同源**。

| 参数 | 值 | 来源 |
| --- | --- | --- |
| config | `configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml` | 官方 SANA-1.5 文档 |
| 模型 | `SanaMS_1600M_P1_D20`，1,604,641,952 参数，bf16，`fp32_attention: true` | config |
| 分辨率 | 1024×1024（latent 32×32，因为 prompt 无 `--ar` → `ar=1.0` → `[1024,1024]`） | config + `ASPECT_RATIO_1024_TEST` |
| 采样器 | `flow_dpm-solver`，20 步，order=2，multistep | `SanaInference` 默认（`step=-1`） |
| cfg_scale | 4.5 | `SanaInference` 默认 |
| pag_scale | 1.0 → `guidance_type=classifier-free` | `SanaInference` 默认 + `attn_type=linear` |
| flow_shift | 3.0 | config `scheduler` |
| batch size | 1 | `SanaInference` 默认 |
| VAE | `AutoencoderDC`，**float32**，scaling_factor 0.41407 | config `vae` |
| 文本编码器 | `gemma-2-2b-it`，`model_max_length=300`，`y_norm` 0.01 | config `text_encoder` |
| **CHI** | 启用：8 行指令，**208 tokens**；`max_length_all = 208+300-2 = 506` | config `text_encoder.chi_prompt` |

脚本只实现了官方默认的 `flow_dpm-solver` 路径；若 config 换成别的采样器会**直接报错**而不是静默降级。

### 明确不做的事（都会破坏与官方产出的对齐）

batch>1 合批、`torch.compile`、改步数、改 cfg、改分辨率桶。

## 输出结构

### 生成阶段（`generate_t2i_dataset.py` 运行期间）

```
SANA-Pixel-Dataset/
├── prompts/                       # 输入，只读
├── images/<Class>/<Subclass>/
│   └── <Class>-<Subclass>-<NNNN>.png
├── metadata/
│   ├── all.jsonl                  # 每条成功记录：文件/原始 id/类别/主题/角度/prompt/seed/参数
│   ├── manifest.csv               # 同上，扁平表
│   ├── failed.jsonl               # 失败项 + 类型/消息/traceback
│   ├── run_config.json            # 本次运行参数 + 环境 + CHI 统计 + 分片信息
│   └── parts/                     # 各 worker 的分片记录（可安全删除，finalize 会重建合并文件）
└── logs/
    ├── gpu-<id>.log               # 每个 worker 的完整 stdout/stderr
    └── progress-<run>-<gpu>.json  # 供父进程聚合 ETA 的进度文件
```

命名规则：`{大类文件夹名}-{小类 JSON 文件名去扩展名}-{该 JSON 内 0000 起的顺序号，补零 4 位}.png`。

`metadata/` 与 `logs/` 只服务于**生成期的断点续传与排障**。生成跑完后按下面的收尾流水线把它们
压缩成长期形态（`logs/` 与 `metadata/` 会被删除，溯源信息折进 `metadata.json`）。

### 收尾流水线

```bash
cd /fs1/private/user/ganchangyi/code/Sana-pixel/Sana

# 1) 只校验：CSV 的 N×25 字段与 all.jsonl 逐字段等价 + 每张图都在盘上
python tools/dataset_gen/consolidate_dataset.py --dry-run
# 2) 写 metadata.json（CSV→JSON，数值类型按 all.jsonl 还原）
python tools/dataset_gen/consolidate_dataset.py
# 3) 校验通过后才删 logs/ 与 metadata/（不可逆，之后无法再断点续传）
python tools/dataset_gen/consolidate_dataset.py --purge

# 4) 打包成官方 SanaWebDatasetMS 可直接读取的 WebDataset 分片
python tools/dataset_gen/pack_webdataset.py
# 5) 校验分片 + 真实跑一遍官方 dataloader
python tools/dataset_gen/verify_webdataset.py
```

### 追加 variant（每 prompt 多张图）之后的流水线

`--variant V` 只写自己那一份 records，所以要把上一轮的 `metadata.json` 和新一轮的
`metadata/manifest.csv` 求并集（`--merge` 会按 `global_index` 去重、给老记录回填 `variant`/`prompt_index`，
并交叉校验两个 variant 的同一 prompt 的 `source_id` 与文本完全一致）：

```bash
# 生成 variant 2（新的 10K）
mkdir -p /home/ganchangyi/dataset/SANA-Pixel-Dataset/logs/slurm
sbatch tools/dataset_gen/sbatch_t2i_dataset.sh --repeats 2 --variant 2

# 合并成 20K 的 metadata.json，然后清掉这一轮的 logs/ 与 metadata/
python tools/dataset_gen/consolidate_dataset.py --merge --dry-run   # 只校验
python tools/dataset_gen/consolidate_dataset.py --merge --purge

# 重打包：20K 需要抬高每片样本数，否则分片数会超过 SanaWebDataset 硬编码的 lru_size=10
python tools/dataset_gen/pack_webdataset.py --strict --samples-per-shard 2000
python tools/dataset_gen/verify_webdataset.py
```

### 长期形态（收尾后）

```
SANA-Pixel-Dataset/
├── prompts/                       # 17 个源 JSON（只读输入）
├── images/                        # N 张 PNG（只读真值）；variant≥2 为 <...>-v<v>.png
├── metadata.json                  # N 条记录 + generation(variant_runs) 溯源 + stats
└── webdataset/                    # 训练入口（data_dir 指向这里）
    ├── wids-meta.json             # 分片清单：url / nsamples / filesize
    └── shards/shard-00000.tar …   # 每个 tar = <key>.png + <key>.json（+ 可选 <key>.npy）
```

实测（`--repeats 1`，10K）：收尾后数据集根目录只有 `images/ metadata.json prompts/ webdataset/`，
文件总数 `10000 (png) + 17 (prompt json) + 1 (metadata.json) = 10018`（不含 tar 内部成员）。
`--repeats 2` 之后是 `20000 + 17 + 1 = 20018`，10 个分片 × 2000 样本。

## WebDataset 打包（给官方 dataloader 用）

`webdataset` 只是「把大量小文件装进 tar + 一份 JSON 清单」的文件组织约定，**不需要联网、不上传**：
WIDS 里只有 `http(s)://`/`gs://` 的 shard 才会下载；`wids-meta.json` 用相对 url，
解析后会拼成本地绝对路径，`download_and_open()` 直接原地 `open()`，既不下载也不复制。

每个 `<key>.json` 的内容（前 4 个是 loader 的硬性要求）：

```json
{"file_name":"Design-Anime-0000.png","prompt":"...","height":1024,"width":1024,
 "global_index":0,"class":"Design","subclass":"Anime","topic":"magical girl",
 "angle":"Studio hero","seed":0,"source_id":"Design/Anime/0000/magical_girl/01"}
```

几个必须守住的点（都已在 `pack_webdataset.py` 里落实）：

| 约束 | 原因 |
|---|---|
| `height`/`width` 必须有 | `SanaWebDatasetMS.getdata` 直接取 `info["height"]`，缺了会在 `get_data_info` 的 `try/except` 里**静默返回 None → 样本被悄悄丢掉** |
| **分片数 ≤ 10** | `SanaWebDataset` 的 `lru_size` 硬编码为 10；≤10 片时每卡只 open 一次，之后不再重开 |
| USTAR + 纯 basename 成员名 + 只有 `REGTYPE` | `MMIndexedTar` 手工按 512B 解析 tar 头，跳过 PAX 头，不认目录项 |
| **打包时做固定种子 shuffle** | `DistributedRangedSampler.__iter__` 实际是纯连续区间、**完全不 shuffle**；本数据集 `global_index` 按 class/subclass/topic 严格聚类，按原序打包会让每个 batch 都是同一主题 → 训练流同质化 |
| `.png` 字节原样直传 | 不重编码，打包前后 sha256 一致 |

打包后启动第一阶段 pixel-space 训练：

```bash
bash train_scripts/train_sana_pixel.sh
```

> Pixel config 使用明确的 `data_space: pixel`，RGB 会直接送入模型且不会创建 VAE。
> `load_vae_feat` 保留原本的“读取预计算 latent”语义，不能把它当成 pixel/latent 开关。

> **注意**：`Synthetic/Chinese_Text.json` / `English_Text.json` 的 `subclass` 字段是
> `"Chinese Text"` / `"English Text"`（**空格**），与文件名不一致。按你的规则「小类 **json 文件名称**」，
> 脚本以**文件名 stem** 为准，因此是 `Synthetic-Chinese_Text-0000.png`（下划线、无空格，路径对 shell/glob 友好）。
> 原始字段值保留在 manifest 的 `subclass_field` 列里。

## 断点续传与确定性

* 全局序号按 `(文件名字典序, data 数组序, prompts 数组序)` 展开，**先于**任何 `--files`/`--variant`/`--limit`
  选择确定，因此换分片方式、换子集、加 `--limit` 都不会改变某张图的 seed。
* `seed = seed_base + global_index`（`--seed-base` 默认 0）。每张图用**独立的** `torch.Generator`，
  所以噪声与运行顺序、分片方式、是否跳过都无关。
* 写入是原子的（先写 `*.tmp-<pid>.png` 再 `os.replace`），崩溃不会留下半张图被误判为已完成。
* 已实测：1 分片 vs 2 分片产出 **逐字节相同**；删除后重算 **逐字节相同**。
* 与官方 `scripts/inference.py` 在 seed 0 下 **逐位一致**（见下）。

### 与官方脚本对齐的实测证据

官方脚本只能存 JPG，而我们存 PNG。判定方法：把我们的 PNG 用**同样的 JPEG 设置**重新编码，
得到的字节与官方 JPG **完全相同**（91725 bytes）——JPEG 编码是确定性的，这等价于像素级逐位相等。
直接比 PNG 与官方 JPG 得到 PSNR 35.18 dB，该 MSE(19.7164) 与我们自己 PNG 的 JPEG 往返误差(19.7164)
完全相等，说明这点差异**全部来自 JPEG 有损压缩**，与生成结果无关。

## 多变体：每 prompt 生成多张图（`--repeats` / `--variant`）

`--repeats K` 让每个 prompt 产出 K 张图。样本下标是 **variant-major** 的：

```
prompt_index = 该 prompt 在 prompts 展开序里的位置 (0 … n_prompts-1)
global_index = (variant - 1) * n_prompts + prompt_index
seed         = seed_base + global_index
```

| variant | 文件名 | seed（seed_base=0, n_prompts=10000） |
| --- | --- | --- |
| 1 | `<Class>-<Subclass>-<NNNN>.png` | 0 … 9999 |
| 2 | `<Class>-<Subclass>-<NNNN>-v2.png` | 10000 … 19999 |
| v | `<Class>-<Subclass>-<NNNN>-v<v>.png` | (v-1)*10000 … v*10000-1 |

**关键性质**：variant 1 的 seed 与文件名和「单图（`--repeats 1`）时代完全一致」，因此

* 已有 10K 老数据在 `--repeats 2` 下会被 `--skip-existing` **全部跳过，一张都不重算**；
* 每个 prompt 的 K 张图 seed 互不相同、也互不重复（`--scan-only` 会打印 `duplicate seeds` 检查）。

把新图铺满所有 GPU 要用 `--variant` 过滤（否则 variant 1 占满前一半下标，前几张卡空转）：

```bash
# 0) 空跑校验，不碰 GPU：确认 variant 1 与盘上老数据完全重合、新 seed 无重复
python tools/dataset_gen/generate_t2i_dataset.py --scan-only --repeats 2 --gpu-ids 0,1,2,3

# 1) 只生成 variant 2（新的 10K），4 卡均分各 2500 张
bash output/gen_sana_t2i_dataset.sh --repeats 2 --variant 2 --gpu-ids 0,1,2,3
```

`--scan-only` 会打印每个 variant 的「planned / on disk / to generate」，这是判断"会不会重算老图"的直接证据。
已实测（2026-09-22）：`--repeats 2` 下 variant 1 = 10000 planned / 10000 on disk / **0 to generate**，
variant 1 的计划文件名集合与盘上 10K **完全相等**；20000 个 seed 全不重复。

## CLI 参考

| 参数 | 说明 |
| --- | --- |
| `--dataset-root` | 数据集根目录（含 `prompts/`），默认 `/home/ganchangyi/dataset/SANA-Pixel-Dataset` |
| `--config` / `--model-path` | 官方 config 与本地 `.pth`，默认已指向正确位置 |
| `--gpu-ids 0,1,2,3` | 逗号分隔的 GPU id；父进程为每张卡起一个 worker |
| `--files Design/Anime,Nature/Food` | 只跑子集（可写 `类/小类`、`小类` 或 `类`） |
| `--limit N` | 只跑全局顺序前 N 条（冒烟用） |
| `--repeats K` | 每 prompt 生成 K 张图（默认 1）；variant 1 沿用历史文件名与 seed，v≥2 加 `-v<v>` 后缀并使用不重叠的 seed 区间 |
| `--variant V` | 只生成第 V 个 variant（1 起）；用来把新图均分到所有卡 |
| `--seed-base 0` | seed 偏移 |
| `--skip-existing` / `--no-skip-existing` | 默认跳过已存在的 PNG（续传） |
| `--overwrite` | 强制重算（配合续传慎用） |
| `--max-failures 50` | 单 worker 失败超过此数则中止 |
| `--sha256` | 额外把每张 PNG 的 sha256 写入 manifest |
| `--aggregate` / `--no-aggregate` | 默认父进程渲染一条聚合进度条；后者各 worker 自己显示 |
| `--progress-interval 2` | 聚合进度刷新间隔（秒） |
| `--scan-only` | 只校验与打印计划，不碰 GPU |
| `--debug-fail-at 1,2` | 测试用：强制让这些全局序号失败（验证失败隔离） |

手工分片（多机/多 shell）：`--worker --shard-index r --num-shards W --gpu-id N`。

## 进度条

聚合模式下父进程每 2 秒汇总各 worker 的进度文件，渲染一行：

```
TOTAL:  32%|███       | 3210/10000 [01:12:04<04:07:33, 6.4s/img] eta 4:07:33 | 0.16 img/s | left 6790 | fail 0 | gpu0:812/2500 gpu1:800/2500 ...
```

含：总进度、已用时间、`<remaining>`、我自算的 `eta`、总吞吐、剩余张数、失败数、各卡明细。
`--no-aggregate` 时各 worker 用 tqdm 在自己的 `position` 输出自己的 ETA（单卡直跑更直观）。

## 性能与时间预算

### 实测校准（97 张 / 单张 A40 空闲 / 不含模型加载）

```
生成 97 张, 用时 520s -> 5.36 s/张, 0.187 img/s
```

| GPU 数 | 10000 张预计 | 含每卡约 45 s 模型加载 |
| --- | --- | --- |
| 1 | 14.9 h | 14.9 h |
| 2 | 7.4 h | 7.5 h |
| 4 | **3.7 h** | 3.7 h |

有其他任务占用 GPU 时实测会退化到 8.6–10.5 s/张（此时 4 卡约 6 h），
所以正式跑之前先看 `--limit 200` 的实测速率。每个 worker 各自加载一份模型/文本编码器/VAE，
**显存约 10–12 GB/卡**。

## 本机环境要求

主脚本不新增任何依赖，但运行前必须满足：

1. `conda activate pixel`（已补齐 SANA 推理依赖：`mmcv==1.7.2`、`flash-linear-attention==0.3.2`、
   `pyrallis` 等，均以 `--no-deps` 安装、未降级任何既有包）。
2. `HF_HUB_OFFLINE=1` —— `gemma-2-2b-it` 与 `mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers`
   已软链进 `~/.cache/huggingface/hub`，离线可加载。
3. `NO_PROXY=no_proxy=127.0.0.1,localhost` —— 本机 relay 设的 `NO_PROXY` 含 `[::1]`，
   httpx 会解析成 `Invalid port: ':1]'`。
4. PyPI 直连不通，如需装包用镜像 `https://mirror.sjtu.edu.cn/pypi/web/simple/`。

包装脚本 `output/gen_sana_t2i_dataset.sh` 已经把这些环境变量都设好了。

## 排障

| 现象 | 原因 / 处理 |
| --- | --- |
| `unrecognized arguments: --worker ...` | pyrallis 严格解析 `sys.argv`。脚本已在 worker 内清空 argv，若自行改动请保留该逻辑 |
| `FileNotFoundError ... .tmp-*.png` | 输出子目录不存在。`save_png_atomic()` 会 `mkdir -p`，若自行改动请保留 |
| 某张图反复失败 | 看 `metadata/failed.jsonl` 的 `error_type`/`error_message`，用 `--files` 或 `--debug-fail-at` 单独复现 |
| 重跑后 manifest 记录比预期多 | 不会：`finalize` 按 `global_index` **取时间戳最新**去重；`--files`/`--limit` 子集运行时会有 "manifest has N records but M were selected" 提示，属预期 |
| GPU 显存不足 | 换空闲卡；OOM 会被记为失败并继续，不影响其他图 |
| 想彻底重来 | 删 `metadata/parts/` 与 `images/`，或加 `--overwrite`（不删 manifest 历史） |
