# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Batch T2I dataset generator for SANA-1.5 1.6B 1024px (with CHI).

Reads every ``<dataset-root>/prompts/<Class>/<Subclass>.json`` prompt file and renders one
1024x1024 PNG per prompt using the *official* SANA-1.5 inference recipe:

* config  : ``configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml``
* params  : taken verbatim from ``scripts/inference.py::SanaInference`` defaults
            (cfg_scale=4.5, pag_scale=1.0, flow_dpm-solver/20 steps, flow_shift=3.0, bs=1)
* CHI     : enabled automatically because the config's ``text_encoder.chi_prompt`` is
            non-empty (208 instruction tokens + 300-token window)

The per-image generation path is a faithful re-implementation of
``scripts/inference.py::visualize`` (that function reads only module-level globals, so it
cannot be imported); ``scripts/inference.py`` itself is left untouched so it can still be
used as the ground truth for the parity test.

Key properties
--------------
* **Deterministic & shard-invariant** -- ``seed = seed_base + global_index`` where
  ``global_index`` is the prompt's position in the *full* dataset ordering.  The noise for a
  given image therefore does not depend on batch position, on how the work was sharded, or on
  whether earlier images were skipped.  Global index 0 is bit-identical to the official
  script at ``--seed 0``.
* **Resumable** -- existing, non-empty PNGs are skipped; writes are atomic
  (``tmp`` file + ``os.replace``) so a crash can never leave a half-written image behind.
* **Multi-GPU** -- one worker subprocess per GPU id; the parent renders a single aggregated
  progress bar with throughput and ETA.

Usage
-----
    python tools/dataset_gen/generate_t2i_dataset.py \
        --dataset-root /home/ganchangyi/dataset/SANA-Pixel-Dataset \
        --gpu-ids 3

See ``tools/dataset_gen/README.md`` for the full reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATASET_ROOT = "/home/ganchangyi/dataset/SANA-Pixel-Dataset"
DEFAULT_CONFIG = "configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml"
DEFAULT_MODEL_PATH = (
    "/home/ganchangyi/huggingface_ckpts/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth"
)

INDEX_PAD = 4
IMAGE_SUFFIX = ".png"
# Mirrors the dict in scripts/inference.py __main__.
SAMPLE_STEPS_DICT = {"dpm-solver": 20, "sa-solver": 25, "flow_dpm-solver": 20, "flow_euler": 28}
# Aspect-ratio flags that prepare_prompt_ar() would silently strip from a prompt.
AR_FLAG_RE = re.compile(r"--(?:aspect_ratio|ar|hw)\b")


# --------------------------------------------------------------------------------------
# dataset discovery
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ItemSpec:
    """One (prompt, variant) pair = one output image.

    With ``--repeats K`` every prompt yields ``K`` images.  ``global_index`` is the sample
    index in the full ``K x n_prompts`` space (variant-major), so it is unique over all
    samples and ``seed = seed_base + global_index`` never repeats.
    """

    global_index: int
    cls: str
    subclass: str
    topic: str
    topic_index: int
    num: int
    angle: str
    prompt: str
    source_json: str
    source_id: str
    out_rel: str
    seed: int
    #: 1-based variant of this prompt (``--repeats``).  1 keeps the historical filename
    #: ``<Class>-<Subclass>-<NNNN>.png``; ``v >= 2`` gets a ``-v<v>`` suffix.
    variant: int = 1
    #: Position of the source prompt in the prompt list; ``global_index - (variant-1)*n``.
    prompt_index: int = -1
    #: Raw ``subclass`` value from the JSON (differs from the filename stem for
    #: Synthetic/{Chinese,English}_Text.json, where it contains a space).  Kept for provenance.
    subclass_field: str = ""

    @property
    def key(self) -> str:
        return f"{self.cls}/{self.subclass}"


def _human(n: float) -> str:
    if n < 1024:
        return f"{n:.0f}B"
    if n < 1024**2:
        return f"{n/1024:.1f}KB"
    if n < 1024**3:
        return f"{n/1024**2:.1f}MB"
    return f"{n/1024**3:.2f}GB"


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def discover_items(
    dataset_root: Path,
    seed_base: int = 0,
    index_pad: int = INDEX_PAD,
    repeats: int = 1,
) -> Tuple[List[ItemSpec], List[str]]:
    """Expand every prompt JSON into a globally-indexed, deterministically ordered list.

    With ``repeats=K`` each prompt produces ``K`` items.  Sample indices are assigned
    **variant-major** over the complete ``K x n_prompts`` space::

        global_index = (variant - 1) * n_prompts + prompt_index
        seed         = seed_base + global_index

    Variant 1 therefore keeps ``seed == prompt_index`` and the historical filename
    ``<Class>-<Subclass>-<NNNN>.png`` -- i.e. ``--repeats 2`` reproduces the plan of an
    existing single-image run exactly, so ``--skip-existing`` skips all of it for free.
    Variant ``v >= 2`` is named ``<Class>-<Subclass>-<NNNN>-v<v>.png`` and draws seeds
    ``n_prompts .. K*n_prompts-1`` (disjoint from variant 1).

    ``global_index`` is assigned over the **complete** dataset, before any ``--files`` /
    ``--variant`` / ``--limit`` selection, so that seeds stay stable no matter how a run
    is sliced.
    """
    prompts_dir = dataset_root / "prompts"
    if not prompts_dir.is_dir():
        raise SystemExit(f"[fatal] prompts directory not found: {prompts_dir}")
    if repeats < 1:
        raise SystemExit(f"[fatal] --repeats must be >= 1, got {repeats}")

    json_files = sorted(prompts_dir.glob("*/*.json"))
    if not json_files:
        raise SystemExit(f"[fatal] no <Class>/<Subclass>.json under {prompts_dir}")

    entries: List[Dict[str, Any]] = []
    problems: List[str] = []

    for jf in json_files:
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{jf}: unreadable JSON ({exc})")
            continue

        cls = payload.get("class") or jf.parent.name
        # The user-facing naming rule is "<Class>-<json file name>-<index>", so the subclass
        # token is the *filename stem*, not the JSON's `subclass` field.  These differ for
        # Synthetic/{Chinese,English}_Text.json (field contains a space); the stem is used so
        # paths stay shell/glob friendly.
        subclass = jf.stem
        subclass_field = str(payload.get("subclass") or subclass)
        if subclass_field.lower().replace(" ", "_") != subclass.lower():
            problems.append(f"{jf}: subclass field {subclass_field!r} does not match filename stem {subclass!r}")
        if jf.parent.name != cls:
            problems.append(f"{jf}: parent dir {jf.parent.name!r} != class {cls!r}")

        blocks = payload.get("data")
        if not isinstance(blocks, list):
            problems.append(f"{jf}: missing/invalid 'data' list")
            continue

        counter = 0
        stats = payload.get("stats") or {}
        declared = stats.get("prompts_total")
        for topic_index, block in enumerate(blocks):
            topic = block.get("topic", "")
            for entry in block.get("prompts", []):
                prompt = entry.get("prompt")
                if not isinstance(prompt, str) or not prompt.strip():
                    problems.append(f"{jf}: empty prompt at {entry.get('id')!r}")
                    continue
                if AR_FLAG_RE.search(prompt):
                    problems.append(
                        f"{jf}: prompt {entry.get('id')!r} contains an --ar/--hw flag; "
                        "prepare_prompt_ar() would truncate it"
                    )
                if counter >= 10**index_pad:
                    raise SystemExit(
                        f"[fatal] {cls}/{subclass} needs more than {index_pad} digits for its index"
                    )
                entries.append(
                    {
                        "cls": cls,
                        "subclass": subclass,
                        "subclass_field": subclass_field,
                        "topic": topic,
                        "topic_index": topic_index,
                        "num": int(entry.get("num", counter + 1)),
                        "angle": entry.get("angle", ""),
                        "prompt": prompt,
                        "source_json": str(jf.relative_to(dataset_root)),
                        "source_id": entry.get("id", ""),
                        "stem": f"{cls}-{subclass}-{counter:0{index_pad}d}",
                    }
                )
                counter += 1

        if isinstance(declared, int) and declared != counter:
            problems.append(
                f"{jf}: stats.prompts_total={declared} but found {counter} prompts in 'data'"
            )

    n_prompts = len(entries)
    items: List[ItemSpec] = []
    seen_rel: Dict[str, str] = {}
    for variant in range(1, repeats + 1):
        suffix = "" if variant == 1 else f"-v{variant}"
        for prompt_index, e in enumerate(entries):
            gid = (variant - 1) * n_prompts + prompt_index
            out_rel = f"{e['cls']}/{e['subclass']}/{e['stem']}{suffix}{IMAGE_SUFFIX}"
            if out_rel in seen_rel:
                problems.append(f"duplicate output name {out_rel}")
            seen_rel[out_rel] = e["source_id"]
            items.append(
                ItemSpec(
                    global_index=gid,
                    cls=e["cls"],
                    subclass=e["subclass"],
                    topic=e["topic"],
                    topic_index=e["topic_index"],
                    num=e["num"],
                    angle=e["angle"],
                    prompt=e["prompt"],
                    source_json=e["source_json"],
                    source_id=e["source_id"],
                    out_rel=out_rel,
                    seed=seed_base + gid,
                    variant=variant,
                    prompt_index=prompt_index,
                    subclass_field=e["subclass_field"],
                )
            )

    return items, problems


def select_items(
    items: Sequence[ItemSpec],
    files: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    variant: Optional[int] = None,
) -> List[ItemSpec]:
    """Apply the ``--variant`` / ``--files`` / ``--limit`` selection on the globally indexed list."""
    selected: List[ItemSpec] = list(items)
    if variant is not None:
        available = sorted({it.variant for it in selected})
        selected = [it for it in selected if it.variant == variant]
        if not selected:
            raise SystemExit(f"[fatal] --variant {variant} matched nothing; available variants: {available}")
    if files:
        wanted = {f.strip().lower() for f in files if f.strip()}
        selected = [
            it
            for it in selected
            if it.key.lower() in wanted
            or it.subclass.lower() in wanted
            or it.cls.lower() in wanted
        ]
        missing = wanted - {it.key.lower() for it in selected} - {it.subclass.lower() for it in selected}
        if missing:
            raise SystemExit(f"[fatal] --files entries matched nothing: {sorted(missing)}")
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


def group_counts(items: Sequence[ItemSpec]) -> "Dict[str, int]":
    counts: Dict[str, int] = {}
    for it in items:
        counts[it.key] = counts.get(it.key, 0) + 1
    return counts


def shard_bounds(total: int, shard_index: int, num_shards: int) -> Tuple[int, int]:
    """Contiguous, balanced split: shards never overlap and cover ``[0, total)``."""
    if not 0 <= shard_index < num_shards:
        raise SystemExit(f"[fatal] shard-index must be in [0,{num_shards}); got {shard_index}")
    return total * shard_index // num_shards, total * (shard_index + 1) // num_shards


# --------------------------------------------------------------------------------------
# generation engine
# --------------------------------------------------------------------------------------
def guidance_type_select(pag_scale: float, attn_type: str) -> str:
    """Verbatim logic from scripts/inference.py: PAG only applies to linear attention."""
    if not (pag_scale > 1.0 and attn_type == "linear"):
        return "classifier-free"
    return "classifier-free_PAG"


class Engine:
    """Loads the official SANA-1.5 stack once and renders prompts one by one."""

    def __init__(self, cfg: Any, model_path: str, device: str):
        from diffusion.data.datasets.utils import ASPECT_RATIO_1024_TEST
        from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae
        from diffusion.model.utils import get_weight_dtype
        from diffusion.utils.config import model_init_config
        from tools.download import find_model

        self.cfg = cfg
        self.device = device
        self._get_weight_dtype = get_weight_dtype
        self._vae_decode = None
        self.base_ratios = ASPECT_RATIO_1024_TEST

        if cfg.sampling_algo != "flow_dpm-solver":
            raise SystemExit(
                f"[fatal] only the official default sampler 'flow_dpm-solver' is implemented; "
                f"config requests {cfg.sampling_algo!r}"
            )
        self.sampling_algo = cfg.sampling_algo
        self.sample_steps = cfg.step if cfg.step != -1 else SAMPLE_STEPS_DICT[cfg.sampling_algo]
        self.image_size = cfg.model.image_size
        self.latent_size = self.image_size // cfg.vae.vae_downsample_rate
        self.flow_shift = cfg.scheduler.flow_shift
        self.pag_applied_layers = cfg.model.get("pag_applied_layers")
        self.guidance_type = guidance_type_select(cfg.pag_scale, cfg.model.attn_type)
        self.weight_dtype = get_weight_dtype(cfg.model.mixed_precision)
        self.vae_dtype = get_weight_dtype(cfg.vae.weight_dtype)

        # --- verbatim scripts/inference.py::set_env ---
        torch.manual_seed(cfg.seed)
        torch.set_grad_enabled(False)
        for _ in range(30):
            torch.randn(1, 4, self.latent_size, self.latent_size)

        self.vae = get_vae(cfg.vae.vae_type, cfg.vae.vae_pretrained, device).to(self.vae_dtype)
        self.tokenizer, self.text_encoder = get_tokenizer_and_text_encoder(
            name=cfg.text_encoder.text_encoder_name, device=device
        )

        max_seq = cfg.text_encoder.model_max_length
        self.max_sequence_length = max_seq
        null_token = self.tokenizer(
            "", max_length=max_seq, padding="max_length", truncation=True, return_tensors="pt"
        ).to(device)
        self.null_caption_embs = self.text_encoder(null_token.input_ids, null_token.attention_mask)[0]

        self.model = build_model(
            cfg.model.model,
            use_fp32_attention=cfg.model.get("fp32_attention", False),
            **model_init_config(cfg, latent_size=self.latent_size),
        ).to(device)
        state_dict = find_model(model_path)
        if "pos_embed" in state_dict["state_dict"]:
            del state_dict["state_dict"]["pos_embed"]
        missing, unexpected = self.model.load_state_dict(state_dict["state_dict"], strict=False)
        self.missing_keys = list(missing)
        self.unexpected_keys = list(unexpected)
        self.model.eval().to(self.weight_dtype)
        self.num_params = sum(p.numel() for p in self.model.parameters())

        # --- CHI ---
        chi_lines = list(cfg.text_encoder.chi_prompt or [])
        self.chi_enabled = bool(chi_lines)
        self.chi_prompt = "\n".join(chi_lines)
        self.num_chi_tokens = len(self.tokenizer.encode(self.chi_prompt)) if self.chi_enabled else 0
        self.max_length_all = (
            self.num_chi_tokens + max_seq - 2  # magic number 2: [bos], [_]
            if self.chi_enabled
            else max_seq
        )
        self.select_index = [0] + list(range(-max_seq + 1, 0))

    @torch.inference_mode()
    def generate(self, item: ItemSpec) -> torch.Tensor:
        """Faithful re-implementation of scripts/inference.py::visualize for a single prompt."""
        if self._vae_decode is None:
            from diffusion.model.builder import vae_decode

            self._vae_decode = vae_decode
        from diffusion.model.utils import prepare_prompt_ar
        from diffusion.scheduler.dpm_solver import DPMS

        prompt_clean, _, hw, ar, _ = prepare_prompt_ar(
            item.prompt, self.base_ratios, device=self.device, show=False
        )
        prompts = [prompt_clean.strip()]

        if not self.chi_enabled:
            max_length_all = self.max_sequence_length
            prompts_all = prompts
        else:
            prompts_all = [self.chi_prompt + p for p in prompts]
            max_length_all = self.max_length_all

        caption_token = self.tokenizer(
            prompts_all, max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt"
        ).to(self.device)
        caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
            :, :, self.select_index
        ]
        emb_masks = caption_token.attention_mask[:, self.select_index]
        null_y = self.null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

        generator = torch.Generator(device=self.device).manual_seed(item.seed)
        z = torch.randn(
            len(prompts),
            self.cfg.vae.vae_latent_dim,
            self.latent_size,
            self.latent_size,
            device=self.device,
            generator=generator,
        )
        model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)

        dpm_solver = DPMS(
            self.model,
            condition=caption_embs,
            uncondition=null_y,
            guidance_type=self.guidance_type,
            cfg_scale=self.cfg.cfg_scale,
            pag_scale=self.cfg.pag_scale,
            pag_applied_layers=self.pag_applied_layers,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
            interval_guidance=self.cfg.interval_guidance,
        )
        samples = dpm_solver.sample(
            z,
            steps=self.sample_steps,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=self.flow_shift,
        )
        samples = samples.to(self.vae_dtype)
        samples = self._vae_decode(self.cfg.vae.vae_type, self.vae, samples)
        torch.cuda.empty_cache()
        return samples

    def describe(self) -> Dict[str, Any]:
        return {
            "model_class": self.model.__class__.__name__,
            "num_params": self.num_params,
            "weight_dtype": str(self.weight_dtype),
            "vae_dtype": str(self.vae_dtype),
            "image_size": self.image_size,
            "latent_size": self.latent_size,
            "sampling_algo": self.sampling_algo,
            "sample_steps": self.sample_steps,
            "cfg_scale": self.cfg.cfg_scale,
            "pag_scale": self.cfg.pag_scale,
            "guidance_type": self.guidance_type,
            "flow_shift": self.flow_shift,
            "pag_applied_layers": self.pag_applied_layers,
            "interval_guidance": list(self.cfg.interval_guidance),
            "chi_enabled": self.chi_enabled,
            "chi_prompt_lines": len(self.cfg.text_encoder.chi_prompt or []),
            "num_chi_tokens": self.num_chi_tokens,
            "model_max_length": self.max_sequence_length,
            "max_length_all": self.max_length_all,
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
        }


def save_png_atomic(sample: torch.Tensor, path: Path) -> None:
    """Write a single image atomically so a crash can never leave a partial PNG."""
    from torchvision.utils import save_image

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp-{os.getpid()}{path.suffix}")
    save_image(sample, str(tmp), nrow=1, normalize=True, value_range=(-1, 1))
    os.replace(tmp, path)


# --------------------------------------------------------------------------------------
# worker
# --------------------------------------------------------------------------------------
def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def run_worker(ns: argparse.Namespace, items_all: Sequence[ItemSpec]) -> int:
    from tqdm import tqdm

    dataset_root = Path(ns.dataset_root)
    images_dir = dataset_root / "images"
    metadata_dir = dataset_root / "metadata"
    logs_dir = dataset_root / "logs"
    parts_dir = metadata_dir / "parts"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    selected = select_items(items_all, ns.files, ns.limit, ns.variant)
    total_selected = len(selected)
    start, end = shard_bounds(total_selected, ns.shard_index, ns.num_shards)
    shard = selected[start:end]

    progress_path = logs_dir / f"progress-{ns.run_id}-{ns.gpu_id}.json"
    part_path = parts_dir / f"all.part-{ns.shard_index}.jsonl"
    fail_path = parts_dir / f"failed.part-{ns.shard_index}.jsonl"

    def out_path(item: ItemSpec) -> Path:
        return images_dir / item.out_rel

    def emit_progress(**kw: Any) -> None:
        _write_json_atomic(
            progress_path,
            {
                "run_id": ns.run_id,
                "gpu_id": ns.gpu_id,
                "shard_index": ns.shard_index,
                "num_shards": ns.num_shards,
                "shard_start": start,
                "shard_end": end,
                "updated_at": time.time(),
                **kw,
            },
        )

    # ---- pre-scan for resume -------------------------------------------------------
    pending: List[ItemSpec] = []
    skipped = 0
    for item in shard:
        target = out_path(item)
        if ns.overwrite or not ns.skip_existing:
            pending.append(item)
        elif target.is_file() and target.stat().st_size > 0:
            skipped += 1
        else:
            pending.append(item)
    already = len(shard) - len(pending)

    emit_progress(state="loading", total=len(shard), done=already, pending=len(pending), skipped=skipped)
    print(
        f"[gpu {ns.gpu_id}] shard {ns.shard_index}/{ns.num_shards} "
        f"items [{start},{end}) total={len(shard)} already={already} to_generate={len(pending)} "
        f"device={device}",
        flush=True,
    )

    if not pending:
        print(f"[gpu {ns.gpu_id}] nothing to do (all {len(shard)} images already exist)", flush=True)
        emit_progress(state="finished", total=len(shard), done=len(shard), pending=0, skipped=skipped,
                      elapsed=0.0, rate=0.0, eta=0.0)
        return 0

    engine = Engine(ns.parsed_cfg, ns.model_path, device)
    meta = engine.describe()
    print(f"[gpu {ns.gpu_id}] engine ready: {meta['model_class']} params={meta['num_params']:,} "
          f"dtype={meta['weight_dtype']} chi_tokens={meta['num_chi_tokens']} "
          f"max_length_all={meta['max_length_all']} steps={meta['sample_steps']} "
          f"guidance={meta['guidance_type']}", flush=True)

    run_start = time.time()
    done = already
    failures = 0
    gen_done = 0
    debug_fail = set(ns.debug_fail_at or [])

    progress_kwargs: Dict[str, Any] = {}
    if not ns.worker_quiet:
        progress_kwargs = dict(
            initial=already,
            total=len(shard),
            unit="img",
            desc=f"gpu{ns.gpu_id}",
            position=ns.position,
            leave=True,
            dynamic_ncols=True,
        )
    iterator = tqdm(pending, **progress_kwargs) if not ns.worker_quiet else pending

    with part_path.open("a", encoding="utf-8") as part_fh, fail_path.open("a", encoding="utf-8") as fail_fh:
        for item in iterator:
            target = out_path(item)
            try:
                if item.global_index in debug_fail:
                    raise RuntimeError(f"[debug] injected failure at global index {item.global_index}")
                t0 = time.time()
                samples = engine.generate(item)
                save_png_atomic(samples[0], target)
                gen_secs = time.time() - t0
                gen_done += 1
                record = {
                    "file": str(target.relative_to(dataset_root)),
                    "global_index": item.global_index,
                    "variant": item.variant,
                    "class": item.cls,
                    "subclass": item.subclass,
                    "subclass_field": item.subclass_field,
                    "topic": item.topic,
                    "topic_index": item.topic_index,
                    "num": item.num,
                    "angle": item.angle,
                    "source_json": item.source_json,
                    "source_id": item.source_id,
                    "prompt": item.prompt,
                    "seed": item.seed,
                    "image_size": engine.image_size,
                    "sample_steps": engine.sample_steps,
                    "sampling_algo": engine.sampling_algo,
                    "cfg_scale": engine.cfg.cfg_scale,
                    "pag_scale": engine.cfg.pag_scale,
                    "guidance_type": engine.guidance_type,
                    "flow_shift": engine.flow_shift,
                    "model_path": ns.model_path,
                    "config_path": ns.config,
                    "gen_seconds": round(gen_secs, 3),
                    "size_bytes": target.stat().st_size,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
                if ns.sha256:
                    record["sha256"] = _sha256(target)
                part_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                part_fh.flush()
            except Exception:  # noqa: BLE001
                failures += 1
                fail_fh.write(
                    json.dumps(
                        {
                            "global_index": item.global_index,
                            "variant": item.variant,
                            "source_id": item.source_id,
                            "file": item.out_rel,
                            "seed": item.seed,
                            "error_type": type(sys.exc_info()[1]).__name__,
                            "error_message": str(sys.exc_info()[1]),
                            "failed_at": datetime.now(timezone.utc).isoformat(),
                            "error": traceback.format_exc(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                fail_fh.flush()
                print(f"[gpu {ns.gpu_id}] FAILED {item.source_id}: {sys.exc_info()[1]}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if failures > ns.max_failures:
                    print(f"[gpu {ns.gpu_id}] aborting: failures {failures} > --max-failures {ns.max_failures}",
                          flush=True)
                    break

            done = already + gen_done
            elapsed = time.time() - run_start
            rate = gen_done / elapsed if elapsed > 0 else 0.0
            remaining = len(shard) - done
            emit_progress(
                state="running",
                total=len(shard),
                done=done,
                pending=len(pending),
                skipped=skipped,
                failures=failures,
                elapsed=elapsed,
                rate=rate,
                eta=remaining / rate if rate > 0 else 0.0,
                current=item.out_rel,
            )

    elapsed = time.time() - run_start
    rate = gen_done / elapsed if elapsed > 0 else 0.0
    emit_progress(
        state="failed" if failures > ns.max_failures else "finished",
        total=len(shard),
        done=already + gen_done,
        pending=len(pending),
        skipped=skipped,
        failures=failures,
        elapsed=elapsed,
        rate=rate,
        eta=0.0,
    )
    print(
        f"[gpu {ns.gpu_id}] done: generated={gen_done} skipped={skipped} failed={failures} "
        f"elapsed={fmt_duration(elapsed)} avg={rate:.2f} img/s",
        flush=True,
    )
    return 1 if failures > ns.max_failures else 0


# --------------------------------------------------------------------------------------
# parent
# --------------------------------------------------------------------------------------
def _read_progress(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _collect(progress_paths: Sequence[Path]) -> List[Dict[str, Any]]:
    out = []
    for p in progress_paths:
        data = _read_progress(p)
        if data:
            out.append(data)
    return out


def render_aggregate(records: Sequence[Dict[str, Any]], grand_total: int):
    done = sum(int(r.get("done", 0)) for r in records)
    rate = sum(float(r.get("rate", 0.0)) for r in records)
    failed = sum(int(r.get("failures", 0)) for r in records)
    remaining = max(0, grand_total - done)
    eta = remaining / rate if rate > 0 else 0.0
    return done, rate, failed, eta, remaining


def run_parent(ns: argparse.Namespace, items_all: Sequence[ItemSpec]) -> int:
    from tqdm import tqdm

    dataset_root = Path(ns.dataset_root)
    logs_dir = dataset_root / "logs"
    parts_dir = dataset_root / "metadata" / "parts"
    for d in (dataset_root / "images", dataset_root / "metadata", parts_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    selected = select_items(items_all, ns.files, ns.limit, ns.variant)
    grand_total = len(selected)
    gpu_ids = ns.gpu_ids
    num_shards = len(gpu_ids)

    progress_paths = [logs_dir / f"progress-{ns.run_id}-{gpu}.json" for gpu in gpu_ids]

    # Pre-count what is already on disk so the bar starts in the right place.
    initial_done = 0
    if ns.skip_existing and not ns.overwrite:
        for item in selected:
            target = dataset_root / "images" / item.out_rel
            if target.is_file() and target.stat().st_size > 0:
                initial_done += 1

    print("=" * 78)
    print(f"run_id        : {ns.run_id}")
    print(f"dataset root  : {dataset_root}")
    print(f"prompts       : {len(items_all)} total, {grand_total} selected")
    print(f"gpus          : {gpu_ids} ({num_shards} shards, contiguous)")
    print(f"already built : {initial_done}/{grand_total}")
    print(f"model         : {ns.model_path}")
    print(f"config        : {ns.config}")
    print("=" * 78, flush=True)

    procs: List[Tuple[int, subprocess.Popen, Any]] = []
    for slot, gpu in enumerate(gpu_ids):
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            "--worker",
            "--dataset-root", str(dataset_root),
            "--config", ns.config,
            "--model-path", ns.model_path,
            "--gpu-id", str(gpu),
            "--shard-index", str(slot),
            "--num-shards", str(num_shards),
            "--run-id", ns.run_id,
            "--seed-base", str(ns.seed_base),
            "--repeats", str(ns.repeats),
            "--max-failures", str(ns.max_failures),
            "--position", str(slot),
        ]
        if ns.files:
            cmd += ["--files", ",".join(ns.files)]
        if ns.limit is not None:
            cmd += ["--limit", str(ns.limit)]
        if ns.variant is not None:
            cmd += ["--variant", str(ns.variant)]
        if ns.overwrite:
            cmd.append("--overwrite")
        if not ns.skip_existing:
            cmd.append("--no-skip-existing")
        if ns.sha256:
            cmd.append("--sha256")
        if ns.debug_fail_at:
            cmd += ["--debug-fail-at", ",".join(str(i) for i in ns.debug_fail_at)]
        if ns.aggregate:
            cmd.append("--worker-quiet")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = logs_dir / f"gpu-{gpu}.log"
        if ns.aggregate:
            # Workers are quiet in aggregate mode, so capture their output instead of the tty.
            log_fh = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT, env=env)
        else:
            # Let workers own the tty so their per-GPU tqdm bars are visible.
            log_fh = None
            proc = subprocess.Popen(cmd, stderr=subprocess.STDOUT, env=env)
        procs.append((gpu, proc, log_fh))
        print(f"  launched gpu {gpu} -> shard {slot}/{num_shards} (pid {proc.pid}, log {log_path})")

    if not ns.aggregate:
        print("\nwaiting for workers (per-GPU bars are shown by the workers themselves)...\n", flush=True)

    bar_kwargs: Dict[str, Any] = {}
    # In a batch job stdout is a log file, not a tty: tqdm's carriage-return redraws would
    # shred the log, so fall back to one plain progress line per interval.
    use_tty_bar = ns.aggregate and sys.stdout.isatty()
    if not use_tty_bar:
        bar_kwargs = dict(disable=True)
    bar = tqdm(
        total=grand_total,
        initial=initial_done,
        unit="img",
        desc="TOTAL",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        **bar_kwargs,
    )

    interrupted = False
    t_start = time.time()
    # Workers are separate processes that can be SIGKILLed out from under us (cgroup OOM, node
    # faults, ...).  Detect that instead of waiting forever for a progress file that will never
    # advance again; the healthy workers are left to finish so their GPU time is not wasted.
    dead_workers: Dict[int, str] = {}
    try:
        while True:
            alive = [p for _, p, _ in procs if p.poll() is None]
            records = _collect(progress_paths)
            state_by_gpu = {r.get("gpu_id"): r.get("state") for r in records}
            for gpu, proc, _ in procs:
                if proc.poll() is None or gpu in dead_workers:
                    continue
                state = state_by_gpu.get(gpu)
                if proc.returncode != 0 or state != "finished":
                    dead_workers[gpu] = f"exit={proc.returncode} last_state={state}"
                    print(
                        f"[parent] WARNING: gpu {gpu} worker died unexpectedly "
                        f"({dead_workers[gpu]}); NOT a clean finish. Remaining workers keep "
                        f"running; re-run the same command afterwards to resume.",
                        flush=True,
                    )
            done, rate, failed, eta, remaining = render_aggregate(records, grand_total)
            per_gpu = " ".join(
                f"gpu{r.get('gpu_id')}:{r.get('done')}/{r.get('total')}"
                for r in sorted(records, key=lambda x: x.get("gpu_id", 0))
            )
            status = f"eta {fmt_duration(eta)} | {rate:.2f} img/s | left {remaining} | fail {failed} | {per_gpu}"
            if dead_workers:
                status += f" | DEAD {sorted(dead_workers)}"
            if use_tty_bar:
                bar.n = min(grand_total, done)
                bar.set_postfix_str(status)
                bar.refresh()
            elif ns.aggregate:
                print(f"[{fmt_duration(time.time() - t_start)}] {done}/{grand_total} | {status}", flush=True)
            if not alive:
                break
            time.sleep(ns.progress_interval)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[parent] interrupted -- terminating workers (progress is preserved; rerun to resume)")
        for _, p, _ in procs:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        for _, p, _ in procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
    finally:
        bar.close()
        for _, _, fh in procs:
            if fh is not None:
                fh.close()

    records = _collect(progress_paths)
    done, rate, failed, eta, remaining = render_aggregate(records, grand_total)
    print("\n" + "=" * 78)
    print(f"generated={done - initial_done}  skipped={initial_done}  failed={failed}  total={grand_total}")
    print(f"throughput   : {rate:.2f} img/s" + (f"  ({rate*3600:.0f} img/h)" if rate else ""))
    for gpu, proc, _ in procs:
        print(f"  gpu {gpu}: exit={proc.returncode}" + (f"  <- {dead_workers[gpu]}" if gpu in dead_workers else ""))
    if dead_workers:
        print(
            f"\n[warn] {len(dead_workers)} worker(s) died unexpectedly: {dead_workers}\n"
            f"       Images already written are valid and will be skipped on re-run.\n"
            f"       Resume with:  bash tools/dataset_gen/sbatch_t2i_dataset.sh   (or sbatch ...)",
            flush=True,
        )
    print("=" * 78, flush=True)

    finalize(ns, items_all, selected, records)
    if interrupted:
        return 130
    if dead_workers or any(p.returncode not in (0, None) for _, p, _ in procs):
        return 1
    return 0


# --------------------------------------------------------------------------------------
# manifest finalisation
# --------------------------------------------------------------------------------------
MANIFEST_COLUMNS = [
    "file", "global_index", "variant", "class", "subclass", "subclass_field", "topic", "topic_index", "num",
    "angle", "source_json", "source_id", "prompt", "seed", "image_size", "sample_steps", "sampling_algo",
    "cfg_scale", "pag_scale", "guidance_type", "flow_shift", "gen_seconds", "size_bytes",
    "generated_at", "model_path", "config_path",
]


def finalize(
    ns: argparse.Namespace,
    items_all: Sequence[ItemSpec],
    selected: Sequence[ItemSpec],
    records: Sequence[Dict[str, Any]],
) -> None:
    dataset_root = Path(ns.dataset_root)
    metadata_dir = dataset_root / "metadata"
    parts_dir = metadata_dir / "parts"

    merged_by_index: Dict[int, Dict[str, Any]] = {}
    for part in sorted(parts_dir.glob("all.part-*.jsonl")):
        for line in part.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = int(rec["global_index"])
            prev = merged_by_index.get(idx)
            # Workers append across runs and the same index can appear in more than one
            # part file (sharding changed between runs), so keep the **newest** record
            # rather than whichever file happened to be read last.
            if prev is None or str(rec.get("generated_at", "")) >= str(prev.get("generated_at", "")):
                merged_by_index[idx] = rec
    merged = [merged_by_index[k] for k in sorted(merged_by_index)]

    all_path = metadata_dir / "all.jsonl"
    with all_path.open("w", encoding="utf-8") as fh:
        for rec in merged:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    columns = list(MANIFEST_COLUMNS) + (["sha256"] if ns.sha256 else [])
    with (metadata_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rec in merged:
            writer.writerow({k: rec.get(k, "") for k in columns})

    failures_by_index: Dict[int, Dict[str, Any]] = {}
    for part in sorted(parts_dir.glob("failed.part-*.jsonl")):
        for line in part.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            idx = int(rec["global_index"])
            prev = failures_by_index.get(idx)
            if prev is None or str(rec.get("failed_at", "")) >= str(prev.get("failed_at", "")):
                failures_by_index[idx] = rec
    # A failure is only resolved if that index was generated *after* it failed.  A stale
# success from an earlier run must not hide an --overwrite that just fell over.
    failures = [
        failures_by_index[k]
        for k in sorted(failures_by_index)
        if not (
            k in merged_by_index
            and str(merged_by_index[k].get("generated_at", "")) > str(failures_by_index[k].get("failed_at", ""))
        )
    ]
    with (metadata_dir / "failed.jsonl").open("w", encoding="utf-8") as fh:
        for rec in failures:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    expected_rels = {it.out_rel for it in selected}
    on_disk = {
        str(p.relative_to(dataset_root / "images"))
        for p in (dataset_root / "images").rglob(f"*{IMAGE_SUFFIX}")
    }
    missing = expected_rels - on_disk

    run_config = {
        "run_id": ns.run_id,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "config_path": ns.config,
        "config_sha256": _sha256(Path(REPO_ROOT / ns.config)) if (REPO_ROOT / ns.config).is_file() else None,
        "model_path": ns.model_path,
        "model_size_bytes": Path(ns.model_path).stat().st_size if Path(ns.model_path).is_file() else None,
        "model_mtime": Path(ns.model_path).stat().st_mtime if Path(ns.model_path).is_file() else None,
        "gpu_ids": ns.gpu_ids,
        "num_shards": len(ns.gpu_ids),
        "seed_base": ns.seed_base,
        "seed_rule": "seed = seed_base + global_index "
        "(global_index = (variant-1)*n_prompts + prompt_index over the full dataset)",
        "repeats": ns.repeats,
        "variant_filter": ns.variant,
        "files_filter": ns.files,
        "limit": ns.limit,
        "expected_images": len(selected),
        "manifest_records": len(merged),
        "images_on_disk": len(on_disk),
        "failed_records": len(failures),
        "elapsed_seconds": sum(float(r.get("elapsed", 0.0)) for r in records),
        "workers": records,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else [],
        "python_executable": sys.executable,
        "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "CUDA_VISIBLE_DEVICES", "NO_PROXY", "PYTHONPATH")},
    }
    (metadata_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n--- manifest ---")
    print(f"  {all_path}")
    print(f"  {metadata_dir / 'manifest.csv'}")
    print(f"  {metadata_dir / 'failed.jsonl'}  ({len(failures)} records)")
    print(f"  {metadata_dir / 'run_config.json'}")
    print(f"  records={len(merged)}  on_disk={len(on_disk)}  expected={len(selected)}")
    if missing:
        print(f"  [warn] {len(missing)} expected image(s) still missing, e.g. {sorted(missing)[:3]}")
    if len(merged) != len(selected):
        print(
            f"  [warn] manifest has {len(merged)} records but {len(selected)} were selected "
            f"({len(selected) - len(merged)} not generated yet)"
        )


# --------------------------------------------------------------------------------------
# scan-only report
# --------------------------------------------------------------------------------------
def run_scan(ns: argparse.Namespace, items_all: Sequence[ItemSpec], problems: Sequence[str]) -> int:
    selected = select_items(items_all, ns.files, ns.limit, ns.variant)
    counts = group_counts(items_all)
    name_re = re.compile(rf"^[A-Za-z_]+-[A-Za-z_]+-\d{{{INDEX_PAD}}}(-v\d+)?{IMAGE_SUFFIX}$")
    images_dir = Path(ns.dataset_root) / "images"

    print(f"dataset root : {ns.dataset_root}")
    print(f"repeats      : {ns.repeats}   (variants: {sorted({it.variant for it in items_all})})")
    print(f"prompts total: {len(items_all) // max(ns.repeats, 1)} x {ns.repeats} = {len(items_all)} images")
    print(f"selected     : {len(selected)}" + (f"   (--variant {ns.variant})" if ns.variant else ""))
    print(f"seed range   : [{items_all[0].seed}, {items_all[-1].seed}]  ({len({it.seed for it in items_all})} distinct)"
          if items_all else "seed range   : n/a")
    print("\nper subclass (all variants, planned):")
    for key in sorted(counts):
        print(f"  {key:28s} {counts[key]:5d}")

    print("\nper variant (all items / already on disk / to generate):")
    for v in sorted({it.variant for it in items_all}):
        vs = [it for it in items_all if it.variant == v]
        existing = sum(1 for it in vs if (images_dir / it.out_rel).is_file())
        seeds = {it.seed for it in vs}
        print(f"  variant {v}: {len(vs):6d} planned, {existing:6d} on disk, {len(vs) - existing:6d} to generate, "
              f"seeds [{min(seeds)}..{max(seeds)}]")
    sel_existing = sum(1 for it in selected if (images_dir / it.out_rel).is_file())
    print(f"  -> selected: {len(selected)} planned, {sel_existing} already on disk, "
          f"{len(selected) - sel_existing} would be generated")

    bad = [it.out_rel for it in items_all if not name_re.match(Path(it.out_rel).name)]
    dup = len(items_all) - len({it.out_rel for it in items_all})
    dup_seed = len(items_all) - len({it.seed for it in items_all})
    print("\nchecks:")
    print(f"  name format violations : {len(bad)}")
    print(f"  duplicate output names : {dup}")
    print(f"  duplicate seeds        : {dup_seed}")
    print(f"  data problems          : {len(problems)}")
    for p in problems[:10]:
        print(f"    - {p}")
    if selected:
        print("\nfirst 3:")
        for it in selected[:3]:
            print(f"  v{it.variant} {it.out_rel}  seed={it.seed}  <= {it.source_id}")
        print("last 3:")
        for it in selected[-3:]:
            print(f"  v{it.variant} {it.out_rel}  seed={it.seed}  <= {it.source_id}")
    if ns.gpu_ids:
        print(f"\nshard plan for gpu_ids={ns.gpu_ids}:")
        for slot, gpu in enumerate(ns.gpu_ids):
            s, e = shard_bounds(len(selected), slot, len(ns.gpu_ids))
            print(f"  gpu {gpu}: items [{s},{e}) -> {e - s}")
    ok = not problems and not bad and dup == 0 and dup_seed == 0
    print("\nRESULT:", "OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch T2I dataset generator for SANA-1.5 1.6B 1024px (CHI enabled).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="official inference config")
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="local SANA-1.5 .pth checkpoint")
    p.add_argument("--gpu-ids", default="0", help="comma separated GPU ids, e.g. 0,1,2,3")
    p.add_argument("--files", default=None, help="comma separated Class/Subclass (or Subclass) subset")
    p.add_argument("--limit", type=int, default=None, help="only the first N items of the global order")
    p.add_argument("--repeats", type=int, default=1,
                   help="generate K images per prompt; variant 1 keeps the historical names/seeds, "
                        "variant v>=2 is <name>-v<v>.png with a disjoint seed range")
    p.add_argument("--variant", type=int, default=None,
                   help="only generate this 1-based variant (use it to spread the new images over all GPUs)")
    p.add_argument("--seed-base", type=int, default=0, help="seed = seed_base + global_index")
    p.add_argument("--index-pad", type=int, default=INDEX_PAD, help="zero padding of the file index")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--overwrite", action="store_true", help="regenerate even if the PNG exists")
    p.add_argument("--max-failures", type=int, default=50, help="abort a worker after this many failures")
    p.add_argument("--sha256", action="store_true", help="also hash every PNG into the manifest (slower)")
    p.add_argument("--aggregate", dest="aggregate", action="store_true", default=True,
                   help="parent renders one aggregated progress bar")
    p.add_argument("--no-aggregate", dest="aggregate", action="store_false",
                   help="workers render their own per-GPU bars")
    p.add_argument("--progress-interval", type=float, default=2.0)
    p.add_argument("--scan-only", action="store_true", help="validate + print the plan, never touch the GPU")
    p.add_argument("--debug-fail-at", default=None, help="comma separated global indices to force-fail (testing)")
    # internal / worker mode
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--gpu-id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--run-id", default=None, help=argparse.SUPPRESS)
    p.add_argument("--worker-quiet", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--position", type=int, default=0, help=argparse.SUPPRESS)
    return p


def parse_gpu_ids(raw: str) -> List[int]:
    ids: List[int] = []
    for token in str(raw).replace(" ", "").split(","):
        if not token:
            continue
        try:
            ids.append(int(token))
        except ValueError:
            raise SystemExit(f"[fatal] --gpu-ids expects integers, got {token!r}")
    if not ids:
        raise SystemExit("[fatal] --gpu-ids is empty")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"[fatal] duplicate gpu ids: {ids}")
    return ids


def main(argv: Optional[Sequence[str]] = None) -> int:
    ns = build_parser().parse_args(argv)
    ns.dataset_root = str(Path(ns.dataset_root).expanduser().resolve())
    ns.files = [f for f in ns.files.split(",") if f.strip()] if ns.files else None
    ns.gpu_ids = parse_gpu_ids(ns.gpu_ids)
    ns.debug_fail_at = [int(x) for x in ns.debug_fail_at.split(",") if x.strip()] if ns.debug_fail_at else None
    if ns.run_id is None:
        ns.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    if ns.repeats < 1:
        raise SystemExit(f"[fatal] --repeats must be >= 1, got {ns.repeats}")
    if ns.variant is not None and not (1 <= ns.variant <= ns.repeats):
        raise SystemExit(
            f"[fatal] --variant {ns.variant} is out of range for --repeats {ns.repeats}; "
            f"add --repeats {ns.variant} (or larger)"
        )

    items_all, problems = discover_items(Path(ns.dataset_root), ns.seed_base, ns.index_pad, ns.repeats)
    if not items_all:
        raise SystemExit("[fatal] no prompts discovered")

    if ns.scan_only:
        return run_scan(ns, items_all, problems)
    for p in problems[:10]:
        print(f"[warn] {p}")

    if ns.worker:
        import pyrallis
        from scripts.inference import SanaInference

        # pyrallis.parse() consumes sys.argv strictly, and would reject our own internal flags
        # (--worker, --gpu-id, ...).  Everything it needs is passed explicitly via config_path,
        # so hide argv from it while it parses the official config.
        saved_argv = sys.argv
        sys.argv = [saved_argv[0]]
        try:
            ns.parsed_cfg = pyrallis.parse(config_class=SanaInference, config_path=ns.config)
        finally:
            sys.argv = saved_argv
        return run_worker(ns, items_all)

    return run_parent(ns, items_all)


if __name__ == "__main__":
    sys.exit(main())