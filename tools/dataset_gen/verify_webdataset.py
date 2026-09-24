#!/usr/bin/env python
"""Exercise the *real* official dataloader against the packed WebDataset shards.

This is the end-to-end gate: it builds ``SanaWebDatasetMS`` exactly the way
``train_scripts/train.py`` does (same class, same kwargs shape) and then

* AC6: checks ``len``, pulls samples, and walks **every** index through
  ``get_data_info`` to prove no sample is silently dropped (a missing
  ``height``/``width`` in ``<key>.json`` makes ``get_data_info`` return ``None``
  and the batch sampler would skip it without any error).
* AC7: runs ``AspectRatioBatchSampler`` over ``DistributedRangedSampler`` and a
  real ``DataLoader``, checking tensor shapes and that a batch mixes classes
  (proof that the pack-time shuffle worked; ``DistributedRangedSampler`` itself
  does no shuffling).
* AC8 (optional): if ``<key>.npy`` latents are present, verifies the
  ``load_vae_feat=True`` path returns ``(32, 32, 32)`` latents.

Usage::

    python tools/dataset_gen/verify_webdataset.py
    python tools/dataset_gen/verify_webdataset.py --scan-limit 500 --batches 2
"""

from __future__ import annotations

import argparse
import json
import os.path as osp
import sys
from collections import Counter

DEFAULT_OUT = "/home/ganchangyi/dataset/SANA-Pixel-Dataset/webdataset"
RESOLUTION = 1024
ASPECT_RATIO_TYPE = "ASPECT_RATIO_1024"


def log(msg: str) -> None:
    print(msg, flush=True)


def build_dataset(data_dir: str, load_vae_feat: bool):
    from diffusion.data.datasets.sana_data_multi_scale import SanaWebDatasetMS
    from diffusion.data.transforms import get_transform

    # num_replicas must be explicit: SanaWebDataset._initialize_dataset calls
    # dist.get_world_size() eagerly, which raises outside a process group.
    return SanaWebDatasetMS(
        data_dir=data_dir,
        resolution=RESOLUTION,
        transform=None if load_vae_feat else get_transform("default_train", RESOLUTION),
        max_length=300,
        num_replicas=1,
        aspect_ratio_type=ASPECT_RATIO_TYPE,
        load_vae_feat=load_vae_feat,
        load_text_feat=False,
        caption_proportion={"prompt": 1},
        caption_selection_type="proportion",
        config=None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_OUT)
    ap.add_argument("--scan-limit", type=int, default=None, help="how many indices to walk in AC6 (default: all)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--check-latents", action="store_true")
    args = ap.parse_args()

    data_dir = osp.abspath(osp.expanduser(args.data_dir))
    wids = osp.join(data_dir, "wids-meta.json")
    if not osp.exists(wids):
        log(f"[verify-loader] ERROR: {wids} not found")
        return 2
    spec = json.load(open(wids, encoding="utf-8"))
    n_total = sum(s["nsamples"] for s in spec["shardlist"])
    log(f"[verify-loader] data_dir={data_dir}")
    log(f"[verify-loader] wids-meta.json: {len(spec['shardlist'])} shards, {n_total} samples")

    log("[verify-loader] building SanaWebDatasetMS (official class, train.py kwargs shape) ...")
    ds = build_dataset(data_dir, load_vae_feat=False)
    log(f"[verify-loader] len(dataset) = {len(ds)}  ori_imgs_nums = {ds.ori_imgs_nums}")
    assert len(ds) == n_total, f"len(dataset)={len(ds)} != {n_total}"

    # --- AC6a: one fully transformed sample -----------------------------------
    img, txt, mask, data_info, idx, caption_type, dataindex_info = ds[0]
    log(f"[verify-loader] sample[0]: img={tuple(img.shape)} {img.dtype} "
        f"range=[{float(img.min()):.3f},{float(img.max()):.3f}]")
    log(f"[verify-loader] sample[0]: prompt[:60]={txt[:60]!r} caption_type={caption_type!r}")
    di_fmt = {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in data_info.items()}
    log(f"[verify-loader] sample[0]: data_info={di_fmt} shard={dataindex_info['shard']}")
    assert tuple(img.shape) == (3, RESOLUTION, RESOLUTION), img.shape
    assert isinstance(txt, str) and txt.strip(), "empty prompt"
    assert caption_type == "prompt", caption_type
    assert float(data_info["aspect_ratio"]) == 1.0, data_info["aspect_ratio"]

    # --- AC6b: every index must survive get_data_info --------------------------
    limit = args.scan_limit if args.scan_limit is not None else len(ds)
    log(f"[verify-loader] AC6: walking get_data_info() for {limit} indices ...")
    keys = set()
    dropped = []
    bad_hw = []
    for i in range(limit):
        info = ds.get_data_info(i)
        if info is None:
            dropped.append(i)
            continue
        if (info["height"], info["width"]) != (RESOLUTION, RESOLUTION):
            bad_hw.append((i, info["height"], info["width"]))
        keys.add(info["key"])
    if dropped:
        raise AssertionError(f"{len(dropped)} indices dropped by get_data_info, e.g. {dropped[:10]}")
    if bad_hw:
        raise AssertionError(f"{len(bad_hw)} samples with wrong height/width, e.g. {bad_hw[:5]}")
    log(f"[verify-loader] OK: 0 dropped, all {limit} samples 1024x1024, {len(keys)} unique keys")

    # --- AC7: batch sampler + real DataLoader ---------------------------------
    from diffusion.data.builder import custom_collate_fn, build_dataloader
    from diffusion.data.wids import DistributedRangedSampler
    from diffusion.utils.data_sampler import AspectRatioBatchSampler

    sampler = DistributedRangedSampler(ds, num_replicas=1, rank=0)
    batch_sampler = AspectRatioBatchSampler(
        sampler=sampler,
        dataset=ds,
        batch_size=args.batch_size,
        aspect_ratios=ds.aspect_ratio,
        drop_last=True,
        ratio_nums=ds.ratio_nums,
        config=None,
        valid_num=0,
        hq_only=False,
        clipscore_filter_thres=0.0,
    )
    log(f"[verify-loader] AC7: {len(batch_sampler)} batches of {args.batch_size}")

    # class mix per batch, read straight from the tar json (no image decode)
    for bi, batch in enumerate(batch_sampler):
        if bi >= args.batches:
            break
        classes = Counter(ds.dataset[i][".json"]["class"] for i in batch)
        ratios = set()
        for i in batch:
            info = ds.get_data_info(i)
            ratios.add(info["height"] / info["width"])
        log(f"[verify-loader]   batch {bi}: {len(batch)} idx {batch[0]}..{batch[-1]} "
            f"classes={dict(classes)} ratios={ratios}")
        assert len(classes) > 1, f"batch {bi} is class-homogeneous -> pack-time shuffle is not effective"
        assert ratios == {1.0}, f"batch {bi} mixes aspect ratios: {ratios}"

    loader = build_dataloader(
        ds,
        batch_sampler=AspectRatioBatchSampler(
            sampler=DistributedRangedSampler(ds, num_replicas=1, rank=0),
            dataset=ds,
            batch_size=args.batch_size,
            aspect_ratios=ds.aspect_ratio,
            drop_last=True,
            ratio_nums=ds.ratio_nums,
            config=None,
            valid_num=0,
        ),
        num_workers=0,
    )
    log("[verify-loader] AC7: pulling real batches through DataLoader ...")
    for bi, batch in enumerate(loader):
        if bi >= 2:
            break
        images, prompts, attn_mask, d_info, indices, ctype, dinfo = batch
        log(f"[verify-loader]   loader batch {bi}: images={tuple(images.shape)} {images.dtype} "
            f"attn={tuple(attn_mask.shape)} img_hw={tuple(d_info['img_hw'].shape)} "
            f"ar={tuple(d_info['aspect_ratio'].shape)} prompts={len(prompts)}")
        assert tuple(images.shape) == (args.batch_size, 3, RESOLUTION, RESOLUTION), images.shape
        assert isinstance(prompts[0], str)
    log("[verify-loader] OK: DataLoader + custom_collate_fn produce a valid 7-tuple batch")

    # --- AC8 (optional): precomputed latent path ------------------------------
    if args.check_latents:
        log("[verify-loader] AC8: building with load_vae_feat=True ...")
        ds_vae = build_dataset(data_dir, load_vae_feat=True)
        z = ds_vae[0][0]
        log(f"[verify-loader] latent[0]: shape={tuple(z.shape)} dtype={z.dtype}")
        assert tuple(z.shape) == (32, 32, 32), z.shape
        log("[verify-loader] OK: .npy latent path works")

    log("[verify-loader] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())