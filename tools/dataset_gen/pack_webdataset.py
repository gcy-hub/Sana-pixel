#!/usr/bin/env python
"""Pack SANA-Pixel-Dataset into WebDataset shards for the official loader.

Produces ``<out-dir>/shards/shard-XXXXX.tar`` (each member is
``<key>.png`` + ``<key>.json``, optionally ``<key>.npy``) plus
``<out-dir>/wids-meta.json`` so that ``SanaWebDatasetMS`` can read it with
``data_dir=<out-dir>`` and **no copy, no download, no network**.

Design constraints (each one traced to loader code, see tools/dataset_gen/README.md):

* ``<key>.json`` must carry ``prompt``/``height``/``width`` -- ``SanaWebDatasetMS``
  reads ``info["height"]`` directly and a missing key makes ``get_data_info``
  swallow the sample silently.
* at most 10 shards -- ``lru_size`` is hard-coded to 10 in ``SanaWebDataset``.
* USTAR format, plain basenames, ``REGTYPE`` members only -- ``MMIndexedTar``
  parses tar headers by hand and skips PAX headers / directories.
* the record order is shuffled with a fixed seed before sharding, because
  ``DistributedRangedSampler`` iterates a contiguous range without shuffling
  and this dataset's ``global_index`` order is class/topic clustered.

Usage::

    python tools/dataset_gen/pack_webdataset.py                    # pack
    python tools/dataset_gen/pack_webdataset.py --verify-only      # check existing shards
    python tools/dataset_gen/pack_webdataset.py --latents-dir <dir>  # also embed <key>.npy
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import os.path as osp
import random
import sys
import tarfile
from collections import Counter

DEFAULT_ROOT = "/home/ganchangyi/dataset/SANA-Pixel-Dataset"
DEFAULT_OUT = osp.join(DEFAULT_ROOT, "webdataset")

# Hard requirement of SanaWebDatasetMS.getdata / get_data_info.
JSON_REQUIRED = ("file_name", "prompt", "height", "width")
# Extra provenance, ignored by the loader but useful for traceability.
JSON_EXTRA = (
    "global_index", "variant", "prompt_index", "class", "subclass", "topic", "angle", "seed", "source_id",
)
# SanaWebDataset hard-codes lru_size=10; more shards than that thrash the LRU.
LRU_SIZE = 10


def log(msg: str) -> None:
    print(msg, flush=True)


def human(n: float) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024.0
    return f"{n:.1f}T"


def load_metadata(root: str):
    path = osp.join(root, "metadata.json")
    if not osp.exists(path):
        raise FileNotFoundError(f"{path} not found; run consolidate_dataset.py first")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    records = doc["records"] if isinstance(doc, dict) else doc
    return doc, records


def rec_path(root: str, record) -> str:
    p = record["file"]
    return p if osp.isabs(p) else osp.join(root, p)


def key_for(record) -> str:
    """WebDataset sample key: basename without extension (globally unique here)."""
    return osp.splitext(osp.basename(record["file"]))[0]


def sample_json(record, height: int, width: int) -> bytes:
    payload = {
        "file_name": osp.basename(record["file"]),
        "prompt": record["prompt"],
        "height": int(height),
        "width": int(width),
    }
    for k in JSON_EXTRA:
        if k in record:
            payload[k] = record[k]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def read_png_size(path: str):
    """Read width/height from the PNG header only (no full decode)."""
    from PIL import Image

    with Image.open(path) as im:
        return im.size  # (w, h)


def full_verify_image(path: str) -> None:
    from PIL import Image

    with Image.open(path) as im:
        im.verify()


def tar_info(name: str, size: int) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name)
    ti.size = size
    ti.mtime = 0
    ti.mode = 0o644
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.type = tarfile.REGTYPE
    return ti


def plan_order(records, shuffle_seed, shuffle: bool):
    order = list(range(len(records)))
    if shuffle:
        random.Random(shuffle_seed).shuffle(order)
    return [records[i] for i in order]


def write_shard(path: str, items, root: str, latents_dir, strict: bool):
    """Write one USTAR shard atomically; returns (nsamples, nbytes)."""
    tmp = f"{path}.tmp-{os.getpid()}"
    total = 0
    with tarfile.open(tmp, "w", format=tarfile.USTAR_FORMAT) as tar:
        for record in items:
            key = key_for(record)
            src = rec_path(root, record)
            if not osp.exists(src):
                raise FileNotFoundError(src)

            w, h = read_png_size(src)
            expected = int(record.get("image_size", h))
            if (h, w) != (expected, expected):
                raise AssertionError(f"{src}: size {(w, h)} != metadata image_size {expected}")
            if strict:
                full_verify_image(src)

            png_size = osp.getsize(src)
            ti = tar_info(f"{key}.png", png_size)
            with open(src, "rb") as fh:
                tar.addfile(ti, fh)
            total += png_size

            js = sample_json(record, h, w)
            ti = tar_info(f"{key}.json", len(js))
            tar.addfile(ti, io.BytesIO(js))
            total += len(js)

            if latents_dir is not None:
                npy = osp.join(latents_dir, f"{key}.npy")
                if not osp.exists(npy):
                    raise FileNotFoundError(f"missing latent {npy}")
                npy_size = osp.getsize(npy)
                ti = tar_info(f"{key}.npy", npy_size)
                with open(npy, "rb") as fh:
                    tar.addfile(ti, fh)
                total += npy_size
    os.replace(tmp, path)
    return len(items), total


def write_wids_meta(out_dir: str, shard_entries, name: str = "SANA-Pixel-Dataset") -> str:
    doc = {
        "name": name,
        "__kind__": "SANA-WebDataset",
        "wids_version": 1,
        "shardlist": shard_entries,
    }
    path = osp.join(out_dir, "wids-meta.json")
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return path


def iter_tar_members(path: str):
    with tarfile.open(path, "r") as tar:
        for m in tar:
            if m.isfile():
                yield m.name, m


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def do_verify(args, root: str, out_dir: str, records=None) -> int:
    """Shard-level checks: AC4 (counts/groups) + AC5 (byte-exact png)."""
    meta_path = osp.join(out_dir, "wids-meta.json")
    with open(meta_path, encoding="utf-8") as f:
        spec = json.load(f)
    log(f"[verify] {meta_path}: wids_version={spec.get('wids_version')} kind={spec.get('__kind__')}")
    shardlist = spec["shardlist"]
    nsamples_total = sum(s["nsamples"] for s in shardlist)
    log(f"[verify] {len(shardlist)} shards, nsamples total = {nsamples_total}")
    if nsamples_total != len(records):
        raise AssertionError(f"sum(nsamples)={nsamples_total} != {len(records)} records")

    by_key = {}
    dupes = []
    counts = Counter()
    has_npy = None
    for entry in shardlist:
        tpath = entry["url"]
        if not osp.isabs(tpath):
            tpath = osp.abspath(osp.join(out_dir, tpath))
        if not osp.exists(tpath):
            raise FileNotFoundError(tpath)
        keys = set()
        n_png = n_json = n_npy = 0
        for name, member in iter_tar_members(tpath):
            if "/" in name:
                raise AssertionError(f"member name has a directory component: {name!r}")
            if len(name.encode()) > 100:
                raise AssertionError(f"member name exceeds USTAR 100 bytes: {name!r}")
            key, ext = osp.splitext(name)
            keys.add(key)
            counts[ext] += 1
            if ext == ".png":
                n_png += 1
            elif ext == ".json":
                n_json += 1
            elif ext == ".npy":
                n_npy += 1
            else:
                raise AssertionError(f"unexpected extension in tar: {name!r}")
        if has_npy is None:
            has_npy = n_npy > 0
        per_key = 3 if has_npy else 2
        if not (n_png == n_json == entry["nsamples"]) or (has_npy and n_npy != entry["nsamples"]):
            raise AssertionError(
                f"{osp.basename(tpath)}: png={n_png} json={n_json} npy={n_npy} nsamples={entry['nsamples']}"
            )
        assert len(keys) == entry["nsamples"], (
            f"{osp.basename(tpath)}: {len(keys)} keys != nsamples {entry['nsamples']}"
        )
        for k in keys:
            if k in by_key:
                dupes.append(k)
            by_key[k] = tpath
        log(
            f"[verify]   {osp.basename(tpath)}: {len(keys)} keys, {per_key} files/key, "
            f"{human(entry.get('filesize', osp.getsize(tpath)))}"
        )

    if dupes:
        raise AssertionError(f"{len(dupes)} duplicate keys across shards, e.g. {dupes[:5]}")
    if len(by_key) != len(records):
        raise AssertionError(f"union of keys = {len(by_key)} != {len(records)} records")
    log(f"[verify] OK: {len(by_key)} unique keys, extensions {dict(counts)}")

    # AC5: byte-exact png for a random sample.
    n_check = min(args.check_samples, len(records))
    sample = random.Random(1234).sample(records, n_check)
    bad = 0
    for record in sample:
        key = key_for(record)
        tpath = by_key[key]
        src = rec_path(root, record)
        with tarfile.open(tpath, "r") as tar:
            got = tar.extractfile(f"{key}.png").read()
        if sha256_bytes(got) != sha256_file(src):
            bad += 1
            log(f"[verify]   SHA MISMATCH {key}")
        # the json must carry the required keys and the right prompt
        with tarfile.open(tpath, "r") as tar:
            js = json.loads(tar.extractfile(f"{key}.json").read())
        for req in JSON_REQUIRED:
            if req not in js:
                raise AssertionError(f"{key}.json missing {req!r}")
        if js["prompt"] != record["prompt"]:
            raise AssertionError(f"{key}.json prompt != metadata prompt")
        if (js["height"], js["width"]) != (record["image_size"], record["image_size"]):
            raise AssertionError(f"{key}.json height/width wrong: {js['height']}x{js['width']}")
    if bad:
        raise AssertionError(f"{bad}/{n_check} png sha256 mismatches")
    log(f"[verify] OK: {n_check} sampled png byte-identical to images/ + json fields correct")

    if has_npy:
        import numpy as np

        record = sample[0]
        key = key_for(record)
        with tarfile.open(by_key[key], "r") as tar:
            z = np.load(io.BytesIO(tar.extractfile(f"{key}.npy").read()))
        log(f"[verify] OK: latent {key}.npy shape={z.shape} dtype={z.dtype}")
        if z.shape != (32, 32, 32):
            raise AssertionError(f"latent shape {z.shape} != (32,32,32)")
    return 0


def do_pack(args) -> int:
    root = osp.abspath(osp.expanduser(args.dataset_root))
    out_dir = osp.abspath(osp.expanduser(args.out_dir))
    shards_dir = osp.join(out_dir, "shards")
    doc, records = load_metadata(root)
    log(f"[pack] dataset root: {root}")
    log(f"[pack] records: {len(records)}  (metadata.json stats: {doc.get('stats', {}).get('classes')})")
    if args.limit is not None:
        records = records[: args.limit]
        log(f"[pack] --limit {args.limit}: packing only the first {len(records)} records")

    if args.verify_only:
        return do_verify(args, root, out_dir, records)

    latents_dir = osp.abspath(osp.expanduser(args.latents_dir)) if args.latents_dir else None
    if latents_dir is not None:
        missing = [key_for(r) for r in records if not osp.exists(osp.join(latents_dir, f"{key_for(r)}.npy"))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} latents missing in {latents_dir}, e.g. {missing[:3]}")
        log(f"[pack] embedding latents from {latents_dir}")

    ordered = plan_order(records, args.shuffle_seed, not args.no_shuffle)
    per = args.samples_per_shard
    num_shards = math.ceil(len(ordered) / per)
    if num_shards > LRU_SIZE and not args.allow_many_shards:
        raise SystemExit(
            f"[pack] refusing to write {num_shards} shards: SanaWebDataset hard-codes lru_size={LRU_SIZE}. "
            f"Raise --samples-per-shard to {math.ceil(len(ordered) / LRU_SIZE)} or pass --allow-many-shards."
        )
    if args.no_shuffle:
        log("[pack] WARNING: --no-shuffle keeps the class/topic-clustered global_index order")
    else:
        log(f"[pack] deterministic shuffle with seed={args.shuffle_seed} (re-pack with the same seed!)")

    os.makedirs(shards_dir, exist_ok=True)
    entries = []
    for si in range(num_shards):
        chunk = ordered[si * per : (si + 1) * per]
        if not chunk:
            continue
        name = f"shard-{si:05d}.tar"
        path = osp.join(shards_dir, name)
        n, nbytes = write_shard(path, chunk, root, latents_dir, args.strict)
        classes = Counter(r["class"] for r in chunk)
        entries.append({"url": f"shards/{name}", "nsamples": n, "filesize": osp.getsize(path)})
        log(
            f"[pack]   {name}: idx {si * per}..{si * per + n - 1}  {n} samples  "
            f"{human(osp.getsize(path))}  classes={dict(classes.most_common(4))}"
        )

    meta_path = write_wids_meta(out_dir, entries)
    log(f"[pack] wrote {meta_path} ({len(entries)} shards, {sum(e['nsamples'] for e in entries)} samples)")

    # A previous pack with more shards would leave *.tar files this wids-meta.json does
    # not reference; they are harmless but misleading, so call them out.
    referenced = {osp.basename(e["url"]) for e in entries}
    stale = sorted(set(os.listdir(shards_dir)) - referenced - {"wids-meta.json"})
    stale = [f for f in stale if f.endswith(".tar")]
    if stale:
        log(f"[pack] WARNING: {len(stale)} unreferenced tar(s) left in {shards_dir}: {stale[:5]}")

    log("[pack] verifying what was just written ...")
    do_verify(args, root, out_dir, records)
    log("[pack] done")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=DEFAULT_ROOT)
    ap.add_argument("--out-dir", default=None, help=f"default: <dataset-root>/webdataset")
    ap.add_argument("--samples-per-shard", type=int, default=1000)
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--no-shuffle", action="store_true", help="keep metadata order (not recommended)")
    ap.add_argument("--latents-dir", default=None, help="dir of <key>.npy to embed as <key>.npy")
    ap.add_argument("--strict", action="store_true", help="full PIL verify() on every source png")
    ap.add_argument("--limit", type=int, default=None, help="pack only the first N records (smoke test)")
    ap.add_argument("--verify-only", action="store_true", help="only verify existing shards")
    ap.add_argument("--check-samples", type=int, default=50, help="png sha256 spot-check count")
    ap.add_argument("--allow-many-shards", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.out_dir is None:
        args.out_dir = osp.join(osp.abspath(osp.expanduser(args.dataset_root)), "webdataset")
    if args.samples_per_shard <= 0:
        raise SystemExit("--samples-per-shard must be > 0")
    return do_pack(args)


if __name__ == "__main__":
    sys.exit(main())