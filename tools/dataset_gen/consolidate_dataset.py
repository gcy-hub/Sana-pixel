#!/usr/bin/env python
"""Consolidate SANA-Pixel-Dataset.

Turns the generation bookkeeping under ``<root>/metadata/`` into a single
``<root>/metadata.json`` and (optionally) removes the now-redundant
``metadata/`` and ``logs/`` directories.

The CSV is the user-facing source of truth; ``all.jsonl`` is used as an
independent oracle for both value and *type* recovery.  Nothing is written or
deleted before the full 10000 x 25 field equivalence check passes.

Usage::

    # 1) check only, touch nothing
    python tools/dataset_gen/consolidate_dataset.py --dry-run

    # 2) write metadata.json + verify (no deletion)
    python tools/dataset_gen/consolidate_dataset.py

    # 3) write + verify, then delete logs/ and metadata/ (irreversible)
    python tools/dataset_gen/consolidate_dataset.py --purge

    # 2b) after a NEW variant run (e.g. `generate_t2i_dataset.py --repeats 2 --variant 2`),
    #     union the already-published metadata.json with this run's metadata/manifest.csv
    python tools/dataset_gen/consolidate_dataset.py --merge
    python tools/dataset_gen/consolidate_dataset.py --merge --purge
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import shutil
import sys
from collections import Counter

DEFAULT_ROOT = "/home/ganchangyi/dataset/SANA-Pixel-Dataset"

# Fields copied out of run_config.json into metadata.json["generation"].
# The per-worker "workers" array is deliberately dropped: it is resume
# bookkeeping (pending/skipped counters), not dataset provenance.
GENERATION_KEYS = [
    "run_id",
    "finished_at",
    "dataset_root",
    "config_path",
    "config_sha256",
    "model_path",
    "model_size_bytes",
    "model_mtime",
    "gpu_ids",
    "num_shards",
    "seed_base",
    "seed_rule",
    "files_filter",
    "limit",
    "expected_images",
    "manifest_records",
    "images_on_disk",
    "failed_records",
    "elapsed_seconds",
    "python",
    "torch",
    "cuda",
    "gpu_names",
    "python_executable",
    "env",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def cast_value(raw: str, proto):
    """Cast a CSV string to the python type observed in all.jsonl."""
    if isinstance(proto, bool):
        return raw.strip().lower() in ("1", "true", "yes")
    if isinstance(proto, int):
        return int(raw)
    if isinstance(proto, float):
        return float(raw)
    return raw


def load_records_from_source(meta_dir: str):
    """Read manifest.csv, casting each column to the type seen in all.jsonl."""
    csv_path = osp.join(meta_dir, "manifest.csv")
    jsonl_path = osp.join(meta_dir, "all.jsonl")
    if not osp.exists(csv_path):
        raise FileNotFoundError(csv_path)
    if not osp.exists(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    with open(jsonl_path, encoding="utf-8") as f:
        first_line = f.readline()
    proto = json.loads(first_line)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records = []
    for row in rows:
        records.append({k: cast_value(row[k], proto[k]) for k in proto})
    return records, rows, proto


def check_against_jsonl(records, jsonl_path: str):
    """Assert, field by field, that the cast CSV equals all.jsonl."""
    expected = [json.loads(line) for line in open(jsonl_path, encoding="utf-8")]
    if len(expected) != len(records):
        raise AssertionError(f"row count mismatch: csv={len(records)} jsonl={len(expected)}")
    diffs = []
    for i, (rec, exp) in enumerate(zip(records, expected)):
        if set(rec) != set(exp):
            raise AssertionError(f"row {i}: key set mismatch {set(rec) ^ set(exp)}")
        for k, want in exp.items():
            got = rec[k]
            if str(got) != str(want):
                diffs.append((i, k, got, want))
                if len(diffs) > 20:
                    break
        if len(diffs) > 20:
            break
    if diffs:
        for i, k, got, want in diffs[:20]:
            log(f"  MISMATCH row={i} field={k!r}: csv={got!r} jsonl={want!r}")
        raise AssertionError(f"{len(diffs)} field mismatches (csv vs all.jsonl)")
    return len(expected)


def check_records(records, root: str, expect_images: bool = True):
    """Structural + on-disk checks on a record list."""
    if not records:
        raise AssertionError("no records")
    idxs = [r["global_index"] for r in records]
    if len(set(idxs)) != len(idxs):
        raise AssertionError("duplicate global_index values")
    if set(idxs) != set(range(len(records))):
        raise AssertionError(f"global_index is not 0..{len(records) - 1}")

    missing, size_mismatch, empty = [], [], []
    for r in records:
        p = r["file"] if osp.isabs(r["file"]) else osp.join(root, r["file"])
        if not osp.exists(p):
            missing.append(r["file"])
            continue
        size = osp.getsize(p)
        if size == 0:
            empty.append(r["file"])
        if "size_bytes" in r and size != int(r["size_bytes"]):
            size_mismatch.append((r["file"], size, int(r["size_bytes"])))
    if missing:
        for m in missing[:10]:
            log(f"  MISSING {m}")
        raise AssertionError(f"{len(missing)} files in metadata.json are not on disk")
    if empty:
        raise AssertionError(f"{len(empty)} zero-byte images, e.g. {empty[:5]}")
    if size_mismatch:
        for m in size_mismatch[:10]:
            log(f"  SIZE MISMATCH {m[0]}: disk={m[1]} metadata={m[2]}")
        raise AssertionError(f"{len(size_mismatch)} size_bytes mismatches")
    if expect_images:
        return {"images_checked": len(records)}
    return {}


def build_document(records, root: str, meta_dir: str, proto) -> dict:
    run_cfg_path = osp.join(meta_dir, "run_config.json")
    generation = {}
    if osp.exists(run_cfg_path):
        with open(run_cfg_path, encoding="utf-8") as f:
            run_cfg = json.load(f)
        generation = {k: run_cfg[k] for k in GENERATION_KEYS if k in run_cfg}

    classes = Counter(r["class"] for r in records)
    sizes = sorted({int(r["image_size"]) for r in records})
    return {
        "schema_version": 1,
        "dataset": "SANA-Pixel-Dataset",
        "root": root,
        "image_dir": "images",
        "generation": generation,
        "stats": {
            "records": len(records),
            "classes": dict(sorted(classes.items(), key=lambda kv: -kv[1])),
            "subclasses": len({(r["class"], r["subclass"]) for r in records}),
            "image_size": sizes[0] if len(sizes) == 1 else sizes,
            "fields": list(proto.keys()),
        },
        "records": records,
    }


def dump_readable(doc: dict, path: str) -> None:
    """Pretty header + one compact JSON object per line for `records`."""
    records = doc["records"]
    head = {k: v for k, v in doc.items() if k != "records"}
    head_txt = json.dumps(head, ensure_ascii=False, indent=2)
    assert head_txt.endswith("\n}")
    lines = [head_txt[: -len("\n}")]]
    if lines[-1].endswith(","):  # indent=2 already emits the separator
        lines[-1] = lines[-1]
    lines[-1] = lines[-1].rstrip() + ","
    lines.append('  "records": [')
    for i, rec in enumerate(records):
        sep = "," if i < len(records) - 1 else ""
        lines.append("    " + json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + sep)
    lines.append("  ]")
    lines.append("}")
    text = "\n".join(lines) + "\n"

    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def dir_size(path: str) -> int:
    total = 0
    for r, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += osp.getsize(osp.join(r, fn))
            except OSError:
                pass
    return total


def human(n: int) -> str:
    for unit in ("B", "K", "M", "G", "T"):
        if n < 1024 or unit == "T":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024.0
    return f"{n}B"


def purge(root: str, targets) -> None:
    for name in targets:
        p = osp.join(root, name)
        if not osp.exists(p):
            log(f"  skip (absent): {name}")
            continue
        size = dir_size(p) if osp.isdir(p) else osp.getsize(p)
        shutil.rmtree(p) if osp.isdir(p) else os.remove(p)
        log(f"  removed: {name} ({human(size)})")


def count_pngs(root: str) -> int:
    n = 0
    for _r, _d, files in os.walk(osp.join(root, "images")):
        n += sum(1 for f in files if f.endswith(".png"))
    return n


# Fields carried over when summarising each variant's generation run.
RUN_SUMMARY_KEYS = (
    "run_id", "finished_at", "config_path", "config_sha256", "model_path", "model_size_bytes",
    "seed_base", "seed_rule", "repeats", "variant_filter", "gpu_ids", "num_shards",
    "expected_images", "manifest_records", "images_on_disk", "failed_records", "elapsed_seconds",
    "python", "torch", "cuda", "gpu_names", "python_executable", "env",
)


def summarize_run(gen: dict) -> dict:
    return {k: gen[k] for k in RUN_SUMMARY_KEYS if k in gen}


def merge_records(old_records, new_records):
    """Combine already-published records with a fresh single-variant run.

    A new run produced by ``generate_t2i_dataset.py --repeats K --variant V`` assigns
    ``global_index = (V-1)*n_prompts + prompt_index``, which makes the variant and
    prompt_index of the **old** records derivable and lets us cross-check that the old and
    the new copy of a prompt really are the same prompt (same ``source_id``/text).

    Returns ``(merged_records_sorted_by_global_index, n_prompts, max_variant)``.
    """
    if not new_records:
        raise AssertionError("no new records to merge")
    variants = {int(r["variant"]) for r in new_records if r.get("variant") is not None}
    if len(variants) != 1:
        raise AssertionError(f"the new run must cover exactly one variant, got {sorted(variants)}")
    v_new = variants.pop()
    if v_new < 2:
        raise AssertionError(f"the new run is variant {v_new}; expected a fresh variant >= 2")

    n_prompts = len({r["source_id"] for r in new_records})
    expected_idx = {(v_new - 1) * n_prompts + i for i in range(n_prompts)}
    got_idx = {int(r["global_index"]) for r in new_records}
    if got_idx != expected_idx:
        raise AssertionError(
            f"new run indices are not (variant-1)*n_prompts + prompt_index "
            f"({len(got_idx ^ expected_idx)} mismatching global_index values)"
        )
    log(f"[merge] new run: variant {v_new}, {len(new_records)} records, n_prompts={n_prompts}")

    def normalize(rec, source):
        gid = int(rec["global_index"])
        variant, prompt_index = gid // n_prompts + 1, gid % n_prompts
        declared = rec.get("variant")
        if declared is not None and int(declared) != variant:
            raise AssertionError(f"{source}: global_index={gid} implies variant {variant}, record says {declared}")
        out = dict(rec)
        out["variant"] = variant
        out["prompt_index"] = prompt_index
        return out

    merged = [normalize(r, "metadata.json") for r in old_records]
    merged += [normalize(r, "new manifest") for r in new_records]

    seen = set()
    for r in merged:
        gid = int(r["global_index"])
        if gid in seen:
            raise AssertionError(f"duplicate global_index {gid} across old and new records")
        seen.add(gid)

    by_prompt = {}
    for r in merged:
        by_prompt.setdefault(int(r["prompt_index"]), []).append(r)
    for pi, group in sorted(by_prompt.items()):
        if len({r["source_id"] for r in group}) != 1:
            raise AssertionError(f"prompt_index {pi}: source_id differs between variants")
        if len({r["prompt"] for r in group}) != 1:
            raise AssertionError(f"prompt_index {pi}: prompt text differs between variants")

    expected_pairs = {(v, i) for v in range(1, v_new + 1) for i in range(n_prompts)}
    have_pairs = {(int(r["variant"]), int(r["prompt_index"])) for r in merged}
    missing = expected_pairs - have_pairs
    if missing:
        raise AssertionError(f"{len(missing)} (variant, prompt_index) pairs missing, e.g. {sorted(missing)[:5]}")

    merged.sort(key=lambda r: int(r["global_index"]))
    log(f"[merge] OK: {len(merged)} records = {n_prompts} prompts x {v_new} variants, "
        f"seeds [{min(r['seed'] for r in merged)}..{max(r['seed'] for r in merged)}] "
        f"({len({r['seed'] for r in merged})} distinct)")
    return merged, n_prompts, v_new


def apply_merged_generation(doc, old_doc, n_prompts, max_variant, root):
    """Replace the single-run `generation` block with a per-variant one."""
    new_gen = doc.get("generation", {})
    old_gen = (old_doc or {}).get("generation", {})
    runs = {"1": summarize_run(old_gen)}
    runs[str(max_variant)] = summarize_run(new_gen)
    doc["generation"] = {
        "dataset_root": root,
        "n_prompts": n_prompts,
        "repeats": max_variant,
        "seed_base": new_gen.get("seed_base", old_gen.get("seed_base", 0)),
        "seed_rule": new_gen.get("seed_rule", old_gen.get("seed_rule")),
        "expected_images": n_prompts * max_variant,
        "manifest_records": len(doc["records"]),
        "images_on_disk": count_pngs(root),
        "failed_records": sum(int(r.get("failed_records") or 0) for r in runs.values()),
        "cluster": {k: new_gen.get(k) for k in ("python", "torch", "cuda", "gpu_names",
                                                "python_executable", "env")},
        "variant_runs": runs,
    }
    doc["stats"]["n_prompts"] = n_prompts
    doc["stats"]["variants"] = {
        str(v): sum(1 for r in doc["records"] if int(r["variant"]) == v) for v in range(1, max_variant + 1)
    }
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default=DEFAULT_ROOT)
    ap.add_argument("--dry-run", action="store_true", help="verify only; never write or delete")
    ap.add_argument("--purge", action="store_true", help="after a successful verify+write, delete logs/ and metadata/")
    ap.add_argument("--merge", action="store_true",
                    help="union the existing metadata.json (older variants) with the records of this "
                         "generation run (metadata/manifest.csv); use after --variant V runs")
    args = ap.parse_args()

    root = osp.abspath(osp.expanduser(args.dataset_root))
    meta_dir = osp.join(root, "metadata")
    out_path = osp.join(root, "metadata.json")
    log(f"[consolidate] dataset root: {root}")

    old_doc = None
    old_records = []
    if args.merge:
        if not osp.exists(out_path):
            log(f"[consolidate] ERROR: --merge needs an existing {out_path}")
            return 2
        with open(out_path, encoding="utf-8") as f:
            old_doc = json.load(f)
        old_records = old_doc["records"]
        log(f"[consolidate] --merge: loaded {len(old_records)} already-published records")
        if not osp.isdir(meta_dir):
            log("[consolidate] ERROR: --merge needs this run's metadata/ directory")
            return 2

    if osp.isdir(meta_dir):
        log("[consolidate] source: metadata/manifest.csv (+ all.jsonl as oracle)")
        records, rows, proto = load_records_from_source(meta_dir)
        log(f"[consolidate] parsed {len(records)} csv rows, {len(proto)} fields")

        n = check_against_jsonl(records, osp.join(meta_dir, "all.jsonl"))
        log(f"[consolidate] OK: csv == all.jsonl for {n} rows x {len(proto)} fields")

        n_prompts, max_variant = len(records), 1
        if args.merge:
            records, n_prompts, max_variant = merge_records(old_records, records)
            # expose prompt_index in the record schema right after variant
            proto = {k: v for k, v in proto.items() if k != "prompt_index"}
            ordered = {}
            for k, v in proto.items():
                ordered[k] = v
                if k == "variant":
                    ordered["prompt_index"] = 0
            proto = ordered

        check_records(records, root)
        log("[consolidate] OK: every `file` exists, no zero-byte, size_bytes matches")

        pngs = count_pngs(root)
        if len(records) != pngs:
            hint = "" if args.merge else " (did you forget --merge after a --variant run?)"
            raise AssertionError(f"{len(records)} records but {pngs} png on disk{hint}")
        log(f"[consolidate] OK: {len(records)} records == {pngs} png on disk")

        doc = build_document(records, root, meta_dir, proto)
        if args.merge:
            doc = apply_merged_generation(doc, old_doc, n_prompts, max_variant, root)
        if args.dry_run:
            log("[consolidate] --dry-run: metadata.json NOT written")
        else:
            dump_readable(doc, out_path)
            log(f"[consolidate] wrote {out_path} ({human(osp.getsize(out_path))})")
    elif osp.exists(out_path):
        log("[consolidate] metadata/ absent; verifying existing metadata.json")
        with open(out_path, encoding="utf-8") as f:
            doc = json.load(f)
        records = doc["records"]
        log(f"[consolidate] loaded {len(records)} records")

    # Independent re-read of whatever is on disk now (catches a bad write).
    if osp.exists(out_path) and not args.dry_run:
        with open(out_path, encoding="utf-8") as f:
            again = json.load(f)
        if len(again["records"]) != len(records):
            raise AssertionError("re-read record count differs from source")
        if len(again["records"]) != count_pngs(root):
            raise AssertionError(
                f"metadata.json has {len(again['records'])} records but there are "
                f"{count_pngs(root)} png on disk"
            )
        check_records(again["records"], root)
        log(f"[consolidate] OK: metadata.json re-read and re-verified ({len(again['records'])} records)")

    if args.purge:
        log("[consolidate] --purge: removing generation bookkeeping")
        purge(root, ["logs", "metadata"])

    log("[consolidate] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())