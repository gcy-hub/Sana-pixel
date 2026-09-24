#!/usr/bin/env python
"""Convert a latent SANA-1.5 checkpoint into a pixel-space initialization.

The transformer/time/text-conditioning weights are copied exactly.  The
latent patch embedder and latent output head are deliberately discarded; a
new RGB patch embedder and Pixel Detailer Head are initialized deterministically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from diffusion.model.nets.sana_pixel import PixelDetailerHead


DEFAULT_SOURCE = (
    "/home/ganchangyi/huggingface_ckpts/SANA1.5_1.6B_1024px/"
    "checkpoints/SANA1.5_1.6B_1024px.pth"
)
DEFAULT_OUTPUT = (
    "/home/ganchangyi/huggingface_ckpts/SANA1.5_1.6B_1024px_pixel_init/"
    "checkpoints/SANA1.5_1.6B_1024px_pixel_init.pth"
)

DISCARDED_PREFIXES = ("x_embedder.", "final_layer.")


def sha256_file(path: str, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_pixel_interfaces(
    *,
    seed: int,
    hidden_size: int,
    patch_size: int,
    detailer_channels: tuple[int, ...],
) -> OrderedDict[str, torch.Tensor]:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        patch_embedder = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True)
        nn.init.xavier_uniform_(patch_embedder.weight.view(patch_embedder.weight.shape[0], -1))
        nn.init.zeros_(patch_embedder.bias)
        detailer = PixelDetailerHead(
            in_channels=3,
            hidden_size=hidden_size,
            patch_size=patch_size,
            channels=detailer_channels,
        )

    state = OrderedDict()
    state["x_embedder.proj.weight"] = patch_embedder.weight.detach().cpu()
    state["x_embedder.proj.bias"] = patch_embedder.bias.detach().cpu()
    for key, value in detailer.state_dict().items():
        state[f"detailer.{key}"] = value.detach().cpu()
    return state


def convert_state_dict(
    source_state: dict[str, torch.Tensor],
    *,
    seed: int = 1,
    hidden_size: int = 2240,
    patch_size: int = 32,
    detailer_channels: tuple[int, ...] = (64, 128, 256, 512, 512),
) -> tuple[OrderedDict[str, torch.Tensor], dict]:
    required = {
        "x_embedder.proj.weight",
        "final_layer.linear.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.19.attn.qkv.weight",
    }
    missing_required = sorted(required.difference(source_state))
    if missing_required:
        raise KeyError(f"source checkpoint is not SANA-1.5 1.6B/D20; missing keys: {missing_required}")

    converted = OrderedDict(
        (key, value) for key, value in source_state.items() if not key.startswith(DISCARDED_PREFIXES)
    )
    interface_state = initialize_pixel_interfaces(
        seed=seed,
        hidden_size=hidden_size,
        patch_size=patch_size,
        detailer_channels=detailer_channels,
    )
    overlap = sorted(set(converted).intersection(interface_state))
    if overlap:
        raise RuntimeError(f"new pixel interface unexpectedly overlaps copied source keys: {overlap}")
    converted.update(interface_state)

    copied_keys = [key for key in source_state if not key.startswith(DISCARDED_PREFIXES)]
    discarded_keys = [key for key in source_state if key.startswith(DISCARDED_PREFIXES)]
    exact_copy_failures = [key for key in copied_keys if converted[key].data_ptr() != source_state[key].data_ptr()]
    if exact_copy_failures:
        raise RuntimeError(f"conversion did not preserve source tensor objects: {exact_copy_failures[:10]}")

    report = {
        "format": "sana-pixel-init-v1",
        "seed": seed,
        "hidden_size": hidden_size,
        "patch_size": patch_size,
        "detailer_channels": list(detailer_channels),
        "source_tensor_count": len(source_state),
        "copied_tensor_count": len(copied_keys),
        "discarded_tensor_count": len(discarded_keys),
        "new_tensor_count": len(interface_state),
        "output_tensor_count": len(converted),
        "discarded_keys": discarded_keys,
        "new_keys": list(interface_state),
        "copied_parameter_count": sum(source_state[key].numel() for key in copied_keys),
        "new_parameter_count": sum(value.numel() for value in interface_state.values()),
    }
    return converted, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--source-sha256", action="store_true", help="also hash the 6+ GB source checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="validate and print the report without saving")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = os.path.abspath(os.path.expanduser(args.source))
    output_path = os.path.abspath(os.path.expanduser(args.output))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)
    if os.path.abspath(source_path) == os.path.abspath(output_path):
        raise ValueError("source and output checkpoint paths must differ")

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    source_state = checkpoint.get("state_dict", checkpoint)
    converted, report = convert_state_dict(source_state, seed=args.seed)
    report.update(
        {
            "source": source_path,
            "source_size_bytes": os.path.getsize(source_path),
            "output": output_path,
        }
    )
    if args.source_sha256:
        report["source_sha256"] = sha256_file(source_path)

    print(json.dumps(report, indent=2))
    if args.dry_run:
        return 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    temporary_path = f"{output_path}.tmp-{os.getpid()}"
    torch.save({"state_dict": converted, "conversion_report": report}, temporary_path)
    os.replace(temporary_path, output_path)

    report_path = f"{output_path}.conversion.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"saved pixel initialization: {output_path}")
    print(f"saved conversion report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

