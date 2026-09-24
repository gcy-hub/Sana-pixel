#!/usr/bin/env python
"""Strict-load and optionally run a full SANA-Pixel forward/backward smoke test."""

from __future__ import annotations

import argparse
import gc
import json
import os

import pyrallis
import torch

from diffusion.model.builder import build_model
from diffusion.utils.config import SanaConfig, model_init_config


DEFAULT_CONFIG = "configs/sana_pixel/Sana_1600M_1024px_webdataset_bf16_lr2e5.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--caption-length", type=int, default=8)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--optimizer-step", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.config, encoding="utf-8") as handle:
        config = pyrallis.load(SanaConfig, handle)
    checkpoint_path = args.checkpoint or config.model.load_from
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    if args.image_size % 32:
        raise ValueError("image size must be divisible by the 32px patch size")

    model = build_model(
        config.model.model,
        use_grad_checkpoint=config.train.grad_checkpointing,
        use_fp32_attention=config.model.fp32_attention,
        null_embed_path=None,
        **model_init_config(config, latent_size=args.image_size),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=True, assign=True)
    del state_dict, checkpoint
    gc.collect()

    freeze_report = model.configure_l2p_trainable(
        first_n=config.model.l2p_first_trainable_blocks,
        last_n=config.model.l2p_last_trainable_blocks,
    )
    report = {
        "checkpoint": os.path.abspath(checkpoint_path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": freeze_report["trainable"],
        "frozen_parameter_count": freeze_report["frozen"],
    }

    run_backward = args.backward or args.optimizer_step
    if args.forward or run_backward:
        if not torch.cuda.is_available() and args.device.startswith("cuda"):
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device(args.device)
        # Accelerate mixed precision keeps master parameters/optimizer states
        # in FP32 and autocasts compute to BF16.  Mirror that setup here.
        parameter_dtype = torch.float32
        compute_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        model = model.to(device=device, dtype=parameter_dtype).train(run_backward)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        noisy_rgb = torch.randn(1, 3, args.image_size, args.image_size, device=device, dtype=parameter_dtype)
        timestep = torch.tensor([500], device=device)
        captions = torch.randn(
            1,
            1,
            args.caption_length,
            config.text_encoder.caption_channels,
            device=device,
            dtype=parameter_dtype,
        )
        mask = torch.ones(1, 1, 1, args.caption_length, device=device)
        optimizer = None
        trainable_before = frozen_before = None
        if args.optimizer_step:
            optimizer = torch.optim.AdamW(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                lr=config.train.optimizer["lr"],
                betas=tuple(config.train.optimizer["betas"]),
                eps=config.train.optimizer["eps"],
                weight_decay=config.train.optimizer["weight_decay"],
            )
            trainable_before = model.blocks[0].attn.qkv.weight[0, 0].detach().clone()
            interface_before = model.detailer.output.bias[0].detach().clone()
            frozen_before = model.blocks[len(model.blocks) // 2].attn.qkv.weight[0, 0].detach().clone()

        with torch.set_grad_enabled(run_backward):
            with torch.autocast(
                device_type=device.type,
                dtype=compute_dtype,
                enabled=device.type == "cuda",
            ):
                output = model(noisy_rgb, timestep, captions, mask=mask)
                loss = output.float().square().mean()
            if run_backward:
                loss.backward()
            if optimizer is not None:
                optimizer.step()

        report.update(
            {
                "input_shape": list(noisy_rgb.shape),
                "output_shape": list(output.shape),
                "output_finite": bool(torch.isfinite(output).all().item()),
                "loss": float(loss.item()),
            }
        )
        if run_backward:
            report["first_block_has_grad"] = any(
                parameter.grad is not None for parameter in model.blocks[0].parameters()
            )
            report["middle_block_has_grad"] = any(
                parameter.grad is not None for parameter in model.blocks[len(model.blocks) // 2].parameters()
            )
            report["last_block_has_grad"] = any(
                parameter.grad is not None for parameter in model.blocks[-1].parameters()
            )
            report["detailer_has_grad"] = any(
                parameter.grad is not None for parameter in model.detailer.parameters()
            )
            report["detailer_first_encoder_has_grad"] = any(
                parameter.grad is not None for parameter in model.detailer.encoders[0].parameters()
            )
            report["pixel_embedder_has_grad"] = model.x_embedder.proj.weight.grad is not None
            report["all_trainable_grads_finite"] = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        if optimizer is not None:
            report["optimizer_step"] = True
            report["trainable_probe_changed"] = bool(
                model.blocks[0].attn.qkv.weight[0, 0].detach() != trainable_before
            )
            report["interface_probe_changed"] = bool(
                model.detailer.output.bias[0].detach() != interface_before
            )
            report["frozen_probe_changed"] = bool(
                model.blocks[len(model.blocks) // 2].attn.qkv.weight[0, 0].detach() != frozen_before
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            report["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
