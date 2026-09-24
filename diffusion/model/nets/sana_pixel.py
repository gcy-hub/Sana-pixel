# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pixel-space SANA models used for latent-to-pixel transfer.

The transformer keeps the parameter names and token grid of latent SANA.  Only
the image patch embedder is resized for RGB patches and the latent prediction
head is replaced by a shallow convolutional detailer which sees both the noisy
RGB image and the transformer feature map.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from diffusion.model.builder import MODELS
from diffusion.model.nets.sana_multi_scale import SanaMS, _xformers_available
from diffusion.model.utils import auto_grad_checkpoint


class PixelDetailerHead(nn.Module):
    """A shallow U-Net that aligns its bottleneck with the SANA token grid."""

    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        patch_size: int,
        channels: Sequence[int] = (64, 128, 256, 512, 512),
    ) -> None:
        super().__init__()
        if patch_size <= 0 or patch_size & (patch_size - 1):
            raise ValueError(f"pixel patch_size must be a positive power of two, got {patch_size}")

        levels = int(math.log2(patch_size))
        if len(channels) < levels:
            raise ValueError(f"detailer requires at least {levels} channel entries for patch_size={patch_size}")
        self.patch_size = patch_size
        self.channels = tuple(channels[:levels])

        encoders = []
        previous = in_channels
        for channel in self.channels:
            encoders.append(
                nn.Sequential(
                    nn.Conv2d(previous, channel, kernel_size=3, padding=1),
                    nn.SiLU(),
                )
            )
            previous = channel
        self.encoders = nn.ModuleList(encoders)
        self.pools = nn.ModuleList(nn.MaxPool2d(2, stride=2) for _ in self.channels)

        bottleneck_channels = self.channels[-1]
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(bottleneck_channels + hidden_size, bottleneck_channels, kernel_size=1),
            nn.SiLU(),
        )

        upsamplers = []
        decoders = []
        current = bottleneck_channels
        # Mirror L2P's MicroDiffusionModel channel schedule.  The extra first
        # decoder stage is what adapts its 16px design to SANA's 32px patch.
        decoder_channels = (*reversed(self.channels[:-1]), self.channels[0])
        for skip_channels, output_channels in zip(reversed(self.channels), decoder_channels):
            upsamplers.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(current, skip_channels, kernel_size=3, padding=1),
                )
            )
            decoders.append(
                nn.Sequential(
                    nn.Conv2d(skip_channels * 2, output_channels, kernel_size=3, padding=1),
                    nn.SiLU(),
                )
            )
            current = output_channels
        self.upsamplers = nn.ModuleList(upsamplers)
        self.decoders = nn.ModuleList(decoders)
        self.output = nn.Conv2d(self.channels[0], in_channels, kernel_size=1)

    @staticmethod
    def _run(module: nn.Module, *inputs: torch.Tensor) -> torch.Tensor:
        return auto_grad_checkpoint(module, *inputs)

    def forward(self, noisy_rgb: torch.Tensor, token_features: torch.Tensor) -> torch.Tensor:
        if noisy_rgb.ndim != 4 or token_features.ndim != 4:
            raise ValueError(
                f"detailer expects BCHW tensors, got noisy_rgb={noisy_rgb.shape}, features={token_features.shape}"
            )
        height, width = noisy_rgb.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"RGB size {(height, width)} must be divisible by patch_size={self.patch_size}")
        expected_feature_size = (height // self.patch_size, width // self.patch_size)
        if token_features.shape[-2:] != expected_feature_size:
            raise ValueError(
                f"token feature grid {token_features.shape[-2:]} does not match expected {expected_feature_size}"
            )

        skips = []
        hidden = noisy_rgb
        for encoder, pool in zip(self.encoders, self.pools):
            hidden = self._run(encoder, hidden)
            skips.append(hidden)
            hidden = pool(hidden)

        hidden = self._run(self.feature_fusion, torch.cat((hidden, token_features), dim=1))
        for upsampler, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            hidden = self._run(upsampler, hidden)
            if hidden.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError(f"detailer skip shape mismatch: up={hidden.shape}, skip={skip.shape}")
            hidden = self._run(decoder, torch.cat((hidden, skip), dim=1))
        return self.output(hidden)


@MODELS.register_module()
class SanaMSPixel(SanaMS):
    """SANA multi-scale backbone with RGB input and RGB velocity output."""

    def __init__(
        self,
        *args,
        patch_size: int = 32,
        in_channels: int = 3,
        detailer_channels: Sequence[int] = (64, 128, 256, 512, 512),
        **kwargs,
    ) -> None:
        if in_channels != 3:
            raise ValueError(f"SanaMSPixel is an RGB model and requires in_channels=3, got {in_channels}")
        kwargs["pred_sigma"] = False
        kwargs["learn_sigma"] = False
        super().__init__(*args, patch_size=patch_size, in_channels=in_channels, **kwargs)

        # The latent final layer is intentionally absent from pixel checkpoints.
        del self.final_layer
        self.detailer = PixelDetailerHead(
            in_channels=in_channels,
            hidden_size=self.hidden_size,
            patch_size=self.patch_size,
            channels=detailer_channels,
        )

    @property
    def data_space(self) -> str:
        return "pixel"

    def configure_l2p_trainable(self, first_n: int = 5, last_n: int = 5) -> dict[str, int]:
        """Freeze middle transformer blocks while leaving both interfaces trainable."""
        depth = len(self.blocks)
        if first_n < 0 or last_n < 0 or first_n + last_n > depth:
            raise ValueError(f"invalid L2P boundary first_n={first_n}, last_n={last_n}, depth={depth}")

        self.requires_grad_(True)
        frozen_start = first_n
        frozen_end = depth - last_n
        for index in range(frozen_start, frozen_end):
            self.blocks[index].requires_grad_(False)
            self.blocks[index].eval()

        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        frozen = sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for block in self.blocks:
                if not any(parameter.requires_grad for parameter in block.parameters()):
                    block.eval()
        return self

    def forward(self, x, timestep, y, mask=None, data_info=None, return_logvar=False, jvp=False, **kwargs):
        if return_logvar:
            raise ValueError("SanaMSPixel does not predict learned variance/logvar")
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(f"SanaMSPixel expects RGB BCHW input, got {x.shape}")
        if x.shape[-2] % self.patch_size or x.shape[-1] % self.patch_size:
            raise ValueError(f"input size {x.shape[-2:]} must be divisible by patch_size={self.patch_size}")

        batch_size = x.shape[0]
        noisy_rgb = x.to(self.dtype)
        if self.timestep_norm_scale_factor != 1.0:
            timestep = (timestep.float() / self.timestep_norm_scale_factor).to(torch.float32)
        else:
            timestep = timestep.long().to(torch.float32)
        y = y.to(self.dtype)

        self.h, self.w = noisy_rgb.shape[-2] // self.patch_size, noisy_rgb.shape[-1] // self.patch_size
        tokens = self.x_embedder(noisy_rgb)
        image_pos_embed = None
        if self.use_pe:
            tokens, image_pos_embed = self._apply_positional_embedding(tokens, batch_size)

        time_embedding = self.t_embedder(timestep)
        if self.cfg_embedder:
            time_embedding += self.cfg_embedder(data_info["cfg_scale"] * self.cfg_embed_scale)
        modulation = self.t_block(time_embedding)

        captions = self.y_embedder(y, self.training, mask=mask)
        if self.y_norm:
            captions = self.attention_y_norm(captions)
        if mask is None:
            if not _xformers_available:
                raise ValueError(f"Attention type is not available due to _xformers_available={_xformers_available}.")
            caption_lengths = [captions.shape[2]] * captions.shape[0]
            captions = captions.squeeze(1).view(1, -1, tokens.shape[-1])
        else:
            mask = mask.to(torch.int16)
            if mask.shape[0] != captions.shape[0]:
                mask = mask.repeat(captions.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            if _xformers_available:
                captions = captions.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, tokens.shape[-1])
                caption_lengths = mask.sum(dim=1).tolist()
            else:
                caption_lengths = mask

        for block in self.blocks:
            if jvp:
                tokens = block(
                    tokens,
                    captions,
                    modulation,
                    caption_lengths,
                    (self.h, self.w),
                    image_pos_embed,
                    **kwargs,
                )
            else:
                tokens = auto_grad_checkpoint(
                    block,
                    tokens,
                    captions,
                    modulation,
                    caption_lengths,
                    (self.h, self.w),
                    image_pos_embed,
                    **kwargs,
                )

        feature_map = tokens.reshape(batch_size, self.h, self.w, self.hidden_size).permute(0, 3, 1, 2)
        return self.detailer(noisy_rgb, feature_map)

    def forward_with_dpmsolver(self, x, timestep, y, data_info=None, **kwargs):
        return self.forward(x, timestep, y, data_info=data_info, **kwargs)


@MODELS.register_module()
def SanaMSPixel_1600M_P32_D20(**kwargs):
    return SanaMSPixel(depth=20, hidden_size=2240, patch_size=32, num_heads=20, **kwargs)
