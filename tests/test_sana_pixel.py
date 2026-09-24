import unittest

import torch

from diffusion.model.nets.sana_pixel import PixelDetailerHead, SanaMSPixel
from diffusion.model.utils import set_grad_checkpoint
from diffusion.scheduler.iddpm import Scheduler
from tools.convert_sana_to_pixel import convert_state_dict


class TestSanaPixel(unittest.TestCase):
    @staticmethod
    def _tiny_model() -> SanaMSPixel:
        return SanaMSPixel(
            input_size=64,
            patch_size=8,
            in_channels=3,
            hidden_size=64,
            depth=4,
            num_heads=4,
            mlp_ratio=2,
            caption_channels=32,
            model_max_length=8,
            class_dropout_prob=0.0,
            attn_type="vanilla",
            cross_attn_type="vanilla",
            ffn_type="mlp",
            use_pe=False,
            qk_norm=False,
            y_norm=False,
            detailer_channels=(8, 16, 32),
        )

    def test_rectangular_forward_backward_and_freeze_boundary(self):
        model = self._tiny_model()
        set_grad_checkpoint(model)
        report = model.configure_l2p_trainable(first_n=1, last_n=1)
        self.assertEqual(report["total"], report["trainable"] + report["frozen"])
        self.assertGreater(report["frozen"], 0)

        noisy_rgb = torch.randn(2, 3, 64, 96)
        timestep = torch.tensor([10, 900])
        captions = torch.randn(2, 1, 8, 32)
        mask = torch.ones(2, 1, 1, 8)
        output = model(noisy_rgb, timestep, captions, mask=mask)
        self.assertEqual(output.shape, noisy_rgb.shape)

        output.square().mean().backward()
        self.assertIsNotNone(model.blocks[0].attn.qkv.weight.grad)
        self.assertIsNone(model.blocks[1].attn.qkv.weight.grad)
        self.assertIsNone(model.blocks[2].attn.qkv.weight.grad)
        self.assertIsNotNone(model.blocks[3].attn.qkv.weight.grad)
        self.assertIsNotNone(model.x_embedder.proj.weight.grad)
        self.assertIsNotNone(model.detailer.encoders[0][0].weight.grad)
        self.assertIsNotNone(model.detailer.output.weight.grad)

    def test_detailer_rejects_misaligned_feature_grid(self):
        detailer = PixelDetailerHead(3, 64, patch_size=8, channels=(8, 16, 32))
        with self.assertRaisesRegex(ValueError, "token feature grid"):
            detailer(torch.randn(1, 3, 64, 64), torch.randn(1, 64, 7, 8))

    def test_checkpoint_conversion_copies_only_compatible_tensors(self):
        source = {
            "x_embedder.proj.weight": torch.randn(64, 32, 1, 1),
            "x_embedder.proj.bias": torch.randn(64),
            "final_layer.linear.weight": torch.randn(32, 64),
            "final_layer.linear.bias": torch.randn(32),
            "blocks.0.attn.qkv.weight": torch.randn(3, 3),
            "blocks.19.attn.qkv.weight": torch.randn(3, 3),
            "t_embedder.mlp.0.weight": torch.randn(4, 4),
        }
        converted, report = convert_state_dict(
            source,
            hidden_size=64,
            patch_size=8,
            detailer_channels=(8, 16, 32),
        )

        self.assertNotIn("final_layer.linear.weight", converted)
        self.assertEqual(converted["x_embedder.proj.weight"].shape, (64, 3, 8, 8))
        self.assertIn("detailer.output.weight", converted)
        copied = converted["t_embedder.mlp.0.weight"]
        self.assertEqual(copied.data_ptr(), source["t_embedder.mlp.0.weight"].data_ptr())
        self.assertEqual(report["discarded_tensor_count"], 4)

    def test_linear_flow_uses_velocity_target_without_extra_gamma(self):
        scheduler = Scheduler(
            "1000",
            noise_schedule="linear_flow",
            predict_flow_v=True,
            learn_sigma=False,
            pred_sigma=False,
            flow_shift=3.0,
        )
        clean = torch.tensor([[[[-0.5, 0.25], [0.75, -1.0]]]]).repeat(1, 3, 1, 1)
        noise = torch.tensor([[[[0.5, -0.25], [-0.75, 1.0]]]]).repeat(1, 3, 1, 1)

        class ZeroModel(torch.nn.Module):
            def forward(self, x, timestep, **kwargs):
                return torch.zeros_like(x)

        terms = scheduler.training_losses(ZeroModel(), clean, torch.tensor([500]), noise=noise)
        expected = (noise - clean).square().flatten(1).mean(1)
        torch.testing.assert_close(terms["loss"], expected)


if __name__ == "__main__":
    unittest.main()
