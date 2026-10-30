import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "compiler" / "supra2"))

import run_reference as reference
from compile_projection_plan import choose_token_batch, projection_specs


class LatentPatchTests(unittest.TestCase):
    def test_patchify_unpatchify_round_trip(self):
        latent = torch.arange(2 * 4 * 32 * 32, dtype=torch.float32).reshape(
            2, 4, 32, 32
        )
        tokens = reference.patchify_latents(latent)
        self.assertEqual(tuple(tokens.shape), (2, 256, 16))
        torch.testing.assert_close(
            reference.unpatchify_latents(tokens, tuple(latent.shape)), latent,
            rtol=0, atol=0,
        )

    def test_rejects_incompatible_token_shape(self):
        tokens = torch.zeros(1, 255, 16)
        with self.assertRaises(ValueError):
            reference.unpatchify_latents(tokens, (1, 4, 32, 32))


class DiTComponentTests(unittest.TestCase):
    def test_timestep_embedding_shape(self):
        embedder = reference.TimestepEmbedder(576)
        result = embedder(torch.tensor([0.0, 0.5, 1.0]))
        self.assertEqual(tuple(result.shape), (3, 576))
        self.assertTrue(torch.isfinite(result).all())

    def test_self_attention_shape(self):
        attention = reference.Attention(576, 9)
        result = attention(torch.randn(2, 7, 576))
        self.assertEqual(tuple(result.shape), (2, 7, 576))
        self.assertTrue(torch.isfinite(result).all())

    def test_cross_attention_mask_and_shape(self):
        attention = reference.Attention(18, 3, ctx_dim=18)
        x = torch.randn(2, 5, 18)
        context = torch.randn(2, 4, 18)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        result = attention(x, ctx=context, ctx_mask=mask)
        self.assertEqual(tuple(result.shape), (2, 5, 18))
        self.assertTrue(torch.isfinite(result).all())

    def test_dit_block_shape(self):
        block = reference.DiTBlock(18, 3, 4.0)
        result = block(
            torch.randn(2, 5, 18),
            torch.randn(2, 18),
            torch.randn(2, 4, 18),
            torch.ones(2, 4),
        )
        self.assertEqual(tuple(result.shape), (2, 5, 18))
        self.assertTrue(torch.isfinite(result).all())


class ProjectionPlanTests(unittest.TestCase):
    def test_expected_projection_inventory(self):
        specs = projection_specs(14)
        self.assertEqual(len(specs), 118)
        self.assertEqual(reference.TOKEN_COUNT, 256)
        self.assertEqual(reference.HEADS * 64, reference.D_MODEL)

    def test_output_page_aware_token_batches(self):
        self.assertEqual(choose_token_batch(256, 576), (32, 1152))
        self.assertEqual(choose_token_batch(256, 1728), (16, 3456))
        self.assertEqual(choose_token_batch(256, 2304), (8, 4608))
        self.assertEqual(choose_token_batch(128, 1152), (16, 2304))


if __name__ == "__main__":
    unittest.main()
